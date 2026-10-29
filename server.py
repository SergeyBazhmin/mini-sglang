"""nano-serve v0: entry point, HTTP server and TokenizerManager (main process).

SGLang equivalents:
  - launch_server                      entrypoints/http_server.py (~line 3125)
  - /v1/chat/completions               entrypoints/http_server.py (~line 1871)
                                       + entrypoints/openai/serving_chat.py
  - TokenizerManager, ReqState         managers/tokenizer_manager.py
      generate_request (~907), _send_one_request (~1781),
      _stream_one_response (~1958), handle_loop (~2477)

Process layout (the same ring as SGLang):

   main: FastAPI + TokenizerManager --PUSH--> Scheduler --PUSH--> Detokenizer
              ^                                                        |
              +----------------------------PULL------------------------+

Run:  python server.py --model Qwen/Qwen3-0.6B --port 8000
"""

import argparse
import asyncio
import dataclasses
import json
import multiprocessing as mp
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

from common import (
    AbortReq,
    BatchStrOut,
    PortArgs,
    TokenizedGenerateReq,
    get_socket,
)
from detokenizer import run_detokenizer_process
from scheduler import run_scheduler_process

# ---------------------------------------------------------------------------
# TokenizerManager
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ReqState:
    """Per-request state in the main process. SGLang: tokenizer_manager.ReqState."""

    event: asyncio.Event
    out_list: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    finished: bool = False


class TokenizerManager:
    def __init__(self, model_path: str, port_args: PortArgs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        context = zmq.asyncio.Context(2)
        self.send_to_scheduler = get_socket(
            context, zmq.PUSH, port_args.scheduler_input, bind=False
        )
        self.recv_from_detokenizer = get_socket(
            context, zmq.PULL, port_args.tokenizer_input, bind=True
        )
        self.rid_to_state: Dict[str, ReqState] = {}

    def tokenize_chat(
        self, messages: List[Dict[str, Any]], chat_template_kwargs: Dict[str, Any]
    ) -> List[int]:
        # Render to text first, then tokenize. This behaves the same across
        # transformers versions (apply_chat_template's return type has changed).
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        return self.tokenizer(prompt, add_special_tokens=False).input_ids

    async def generate_request(
        self,
        input_ids: List[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Send one request to the scheduler and yield its outputs as they arrive."""
        rid = uuid.uuid4().hex
        state = ReqState(event=asyncio.Event())
        self.rid_to_state[rid] = state

        await self.send_to_scheduler.send_pyobj(
            TokenizedGenerateReq(
                rid=rid,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )

        try:
            while True:
                await state.event.wait()
                state.event.clear()
                outs, state.out_list = state.out_list, []
                for out in outs:
                    yield out
                if state.finished:
                    return
        finally:
            # Runs on normal completion and also when the client disconnects
            # (Starlette cancels the response, which closes this generator).
            if not state.finished:
                # Not awaited: we may be inside a cancellation. zmq.asyncio sends
                # right away when the socket is ready.
                self.send_to_scheduler.send_pyobj(AbortReq(rid=rid))
            self.rid_to_state.pop(rid, None)

    async def handle_loop(self):
        """ONE coroutine receives outputs for ALL requests and wakes the right waiter."""
        while True:
            recv_obj: BatchStrOut = await self.recv_from_detokenizer.recv_pyobj()
            for i, rid in enumerate(recv_obj.rids):
                state = self.rid_to_state.get(rid)
                if state is None:
                    continue  # Aborted, and the client is already gone.
                state.out_list.append(
                    {
                        "text": recv_obj.text_deltas[i],
                        "finish_reason": recv_obj.finish_reasons[i],
                        "completion_tokens": recv_obj.completion_tokens[i],
                    }
                )
                if recv_obj.finish_reasons[i] is not None:
                    state.finished = True
                state.event.set()


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP layer. It only translates the API; the engine never
# sees "messages", only token ids and sampling params.
# ---------------------------------------------------------------------------


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    # Non-standard, same as SGLang. For Qwen3: {"enable_thinking": false}
    chat_template_kwargs: Optional[Dict[str, Any]] = None


DEFAULT_MAX_TOKENS = 1024


def build_app(tm: TokenizerManager, model_name: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(tm.handle_loop())
        yield
        task.cancel()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        input_ids = tm.tokenize_chat(
            request.messages, request.chat_template_kwargs or {}
        )
        max_new_tokens = (
            request.max_completion_tokens or request.max_tokens or DEFAULT_MAX_TOKENS
        )
        outputs = tm.generate_request(
            input_ids, max_new_tokens, request.temperature, request.top_p
        )
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if request.stream:
            return StreamingResponse(
                _stream_chunks(outputs, response_id, created, model_name),
                media_type="text/event-stream",
            )

        text, finish_reason, completion_tokens = [], None, 0
        async for out in outputs:
            text.append(out["text"])
            finish_reason = out["finish_reason"]
            completion_tokens = out["completion_tokens"]
        return JSONResponse(
            {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "".join(text)},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(input_ids),
                    "completion_tokens": completion_tokens,
                    "total_tokens": len(input_ids) + completion_tokens,
                },
            }
        )

    return app


async def _stream_chunks(
    outputs: AsyncIterator[Dict[str, Any]], response_id: str, created: int, model: str
):
    def chunk(delta: Dict[str, Any], finish_reason: Optional[str] = None) -> str:
        body = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

    try:
        yield chunk({"role": "assistant", "content": ""})
        async for out in outputs:
            if out["text"]:
                yield chunk({"content": out["text"]})
            if out["finish_reason"] is not None:
                yield chunk({}, finish_reason=out["finish_reason"])
        yield "data: [DONE]\n\n"
    finally:
        # On client disconnect, close the inner generator now (which sends
        # AbortReq) instead of whenever it gets garbage-collected.
        await outputs.aclose()


# ---------------------------------------------------------------------------
# Launch. SGLang: Engine._launch_subprocesses.
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    port_args = PortArgs.new()

    # "spawn", not "fork": a forked child cannot safely initialize CUDA.
    mp_ctx = mp.get_context("spawn")
    reader, writer = mp_ctx.Pipe(duplex=False)
    scheduler_proc = mp_ctx.Process(
        target=run_scheduler_process,
        args=(args.model, port_args, writer),
        daemon=True,
    )
    detokenizer_proc = mp_ctx.Process(
        target=run_detokenizer_process, args=(args.model, port_args), daemon=True
    )
    scheduler_proc.start()
    detokenizer_proc.start()

    # Block until the model is loaded, so the first request isn't stuck.
    msg = reader.recv()
    if msg != "ready":
        raise SystemExit(f"Scheduler failed to start:\n{msg}")

    tm = TokenizerManager(args.model, port_args)
    uvicorn.run(build_app(tm, args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
