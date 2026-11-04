"""nano-serve v2: entry point, HTTP server and TokenizerManager (main process).

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

This file only gains the KV pool options since M2: memory is the scheduler
process's business, and nothing here knows it exists.

Run:  uv run m4-radix-cache/server.py --model Qwen/Qwen3-0.6B --port 8000
"""

import asyncio
import dataclasses
import json
import multiprocessing as mp
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator, Dict, List, Optional

import typer
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel
from transformers import AutoTokenizer

from common import (
    AbortReq,
    BatchStrOut,
    EngineOptions,
    FlushCacheReq,
    PortArgs,
    TokenizedGenerateReq,
    get_socket,
    setup_logging,
)
from detokenizer import run_detokenizer_process
from radix_cache import EVICTION_POLICIES
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
    # {"include_usage": true} adds a final chunk with token counts (bench.py uses it).
    stream_options: Optional[Dict[str, Any]] = None
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

    @app.post("/flush_cache")
    async def flush_cache():
        """The socket keeps its order, so this request sees the cache
        flushed if the server is idle. Busy servers log a warning and keep it."""
        await tm.send_to_scheduler.send_pyobj(FlushCacheReq())
        return {"status": "sent"}

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
            include_usage = bool((request.stream_options or {}).get("include_usage"))
            return StreamingResponse(
                _stream_chunks(
                    outputs,
                    response_id,
                    created,
                    model_name,
                    prompt_tokens=len(input_ids) if include_usage else None,
                ),
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
    outputs: AsyncIterator[Dict[str, Any]],
    response_id: str,
    created: int,
    model: str,
    prompt_tokens: Optional[int] = None,  # Set to send a usage chunk at the end.
):
    def chunk(choices: List[Dict[str, Any]], **extra) -> str:
        body = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": choices,
            **extra,
        }
        return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

    def choice(delta: Dict[str, Any], finish_reason: Optional[str] = None):
        return [{"index": 0, "delta": delta, "finish_reason": finish_reason}]

    try:
        yield chunk(choice({"role": "assistant", "content": ""}))
        completion_tokens = 0
        async for out in outputs:
            completion_tokens = out["completion_tokens"]
            if out["text"]:
                yield chunk(choice({"content": out["text"]}))
            if out["finish_reason"] is not None:
                yield chunk(choice({}, finish_reason=out["finish_reason"]))
        if prompt_tokens is not None:
            # OpenAI's format: an extra chunk with empty choices and the usage.
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            yield chunk([], usage=usage)
        yield "data: [DONE]\n\n"
    finally:
        # On client disconnect, close the inner generator now (which sends
        # AbortReq) instead of whenever it gets garbage-collected.
        await outputs.aclose()


# ---------------------------------------------------------------------------
# Launch. SGLang: Engine._launch_subprocesses.
# ---------------------------------------------------------------------------


def main(
    model: Annotated[
        str, typer.Option(help="HF model id or local path.")
    ] = "Qwen/Qwen3-0.6B",
    host: Annotated[str, typer.Option(help="Address to listen on.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    max_running_requests: Annotated[
        int, typer.Option(help="Most requests decoding at once (SGLang: --max-running-requests).")
    ] = 64,
    max_prefill_tokens: Annotated[
        int, typer.Option(help="Prompt-token budget of one prefill batch (SGLang: --max-prefill-tokens).")
    ] = 8192,
    max_total_tokens: Annotated[
        Optional[int],
        typer.Option(help="KV pool size in tokens. Default: from free GPU memory (SGLang: --max-total-tokens)."),
    ] = None,
    mem_fraction_static: Annotated[
        float,
        typer.Option(help="Share of free GPU memory for the KV pool (SGLang: --mem-fraction-static)."),
    ] = 0.6,
    context_length: Annotated[
        int, typer.Option(help="Longest request, prompt + output, in tokens (SGLang: --context-length).")
    ] = 8192,
    new_token_ratio: Annotated[
        float, typer.Option(help="Share of new tokens to reserve, 1 never retracts.")
    ] = 1.0,
    batched_prefill: Annotated[
        bool, typer.Option(help="One padded attention call per prefill batch.")
    ] = False,
    page_size: Annotated[int, typer.Option(help="Slots handed out at a time.")] = 1,
    lazy_free: Annotated[
        bool, typer.Option(help="Merge the free list only when alloc runs short.")
    ] = False,
    check_double_free: Annotated[
        bool, typer.Option(help="Track free pages and raise on a double free.")
    ] = False,
    disable_radix_cache: Annotated[
        bool, typer.Option(help="Prefill every request in full (SGLang: --disable-radix-cache).")
    ] = False,
    eviction_policy: Annotated[
        str, typer.Option(help="LRU or LFU (SGLang: --radix-eviction-policy).")
    ] = "lru",
    schedule_policy: Annotated[
        str, typer.Option(help="FCFS or LPM (SGLang: --schedule-policy).")
    ] = "fcfs",
    in_batch_dedup: Annotated[
        bool, typer.Option(help="Hold back requests whose prefix this batch computes.")
    ] = False,
    cache_retracted: Annotated[
        bool, typer.Option(help="Retraction moves K/V to the tree instead of freeing it.")
    ] = False,
    fast_match: Annotated[
        bool, typer.Option(help="Compare keys by slices, not token by token.")
    ] = False,
):
    """Launch nano-serve: HTTP server + scheduler + detokenizer processes."""
    if not 0 < new_token_ratio <= 1:
        raise typer.BadParameter("--new-token-ratio must be in (0, 1].")
    if eviction_policy not in EVICTION_POLICIES:
        raise typer.BadParameter(f"--eviction-policy must be in {list(EVICTION_POLICIES)}.")
    if schedule_policy not in ("fcfs", "lpm"):
        raise typer.BadParameter("--schedule-policy must be fcfs or lpm.")
    options = EngineOptions(
        new_token_ratio=new_token_ratio,
        batched_prefill=batched_prefill,
        page_size=page_size,
        lazy_free=lazy_free,
        check_double_free=check_double_free,
        disable_radix_cache=disable_radix_cache,
        eviction_policy=eviction_policy,
        schedule_policy=schedule_policy,
        in_batch_dedup=in_batch_dedup,
        cache_retracted=cache_retracted,
        fast_match=fast_match,
    )
    setup_logging("main")
    port_args = PortArgs.new()
    logger.info(f"IPC addresses: {port_args}")

    # "spawn", not "fork": a forked child cannot safely initialize CUDA.
    mp_ctx = mp.get_context("spawn")
    reader, writer = mp_ctx.Pipe(duplex=False)
    scheduler_proc = mp_ctx.Process(
        target=run_scheduler_process,
        args=(
            model,
            port_args,
            max_running_requests,
            max_prefill_tokens,
            max_total_tokens,
            mem_fraction_static,
            context_length,
            options,
            writer,
        ),
        daemon=True,
    )
    detokenizer_proc = mp_ctx.Process(
        target=run_detokenizer_process, args=(model, port_args), daemon=True
    )
    scheduler_proc.start()
    detokenizer_proc.start()

    # Block until the model is loaded, so the first request isn't stuck.
    logger.info(f"Waiting for the scheduler to load {model} ...")
    msg = reader.recv()
    if msg != "ready":
        logger.error(f"Scheduler failed to start:\n{msg}")
        raise SystemExit(1)

    tm = TokenizerManager(model, port_args)
    logger.info(f"Serving on http://{host}:{port}")
    uvicorn.run(build_app(tm, model), host=host, port=port)


if __name__ == "__main__":
    typer.run(main)
