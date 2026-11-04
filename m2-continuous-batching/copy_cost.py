"""The cost of copying. How much of a decode step is not the model?

    uv run m2-continuous-batching/copy_cost.py --model Qwen/Qwen3-0.6B

Runs ModelRunner directly, with no server, so that every request decodes for
exactly --steps tokens: there is no EOS here to cut a long run short. Prompts
have a few different lengths so that padding is covered too.

  pad    pad_sequence, plus splitting it into the HF cache layer by layer
  model  the model call
  cat    stacking the new K/V and torch.cat onto each request's cache

"model" is not free of copying either: HF's cache runs torch.cat on every layer
inside the call (DynamicLayer.update). So the share printed here is a lower bound.

32 requests at 1024 tokens hold a few GB of KV, and pad + cat keep a second and
third copy alive at the peak. Lower --batch-size if you run out of GPU memory.
"""

from typing import Annotated

import typer
from transformers import AutoTokenizer

from model_runner import FakeReq, ModelRunner

PROMPT = "Write a detailed essay about the history of the printing press."


def main(
    model: Annotated[str, typer.Option(help="HF model id or local path.")] = "Qwen/Qwen3-0.6B",
    batch_size: Annotated[int, typer.Option(help="Requests decoding together.")] = 32,
    steps: Annotated[int, typer.Option(help="Decode steps, so tokens per request.")] = 1024,
    report_every: Annotated[int, typer.Option(help="Print a row every N steps.")] = 128,
):
    """Split decode time into copying and the model call as the context grows."""
    runner = ModelRunner(model, profile_decode=True)
    ids = AutoTokenizer.from_pretrained(model)(PROMPT).input_ids
    reqs = [FakeReq(ids + ids[: i % 8]) for i in range(batch_size)]

    for req, token in zip(reqs, runner.forward(reqs, is_extend=True)):
        req.output_ids.append(token)

    print(f"device={runner.device} batch_size={batch_size}")
    print(f"{'step':>6} {'context':>8} {'pad ms':>8} {'model ms':>9} {'cat ms':>8} {'copy share':>11}")
    for step in range(1, steps + 1):
        for req, token in zip(reqs, runner.forward(reqs, is_extend=False)):
            req.output_ids.append(token)
        if step % report_every == 0:
            r = runner.decode_timer.report()
            copy_ms = r["pad"] + r["cat"]
            share = copy_ms / (copy_ms + r["model"])
            print(
                f"{step:>6} {r['mean_context']:>8.0f} {r['pad']:>8.2f} "
                f"{r['model']:>9.2f} {r['cat']:>8.2f} {share:>10.0%}"
            )


if __name__ == "__main__":
    typer.run(main)
