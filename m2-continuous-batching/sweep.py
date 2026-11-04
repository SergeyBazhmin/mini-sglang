"""The scaling curve. Run bench.py at growing concurrency and plot it.

    uv run --with matplotlib m2-continuous-batching/sweep.py --port 8000

matplotlib is not a project dependency, so --with pulls it in for this run only.
Writes sweep.csv and sweep.png to the current directory (override with --output).

Throughput and latency have different units, so they get a chart each, not two
axes on one chart.

Start the server with --max-running-requests at least as high as the largest
concurrency, or the top of the curve only measures the queue.
"""

import asyncio
import csv
from pathlib import Path
from typing import Annotated, Dict, List

import typer

from bench import run_bench

COLOR = "#2a78d6"
TEXT = "#52514e"
GRID = "#e4e3df"


def plot(rows: List[Dict[str, float]], path: Path):
    import matplotlib

    matplotlib.use("Agg")  # The GPU machine has no display.
    import matplotlib.pyplot as plt

    xs = [r["concurrency"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    panels = [
        (axes[0], "throughput", "Output throughput (tok/s)"),
        (axes[1], "latency_mean", "Mean request latency (s)"),
    ]
    for ax, key, title in panels:
        ax.plot(xs, [r[key] for r in rows], color=COLOR, linewidth=2, marker="o", markersize=6)
        ax.set_title(title, loc="left", color=TEXT)
        ax.set_xlabel("Concurrent requests", color=TEXT)
        # Concurrency doubles each point, so a log axis spaces them evenly.
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, labels=[str(x) for x in xs])
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT, length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(
    port: Annotated[int, typer.Option(help="Port nano-serve is listening on.")] = 8000,
    max_concurrency: Annotated[int, typer.Option(help="Double concurrency from 1 up to this.")] = 64,
    rounds: Annotated[int, typer.Option(help="Requests per point, as a multiple of concurrency.")] = 2,
    min_requests: Annotated[int, typer.Option(help="Fewest requests per point.")] = 8,
    max_tokens: Annotated[int, typer.Option(help="max_tokens of every request.")] = 256,
    output: Annotated[str, typer.Option(help="Output path without extension.")] = "sweep",
):
    """Measure throughput and latency at concurrency 1, 2, 4, ... and plot them."""
    rows = []
    concurrency = 1
    while concurrency <= max_concurrency:
        num_requests = max(rounds * concurrency, min_requests)
        rows.append(asyncio.run(run_bench(port, num_requests, concurrency, max_tokens)))
        print()
        concurrency *= 2

    csv_path = Path(output).with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'concurrency':>11} {'tok/s':>9} {'TTFT s':>8} {'latency s':>10} {'tok/s per req':>14}")
    for r in rows:
        print(
            f"{r['concurrency']:>11} {r['throughput']:>9.1f} {r['ttft_mean']:>8.2f} "
            f"{r['latency_mean']:>10.2f} {r['throughput'] / r['concurrency']:>14.1f}"
        )

    png_path = Path(output).with_suffix(".png")
    plot(rows, png_path)
    print(f"\nWrote {csv_path} and {png_path}")


if __name__ == "__main__":
    typer.run(main)
