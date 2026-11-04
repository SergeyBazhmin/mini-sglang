"""Shared by all chapters. Each chapter's own conftest.py makes its modules importable.

    uv run pytest m4-radix-cache                                     # No model.
    uv run pytest m4-radix-cache -m model --model Qwen/Qwen3-0.6B    # Only with a model.
    uv run pytest m4-radix-cache -m "" --model Qwen/Qwen3-0.6B       # Everything.
"""

from pathlib import Path
from typing import List

import pytest


def pytest_addoption(parser):
    parser.addoption("--model", default="Qwen/Qwen3-0.6B", help="HF model id or local path.")
    parser.addoption(
        "--max-new-tokens", type=int, default=32, help="Most tokens to generate per prompt."
    )


@pytest.fixture(scope="session")
def model(request) -> str:
    return request.config.getoption("--model")


@pytest.fixture(scope="session")
def max_new_tokens(request) -> int:
    return request.config.getoption("--max-new-tokens")


def pytest_collection_modifyitems(items: List[pytest.Item]):
    """Every chapter has its own scheduler.py, common.py and so on. Python keeps one
    module per name, so a second chapter in a session would silently test the first."""
    chapters = set()
    for item in items:
        folder = Path(item.path).parent
        chapters.add(folder.parent if folder.name == "tests" else folder)
    if len(chapters) > 1:
        names = sorted(str(c.relative_to(Path(__file__).parent)) for c in chapters)
        raise pytest.UsageError(f"Run one chapter at a time, not {names}.")
