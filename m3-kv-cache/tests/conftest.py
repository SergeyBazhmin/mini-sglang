"""Make this chapter's modules importable. The tests import them by their plain names."""

import sys
from pathlib import Path

CHAPTER = Path(__file__).resolve().parent
sys.path.insert(0, str(CHAPTER.parent if CHAPTER.name == "tests" else CHAPTER))
