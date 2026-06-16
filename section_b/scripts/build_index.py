"""Build Section B retrieval artifacts."""
from __future__ import annotations

import sys
from pathlib import Path

SECTION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SECTION_ROOT))

from index import build_index


if __name__ == "__main__":
    build_index()
    print("Index built under artifacts/.")
