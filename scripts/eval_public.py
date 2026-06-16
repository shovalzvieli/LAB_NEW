"""Root-level entry point for the Section B public evaluation."""
from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
runpy.run_path(
    str(ROOT / "section_b" / "scripts" / "eval_public.py"),
    run_name="__main__",
)
