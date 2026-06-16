"""Evaluate Section B on public_queries.json."""
from __future__ import annotations

import sys
from pathlib import Path

SECTION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SECTION_ROOT))

from evaluation import evaluate_run, load_query_file
from main import run
from utils import PUBLIC_QUERIES_PATH


def main() -> None:
    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [str(row.get("query", row.get("question", row.get("text", "")))) for row in rows]
    relevant = [row["relevant_page_ids"] for row in rows]
    stats = evaluate_run(queries, relevant, run)
    print(f"mean_ndcg@10={stats['mean_ndcg@10']:.4f}")
    print(f"num_queries={int(stats['num_queries'])}")


if __name__ == "__main__":
    main()
