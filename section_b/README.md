# Section B - Wikipedia Retrieval Pipeline

## Project Overview

This directory contains the Section B retrieval system for Project A. The
submission exposes one required function:

```python
run(queries: list[str]) -> list[list[int]]
```

The autograder imports `run` from `main.py` and calls it once with a batch of
evaluation queries. The function returns one ranked list of Wikipedia page IDs
per query, ordered from most relevant to least relevant. Only the first 10 page
IDs are scored with NDCG@10.

## Pipeline

The system is split across small modules:

- `chunk.py` converts each Wikipedia JSON entry into retrieval chunks: title,
  lead paragraph, full-page text for short pages, and overlapping content
  chunks.
- `embed.py` embeds text with `sentence-transformers/all-MiniLM-L6-v2`.
- `index.py` builds and loads the offline FAISS index, chunk metadata, and a
  compressed lexical index.
- `retrieve.py` embeds the query batch, searches FAISS, combines dense and
  lexical signals, and returns ranked page IDs.
- `main.py` exposes the required `run(queries)` API.

## Setup

From the `section_b/` directory, install the required dependencies:

```bash
pip install -r requirements.txt
```

The query embedding model
`sentence-transformers/all-MiniLM-L6-v2` must be available in the execution
environment.  The code uses only repository-relative paths, so it does not
depend on local absolute paths.

## Quick Start

For a fresh clone with the committed artifacts already present:

```bash
python scripts/eval_public.py
```

The offline index rebuild is not part of the normal grading path. If running
commands from inside `section_b/`, use `python scripts/eval_public.py`.

## Public Evaluation

From the repository root, run:

```bash
python scripts/eval_public.py
```

From the `section_b/` directory, the equivalent command is:

```bash
python scripts/eval_public.py
```

Current public result:

```text
mean_ndcg@10=0.4991
num_queries=29
```

## Submitted Artifacts

The graders should not rebuild the index.  The required query-time artifacts are
committed under `artifacts/`:

| Path | Purpose | Format |
| --- | --- | --- |
| `artifacts/chunks.faiss.npy.part00` | First part of the serialized FAISS dense-vector index | Split NumPy `.npy` bytes |
| `artifacts/chunks.faiss.npy.part01` | Second part of the serialized FAISS dense-vector index | Split NumPy `.npy` bytes |
| `artifacts/index_meta.json` | Mapping from FAISS rows to `page_id`, chunk IDs, chunk kind, model name, and dimensions | JSON |
| `artifacts/lexical.json.gz` | Compressed lexical postings used during reranking | Gzipped JSON |

The original full FAISS file is `artifacts/chunks.faiss.npy`, but it is larger
than GitHub's normal per-file limit.  It is intentionally ignored by Git.  At
runtime, `index.py` loads the split `.part00` and `.part01` files directly, so a
fresh clone does not need the full unsplit file.

No Git LFS is required for the committed artifact set because each committed
artifact file is below GitHub's 100 MB file limit.

## Optional Offline Rebuild

The artifacts were generated from the handout corpus with:

```bash
python scripts/build_index.py
```

This step is optional for local development only.  Graders do not need to run it
and should be able to run `python scripts/eval_public.py` directly from a fresh
clone after installing dependencies.

The raw corpus is expected at:

```text
data/Wikipedia Entries/
```

The corpus directory is part of the course handout and is not required for the
normal grading-time evaluation path, because `run()` loads the prebuilt
artifacts.

## Runtime Flow

At query time, `run()` loads the prebuilt artifacts, embeds the full query batch,
retrieves dense FAISS candidates, adds lexical matches from `lexical.json.gz`,
and returns the top page IDs for each query. No corpus rebuild is needed during
grading.

## Development Notes

- Keep `eval.py` and `evaluation.py` unchanged; they are the grading/evaluation
  helpers.
- Query-time changes should remain inside `main.py`, `retrieve.py`, `index.py`,
  `embed.py`, `chunk.py`, or `utils.py`.
- Re-run `python scripts/eval_public.py` after retrieval changes to verify the
  public NDCG score.

## Pre-Submission Checklist

- Install requirements from `section_b/requirements.txt`.
- Verify the required files under `artifacts/` exist.
- Run `python scripts/eval_public.py` from the repository root.
- Do not modify `eval.py` or `evaluation.py`.
- Confirm `git status` is clean.
- Add the presentation video link in the assumptions section when available.

## Assumptions

- `data/public_queries.json` is included for public self-testing.
- The prebuilt files under `artifacts/` are included in the GitHub repository.
- `scripts/eval_public.py` and `scripts/build_index.py` are kept as provided
  evaluation/build entry points.
- The presentation video link should be added here before final submission:
  `TODO: add presentation video link`.
