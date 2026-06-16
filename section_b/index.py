"""Offline index build and load (not timed at grading)."""
from __future__ import annotations

import json
import gzip
import io
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries

FAISS_INDEX_NAME = "chunks.faiss.npy"
INDEX_META_NAME = "index_meta.json"
LEXICAL_NAME = "lexical.json.gz"
EMBED_BATCH_SIZE = 128
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "what", "when", "where",
    "which", "who", "how", "did", "was", "were", "are", "about", "also", "into",
    "after", "before", "during", "over", "under", "they", "their", "what", "link",
    "links", "can", "learned", "together", "involved", "served", "year", "years",
}


def _tokens(text: str) -> List[str]:
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) >= 3 and t not in STOPWORDS]


def _features(text: str) -> List[str]:
    tokens = _tokens(text)
    features = list(tokens)
    features.extend(tokens[i] + "_" + tokens[i + 1] for i in range(len(tokens) - 1))
    features.extend(
        tokens[i] + "_" + tokens[i + 1] + "_" + tokens[i + 2]
        for i in range(len(tokens) - 2)
    )
    return features


def _build_lexical(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    docs: List[Dict[str, Any]] = []
    postings: Dict[str, Dict[int, int]] = {}
    for i, record in enumerate(records):
        page_id = int(record["page_id"])
        title = str(record.get("title", ""))
        content = " ".join(str(record.get("content", "")).split()[:1200])
        counts: Dict[str, int] = {}
        for tok in _features(title + " " + title + " " + content):
            counts[tok] = counts.get(tok, 0) + 1
        docs.append({"page_id": page_id, "length": max(1, sum(counts.values()))})
        for tok, count in counts.items():
            postings.setdefault(tok, {})[i] = count
    n_docs = len(docs)
    terms: Dict[str, Any] = {}
    for tok, posting in postings.items():
        df = len(posting)
        if (df <= 1 and not tok.isdigit()) or df > 3000:
            continue
        idf = math.log((n_docs + 1) / (df + 0.5))
        terms[tok] = [round(idf, 6), list(posting.items())]
    return {"docs": docs, "terms": terms}


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[faiss.Index, List[int]]:
    """
    Embed corpus chunks and persist all query-time artifacts.

    Returns (faiss_index, page_ids) where FAISS row i maps to page_ids[i].
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(iter_entries(entries_dir))
    chunks: List[Chunk] = chunk_corpus(records)
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts, batch_size=EMBED_BATCH_SIZE)
    vectors = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
    page_ids = [c.page_id for c in chunks]
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    np.save(out_dir / FAISS_INDEX_NAME, faiss.serialize_index(index))
    meta: Dict[str, Any] = {
        "page_ids": page_ids,
        "chunk_ids": [c.chunk_id for c in chunks],
        "kinds": [c.kind for c in chunks],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(page_ids),
        "dimension": int(vectors.shape[1]),
    }
    (out_dir / INDEX_META_NAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    with gzip.open(out_dir / LEXICAL_NAME, "wt", encoding="utf-8") as f:
        json.dump(_build_lexical(records), f, separators=(",", ":"))
    return index, page_ids


def _load_faiss_array(root: Path) -> np.ndarray:
    """Load the serialized FAISS array, including GitHub-safe split parts."""
    full_path = root / FAISS_INDEX_NAME
    if full_path.exists():
        return np.load(full_path)
    parts = sorted(root.glob(f"{FAISS_INDEX_NAME}.part*"))
    if not parts:
        raise FileNotFoundError(full_path)
    data = b"".join(path.read_bytes() for path in parts)
    return np.load(io.BytesIO(data))


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[faiss.Index, Dict[str, List]]:
    """Load precomputed FAISS index and chunk metadata from artifacts/."""
    root = artifacts_dir or ARTIFACTS_DIR
    index = faiss.deserialize_index(_load_faiss_array(root))
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    meta["page_ids"] = [int(x) for x in meta["page_ids"]]
    meta["chunk_ids"] = [int(x) for x in meta["chunk_ids"]]
    with gzip.open(root / LEXICAL_NAME, "rt", encoding="utf-8") as f:
        meta["lexical"] = json.load(f)
    return index, meta
