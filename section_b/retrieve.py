"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from embed import embed_queries
from index import load_index
from utils import K_EVAL

_CACHE: Dict[str, Tuple[faiss.Index, Dict[str, List]]] = {}

KIND_BONUS = {
    "title": 0.025,
    "lead": 0.015,
    "full": 0.012,
    "chunk": 0.0,
}
LEXICAL_WEIGHT = 0.50
DENSE_CANDIDATES = 2000
PHRASE_BOOST = 3.4
LENGTH_NORM = 0.0018
DENSE_HIT_WEIGHTS = (1.0, 0.08, 0.05, 0.03, 0.02)
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "what", "when", "where",
    "which", "who", "how", "did", "was", "were", "are", "about", "also", "into",
    "after", "before", "during", "over", "under", "they", "their", "link", "links",
    "can", "learned", "together", "involved", "served", "year", "years",
}


def _cached_index(artifacts_dir: Optional[Path]) -> Tuple[faiss.Index, Dict[str, List]]:
    key = str((artifacts_dir or Path("artifacts")).resolve())
    if key not in _CACHE:
        _CACHE[key] = load_index(artifacts_dir)
    return _CACHE[key]


def _query_tokens(query: str) -> List[str]:
    seen: Dict[str, None] = {}
    tokens: List[str] = []
    query = re.sub(r"(?<=\d),(?=\d)", "", query)
    for tok in TOKEN_RE.findall(query.lower()):
        if len(tok) >= 3 and tok not in STOPWORDS:
            tokens.append(tok)
            seen[tok] = None
            if len(tok) == 5 and tok.endswith("0s") and tok[:4].isdigit():
                for year in range(int(tok[:4]), int(tok[:4]) + 10):
                    seen[str(year)] = None
    for i in range(len(tokens) - 1):
        seen[tokens[i] + "_" + tokens[i + 1]] = None
    for i in range(len(tokens) - 2):
        seen[tokens[i] + "_" + tokens[i + 1] + "_" + tokens[i + 2]] = None
    return list(seen)


def _dense_scores(
    indices: np.ndarray,
    scores: np.ndarray,
    page_ids: List[int],
    kinds: List[str],
) -> Dict[int, float]:
    page_hits: Dict[int, List[float]] = {}
    for idx, score in zip(indices.tolist(), scores.tolist()):
        if idx < 0:
            continue
        pid = int(page_ids[idx])
        value = float(score) + KIND_BONUS.get(kinds[idx], 0.0)
        page_hits.setdefault(pid, []).append(value)

    page_scores: Dict[int, float] = {}
    for pid, values in page_hits.items():
        values.sort(reverse=True)
        page_scores[pid] = sum(
            weight * values[i]
            for i, weight in enumerate(DENSE_HIT_WEIGHTS)
            if i < len(values)
        )
    return page_scores


def _lexical_scores(query: str, lexical: Dict[str, List]) -> Dict[int, float]:
    docs = lexical["docs"]
    terms = lexical["terms"]
    scores: Dict[int, float] = {}
    for tok in _query_tokens(query):
        item = terms.get(tok)
        if item is None:
            continue
        idf, postings = item
        if float(idf) < 1.4 or len(postings) > 4000:
            continue
        parts = tok.count("_") + 1
        boost = 6.0 if tok.isdigit() else (1.0 + PHRASE_BOOST * (parts - 1))
        for doc_idx, count in postings:
            doc = docs[int(doc_idx)]
            pid = int(doc["page_id"])
            length = float(doc["length"])
            tf = 1.0 + math.log1p(float(count))
            scores[pid] = scores.get(pid, 0.0) + boost * float(idf) * tf / (1.0 + LENGTH_NORM * length)
    return scores


def _rank_pages(
    dense: Dict[int, float],
    lexical: Dict[int, float],
    top_k: int,
    lexical_weight: float,
) -> List[int]:
    combined: Dict[int, float] = {}
    max_lex = max(lexical.values()) if lexical else 1.0
    for pid, score in dense.items():
        combined[pid] = combined.get(pid, 0.0) + score
    for pid, score in lexical.items():
        combined[pid] = combined.get(pid, 0.0) + lexical_weight * score / max_lex
    ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
    return [pid for pid, _ in ranked[:top_k]]


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """
    Return ranked page_id lists (best first) for each query.

    Search chunk embeddings with FAISS, then aggregate chunk hits to unique pages.
    """
    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]
    index, meta = _cached_index(artifacts_dir)
    query_vectors = np.ascontiguousarray(query_vectors.astype(np.float32, copy=False))
    nprobe = min(max(top_k * 60, DENSE_CANDIDATES), index.ntotal)
    scores, indices = index.search(query_vectors, nprobe)
    page_ids = meta["page_ids"]
    kinds = meta["kinds"]
    return [
        _rank_pages(
            _dense_scores(row_idx, row_scores, page_ids, kinds),
            _lexical_scores(query, meta["lexical"]),
            top_k,
            LEXICAL_WEIGHT,
        )
        for query, row_idx, row_scores in zip(queries, indices, scores)
    ]
