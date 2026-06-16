"""Embedding utilities (sentence-transformers/all-MiniLM-L6-v2 only)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence

import numpy as np

from utils import EMBEDDING_MODEL_NAME

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        try:
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        except TypeError:
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        _model.max_seq_length = 256
    return _model


def embed_texts(texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
    """Return L2-normalized embeddings, shape (n, dim)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_queries(queries: List[str], *, batch_size: int = 64) -> np.ndarray:
    """Embed grading queries in a subprocess to isolate torch from FAISS runtime state."""
    if not queries:
        return np.zeros((0, 384), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        in_path = tmpdir / "queries.json"
        out_path = tmpdir / "vectors.npy"
        in_path.write_text(json.dumps(list(queries)), encoding="utf-8")
        # Keep query embedding isolated from the parent FAISS process state.
        code = (
            "import json, numpy as np, sys; "
            "from sentence_transformers import SentenceTransformer; "
            f"m=SentenceTransformer({EMBEDDING_MODEL_NAME!r}, local_files_only=True); "
            "m.max_seq_length=256; "
            "q=json.loads(open(sys.argv[1], encoding='utf-8').read()); "
            f"v=m.encode(q, batch_size={int(batch_size)}, show_progress_bar=False, "
            "convert_to_numpy=True, normalize_embeddings=True); "
            "np.save(sys.argv[2], np.asarray(v, dtype=np.float32))"
        )
        # Force local model loading during grading, where network access is unavailable.
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        subprocess.run([sys.executable, "-c", code, str(in_path), str(out_path)], check=True, env=env)
        return np.load(out_path)
