"""Sentence embeddings from `all-MiniLM-L6-v2`, served through ONNX Runtime.

Inference deliberately avoids torch. The published ONNX export plus a tokenizers
vocabulary is a few hundred megabytes lighter, and the serving engine already depends on
onnxruntime to run adapted models, so this reuses machinery the system needs anyway.

The pooling here has to match `continuum_trainer.peft_engine.mean_pool_and_normalize`
exactly: attention-masked mean over token states, then L2 normalisation. A LoRA adapter
trained against one pooling strategy and served under another produces vectors that are
silently incomparable with the baseline centroids drift is measured from.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_FILENAME = "onnx/model.onnx"
TOKENIZER_FILENAME = "tokenizer.json"
EMBEDDING_DIMENSION = 384
MAX_SEQUENCE_LENGTH = 256

# Set to a directory holding model.onnx and tokenizer.json to skip the Hub entirely.
# Container images bake the files in so no network call happens at startup.
MODEL_DIR_ENV = "CONTINUUM_EMBEDDING_MODEL_DIR"

_lock = threading.Lock()
_runtime: _EmbeddingRuntime | None = None


class _EmbeddingRuntime:
    def __init__(self, session: Any, tokenizer: Any) -> None:
        self.session = session
        self.tokenizer = tokenizer

    def encode(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = np.array([item.ids for item in encoded], dtype=np.int64)
        attention_mask = np.array([item.attention_mask for item in encoded], dtype=np.int64)
        outputs = self.session.run(
            ["last_hidden_state"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": np.zeros_like(input_ids),
            },
        )[0]
        return mean_pool_and_normalize(outputs, attention_mask)


def mean_pool_and_normalize(token_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Attention-masked mean pooling followed by L2 normalisation."""
    mask = attention_mask[..., None].astype(np.float32)
    summed = (token_states * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    pooled = summed / counts
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return np.asarray(pooled / np.clip(norms, 1e-12, None), dtype=np.float32)


@lru_cache(maxsize=1)
def resolve_model_files() -> tuple[Path, Path]:
    """Locate the ONNX graph and tokenizer, preferring a baked-in directory."""
    local_dir = os.environ.get(MODEL_DIR_ENV)
    if local_dir:
        root = Path(local_dir)
        onnx_path = root / "model.onnx"
        tokenizer_path = root / TOKENIZER_FILENAME
        if not onnx_path.exists() or not tokenizer_path.exists():
            raise FileNotFoundError(
                f"{MODEL_DIR_ENV}={local_dir} must contain model.onnx and {TOKENIZER_FILENAME}"
            )
        return onnx_path, tokenizer_path

    from huggingface_hub import hf_hub_download

    return (
        Path(hf_hub_download(MODEL_REPO, ONNX_FILENAME)),
        Path(hf_hub_download(MODEL_REPO, TOKENIZER_FILENAME)),
    )


def get_runtime() -> _EmbeddingRuntime:
    """Build the ONNX session once per process; it is not cheap to construct."""
    global _runtime
    if _runtime is not None:
        return _runtime

    with _lock:
        if _runtime is not None:
            return _runtime

        import onnxruntime as ort
        from tokenizers import Tokenizer

        onnx_path, tokenizer_path = resolve_model_files()
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_padding()
        tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        _runtime = _EmbeddingRuntime(session, tokenizer)
        return _runtime


def embed_texts(texts: list[str], dimension: int = EMBEDDING_DIMENSION) -> list[list[float]]:
    """Embed a batch of documents into unit-norm vectors."""
    if dimension != EMBEDDING_DIMENSION:
        raise ValueError(
            f"{MODEL_REPO} produces {EMBEDDING_DIMENSION}-dimensional vectors, got {dimension}"
        )
    if not texts:
        return []

    vectors = get_runtime().encode(texts)
    return [[float(value) for value in row] for row in vectors]


def embed_text(text: str, dimension: int = EMBEDDING_DIMENSION) -> list[float]:
    return embed_texts([text], dimension)[0]


def vector_literal(vector: list[float]) -> str:
    return f"[{','.join(f'{value:.8f}' for value in vector)}]"
