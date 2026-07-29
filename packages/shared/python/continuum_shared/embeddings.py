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
# Attention is quadratic in sequence length and linear in batch, so a single large call
# allocates batch x heads x seq x seq floats: 1000 documents at 256 tokens is about 3.1 GB
# in fp32, past the trainer's memory limit. Every caller funnels through
# encode_with_session, so chunking here bounds them all.
ENCODE_BATCH_SIZE = 32

# Set to a directory holding model.onnx and tokenizer.json to skip the Hub entirely.
# Container images bake the files in so no network call happens at startup.
MODEL_DIR_ENV = "CONTINUUM_EMBEDDING_MODEL_DIR"

_lock = threading.Lock()
_runtime: _EmbeddingRuntime | None = None


def encode_with_session(session: Any, tokenizer: Any, texts: list[str]) -> np.ndarray:
    """Run texts through any MiniLM-shaped ONNX encoder and pool the result.

    Shared so the serving engine can run a LoRA-adapted encoder through exactly the
    tokenisation and pooling the baseline vectors were produced with. An adapter trained
    under one pooling strategy and served under another yields vectors that are silently
    incomparable with the centroids drift is measured against.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32)

    pooled: list[np.ndarray] = []
    for start in range(0, len(texts), ENCODE_BATCH_SIZE):
        chunk = texts[start : start + ENCODE_BATCH_SIZE]
        encoded = tokenizer.encode_batch(chunk)
        input_ids = np.array([item.ids for item in encoded], dtype=np.int64)
        attention_mask = np.array([item.attention_mask for item in encoded], dtype=np.int64)
        outputs = session.run(
            ["last_hidden_state"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": np.zeros_like(input_ids),
            },
        )[0]
        pooled.append(mean_pool_and_normalize(outputs, attention_mask))

    return np.concatenate(pooled, axis=0)


class _EmbeddingRuntime:
    def __init__(self, session: Any, tokenizer: Any) -> None:
        self.session = session
        self.tokenizer = tokenizer

    def encode(self, texts: list[str]) -> np.ndarray:
        return encode_with_session(self.session, self.tokenizer, texts)


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


def get_tokenizer() -> Any:
    """The adapted encoder shares the base model's vocabulary; LoRA does not change it."""
    return get_runtime().tokenizer


def vector_literal(vector: list[float]) -> str:
    return f"[{','.join(f'{value:.8f}' for value in vector)}]"
