import hashlib
import math
import re

TOKEN_RE = re.compile(r"[a-z0-9]+")

DOMAIN_HINTS = {
    "software": {
        "api",
        "backend",
        "cache",
        "ci",
        "cluster",
        "component",
        "deploying",
        "endpoint",
        "gateway",
        "index",
        "jwt",
        "kubernetes",
        "memory",
        "microservice",
        "node",
        "postgresql",
        "react",
        "redis",
        "test",
        "typescript",
        "worker",
    },
    "healthcare": {
        "abnormality",
        "antibiotics",
        "biopsy",
        "blood",
        "cardiac",
        "cardiology",
        "cortex",
        "diabetes",
        "echocardiogram",
        "hypertension",
        "infection",
        "losartan",
        "malignancy",
        "mri",
        "patient",
        "respiratory",
        "vital",
    },
}


def embed_text(text: str, dimension: int = 384) -> list[float]:
    """Create a deterministic, normalized lexical embedding for local demos.

    The vector is intentionally lightweight and offline-safe. It is not a substitute for
    transformer embeddings, but it preserves enough lexical/domain signal for the full
    Continuum pipeline to compute meaningful drift and retrieval metrics locally.
    """

    vector = [0.0] * dimension
    tokens = TOKEN_RE.findall(text.lower())

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign

    token_set = set(tokens)
    for offset, words in enumerate(DOMAIN_HINTS.values()):
        vector[offset] += 4.0 * len(token_set.intersection(words))

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector

    return [value / norm for value in vector]


def embed_texts(texts: list[str], dimension: int = 384) -> list[list[float]]:
    return [embed_text(text, dimension) for text in texts]


def vector_literal(vector: list[float]) -> str:
    return f"[{','.join(f'{value:.8f}' for value in vector)}]"
