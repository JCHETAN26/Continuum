import hashlib
import json
from typing import Any


def build_demo_artifact_manifest(
    *,
    version: str,
    base_model: str,
    embedding_dim: int,
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    improvement_pct: float,
    onnx_uri: str | None = None,
    onnx_sha256: str | None = None,
    onnx_bytes: int | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "artifact_format": "continuum.demo.embedding-manifest.v1",
        "version": version,
        "base_model": base_model,
        "embedding_dim": embedding_dim,
        "embedding_engine": "continuum.hash-embedding",
        "adaptation": {
            "kind": "demo-lora-gate",
            "lora_rank": 8,
            "lora_alpha": 16,
        },
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "improvement_pct": improvement_pct,
    }
    if onnx_uri and onnx_sha256 and onnx_bytes is not None:
        manifest["onnx"] = {
            "uri": onnx_uri,
            "sha256": onnx_sha256,
            "bytes": onnx_bytes,
            "input_name": "input",
            "output_name": "embeddings",
        }

    return manifest


def build_peft_artifact_manifest(
    *,
    version: str,
    base_model: str,
    embedding_dim: int,
    domain_tag: str,
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    improvement_pct: float,
    onnx_uri: str,
    onnx_sha256: str,
    onnx_bytes: int,
    adapter_config_uri: str,
    sample_count: int,
) -> dict[str, Any]:
    """Manifest for a LoRA-adapted encoder.

    `kind: encoder` is what tells the serving engine this artifact replaces the base model
    rather than post-multiplying its output. Without it the engine would default to the
    projection contract and feed float vectors into a graph expecting token ids.
    """
    return {
        "artifact_format": "continuum.peft.embedding-manifest.v1",
        "version": version,
        "base_model": base_model,
        "embedding_dim": embedding_dim,
        "embedding_engine": "continuum.peft-lora-encoder",
        "adaptation": {
            "kind": "lora",
            "lora_rank": 8,
            "lora_alpha": 16,
            "domain_tag": domain_tag,
            "sample_count": sample_count,
            "adapter_config_uri": adapter_config_uri,
        },
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "improvement_pct": improvement_pct,
        "onnx": {
            "uri": onnx_uri,
            "sha256": onnx_sha256,
            "bytes": onnx_bytes,
            "kind": "encoder",
        },
    }


def encode_manifest(manifest: dict[str, Any]) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def decode_manifest(data: bytes) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Model artifact manifest must be a JSON object.")
    return value


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Unsupported artifact URI: {uri}")

    bucket_and_key = uri.removeprefix("s3://")
    bucket, separator, key = bucket_and_key.partition("/")
    if not bucket or separator != "/" or not key:
        raise ValueError(f"Invalid S3 artifact URI: {uri}")

    return bucket, key
