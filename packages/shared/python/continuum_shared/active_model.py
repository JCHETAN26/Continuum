"""Resolving and loading whichever model version is currently ACTIVE.

Both the serving engine and the embedding worker need this. Serving needs it to answer
requests with the adapted model; the worker needs it because activating a new encoder
invalidates every stored vector, and re-embedding is the worker's job rather than the
trainer's. Two copies of artifact fetching, checksum verification and manifest parsing
would be two places for the contract to drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from continuum_shared.artifacts import decode_manifest, parse_s3_uri, sha256_hex
from continuum_shared.config import settings

PROJECTION = "projection"
ENCODER = "encoder"
KNOWN_KINDS = frozenset({PROJECTION, ENCODER})


@dataclass(frozen=True)
class LoadedModel:
    """An ACTIVE model version, with a session when it carries an ONNX artifact."""

    model_id: str
    version: str
    kind: str
    session: Any | None
    input_name: str = "input"
    output_name: str = "embeddings"

    @property
    def is_encoder(self) -> bool:
        """True when the artifact replaces the base model rather than post-processing it."""
        return self.kind == ENCODER and self.session is not None


def minio_client() -> Any:
    from minio import Minio

    endpoint = str(settings.s3_endpoint).replace("http://", "").replace("https://", "").rstrip("/")
    return Minio(
        endpoint=endpoint,
        access_key=settings.s3_access_key_id,
        secret_key=settings.s3_secret_access_key,
        secure=str(settings.s3_endpoint).startswith("https://"),
    )


def download_object(uri: str, *, expected_sha256: str | None = None) -> bytes:
    bucket, object_name = parse_s3_uri(uri)
    response = minio_client().get_object(bucket, object_name)
    try:
        data = bytes(response.read())
    finally:
        response.close()
        response.release_conn()

    if expected_sha256 is not None:
        actual = sha256_hex(data)
        if actual != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for {uri}: expected {expected_sha256}, got {actual}"
            )
    return data


def read_manifest(artifact_uri: str, expected_sha256: str) -> dict[str, Any] | None:
    """Fetch and verify an artifact manifest, or None for the built-in baseline."""
    if not artifact_uri or artifact_uri.startswith("builtin://"):
        return None
    return decode_manifest(download_object(artifact_uri, expected_sha256=expected_sha256))


def onnx_kind(manifest: dict[str, Any]) -> str:
    """Read the declared artifact shape, rejecting anything unrecognised.

    Checked before the artifact is fetched: an unservable manifest should not cost a
    download. Absent metadata means a legacy projection, which is what the demo adapter
    produces; only PEFT exports declare themselves as encoders.
    """
    onnx_metadata = manifest.get("onnx")
    if not isinstance(onnx_metadata, dict):
        raise ValueError("Model manifest does not contain ONNX metadata.")

    kind = str(onnx_metadata.get("kind", PROJECTION))
    if kind not in KNOWN_KINDS:
        raise ValueError(f"Unknown ONNX model kind {kind!r} in manifest")
    return kind


def build_session(manifest: dict[str, Any]) -> Any:
    import onnxruntime as ort

    onnx_metadata = manifest["onnx"]
    uri = onnx_metadata.get("uri")
    expected = onnx_metadata.get("sha256")
    if not isinstance(uri, str) or not isinstance(expected, str):
        raise ValueError("Model manifest ONNX metadata is incomplete.")

    data = download_object(uri, expected_sha256=expected)
    with NamedTemporaryFile(suffix=".onnx", delete=False) as artifact_file:
        artifact_file.write(data)
        artifact_path = artifact_file.name
    try:
        return ort.InferenceSession(artifact_path, providers=["CPUExecutionProvider"])
    finally:
        Path(artifact_path).unlink(missing_ok=True)


async def load_active_model(db: Any) -> LoadedModel | None:
    """Resolve the ACTIVE model version and load its artifact if it has one."""
    rows = await db.query_raw(
        """
        SELECT id::text AS id, version, artifact_uri, artifact_sha256
        FROM model_versions
        WHERE status = 'ACTIVE'
        LIMIT 1
        """
    )
    if not rows:
        return None

    row = rows[0]
    manifest = read_manifest(str(row["artifact_uri"] or ""), str(row["artifact_sha256"] or ""))
    if not manifest or not manifest.get("onnx"):
        return LoadedModel(
            model_id=str(row["id"]), version=str(row["version"]), kind=PROJECTION, session=None
        )

    kind = onnx_kind(manifest)
    onnx_metadata = manifest["onnx"]
    return LoadedModel(
        model_id=str(row["id"]),
        version=str(row["version"]),
        kind=kind,
        session=build_session(manifest),
        input_name=str(onnx_metadata.get("input_name", "input")),
        output_name=str(onnx_metadata.get("output_name", "embeddings")),
    )
