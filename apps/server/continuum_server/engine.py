import asyncio
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import onnxruntime as ort
import structlog
from continuum_shared.artifacts import decode_manifest, parse_s3_uri, sha256_hex
from continuum_shared.config import settings
from continuum_shared.embeddings import embed_texts, encode_with_session, get_tokenizer
from continuum_shared.prisma import Prisma
from continuum_shared.prisma.enums import ModelStatus
from minio import Minio
from tokenizers import Tokenizer

from continuum_server.rollback import ModelRollbackPolicy

logger = structlog.get_logger()


class ModelEngine:
    def __init__(self):
        self.session: ort.InferenceSession | None = None
        self.tokenizer: Tokenizer | None = None
        self.current_version: str | None = None
        self.artifact_manifest: dict[str, Any] | None = None
        self.onnx_input_name = "input"
        self.onnx_output_name = "embeddings"
        # Two shapes of adapted model reach serving. "projection" post-multiplies the base
        # vectors by a learned matrix. "encoder" is a LoRA-adapted MiniLM: it consumes
        # tokens, not vectors, so the base model is replaced rather than post-processed.
        self.model_kind: str = "projection"
        self.dimension: int = 384
        self._lock = asyncio.Lock()
        self.db = Prisma()
        self.rollback_policy = ModelRollbackPolicy.from_env()

    async def connect(self):
        await self.db.connect()
        # Initial load
        await self.poll_active_model()

    async def disconnect(self):
        await self.db.disconnect()

    async def poll_active_model(self):
        """Poll the database for the active model and hot-swap if changed."""
        active_model = await self.db.modelversion.find_first(where={"status": ModelStatus.ACTIVE})

        if not active_model:
            active_model = await self.db.modelversion.create(
                data={
                    "version": "baseline",
                    "baseModel": settings.embedding_model,
                    "status": ModelStatus.ACTIVE,
                    "artifactUri": "builtin://continuum/hash-embedding",
                    "artifactSha256": "0" * 64,
                    "artifactBytes": 0,
                    "activatedAt": datetime.now(UTC),
                }
            )
            logger.info("Created built-in ACTIVE baseline model.")

        if active_model.version == self.current_version:
            return  # Unchanged

        logger.info("New ACTIVE model detected, initiating hot-swap", version=active_model.version)
        previous_version = self.current_version

        try:
            manifest = await self._load_artifact_manifest(
                artifact_uri=active_model.artifactUri,
                expected_sha256=active_model.artifactSha256,
            )
            session = None
            input_name = "input"
            output_name = "embeddings"
            kind = "projection"
            if manifest and manifest.get("onnx"):
                session, input_name, output_name, kind = await self._load_onnx_session(manifest)

            async with self._lock:
                self.current_version = active_model.version
                self.artifact_manifest = manifest
                self.session = session
                self.onnx_input_name = input_name
                self.onnx_output_name = output_name
                self.model_kind = kind
                self.rollback_policy.note_activation(previous_version, active_model.version)
                logger.info("Hot-swap complete", version=self.current_version, model_kind=kind)
        except Exception as e:
            logger.error("Failed to load new model", error=str(e), version=active_model.version)

    async def record_request_metric(
        self, model_version: str, *, status_code: int, latency_ms: float
    ) -> None:
        self.rollback_policy.record(
            model_version,
            status_code=status_code,
            latency_ms=latency_ms,
        )
        try:
            await self.db.execute_raw(
                """
                INSERT INTO model_request_metrics (model_version, status_code, latency_ms)
                VALUES ($1, $2, $3)
                """,
                model_version,
                status_code,
                latency_ms,
            )
        except Exception as e:
            logger.debug("Unable to persist request metric", error=str(e))

    async def rollback_if_needed(self):
        decision = await self.rollback_policy.rollback_if_needed(
            self.current_version,
            self._activate_previous_model,
        )
        if decision.should_rollback:
            await self._log_rollback_event(decision)
            logger.error(
                "Model rollback executed",
                failed_version=decision.model_version,
                restored_version=decision.previous_version,
                error_rate=decision.error_rate,
                request_count=decision.request_count,
            )
        return decision

    async def _activate_previous_model(self, failed_version: str, previous_version: str) -> None:
        async with self.db.tx() as tx:
            previous = await tx.modelversion.find_unique(where={"version": previous_version})
            failed = await tx.modelversion.find_unique(where={"version": failed_version})
            if not previous or not failed:
                raise RuntimeError("Rollback target model no longer exists.")
            await tx.modelversion.update(
                where={"id": failed.id},
                data={"status": ModelStatus.ARCHIVED},
            )
            await tx.modelversion.update(
                where={"id": previous.id},
                data={"status": ModelStatus.ACTIVE, "activatedAt": datetime.now(UTC)},
            )
        await self.poll_active_model()

    async def _log_rollback_event(self, decision) -> None:
        if not decision.model_version or not decision.previous_version:
            return
        try:
            await self.db.execute_raw(
                """
                INSERT INTO model_rollbacks (
                    failed_version,
                    restored_version,
                    error_rate,
                    request_count
                )
                VALUES ($1, $2, $3, $4)
                """,
                decision.model_version,
                decision.previous_version,
                decision.error_rate,
                decision.request_count,
            )
        except Exception as e:
            logger.debug("Unable to persist rollback audit event", error=str(e))

    async def embed_batch(
        self, texts: list[str], model_version: str = "auto"
    ) -> tuple[list[list[float]], str, int]:
        """Returns (embeddings, model_version_used, dimension)"""
        async with self._lock:
            version = self.current_version
            session = self.session
            input_name = self.onnx_input_name
            output_name = self.onnx_output_name
            model_kind = self.model_kind

        if not version:
            raise RuntimeError("No active model is loaded.")

        if model_version == "baseline":
            return embed_texts(texts, self.dimension), "baseline", self.dimension

        if model_version not in {"auto", version}:
            raise RuntimeError(f"Requested model '{model_version}' is not active.")

        if session and model_kind == "encoder":
            # The adapted model *is* the embedding model, so the base encoder is bypassed
            # entirely rather than having its output post-processed.
            vectors = encode_with_session(session, get_tokenizer(), texts)
            return [[float(value) for value in row] for row in vectors], version, self.dimension

        embeddings = embed_texts(texts, self.dimension)
        if session:
            input_array = np.array(embeddings, dtype=np.float32)
            output = session.run([output_name], {input_name: input_array})[0]
            embeddings = output.astype(float).tolist()

        return embeddings, version, self.dimension

    async def _load_artifact_manifest(
        self, artifact_uri: str | None, expected_sha256: str | None
    ) -> dict[str, Any] | None:
        if not artifact_uri or artifact_uri.startswith("builtin://"):
            return None

        if artifact_uri.endswith(".onnx"):
            logger.info(
                "ONNX artifact registered; manifest verification skipped for binary artifact."
            )
            return None

        bucket, object_name = parse_s3_uri(artifact_uri)
        endpoint = (
            str(settings.s3_endpoint).replace("http://", "").replace("https://", "").rstrip("/")
        )
        client = Minio(
            endpoint=endpoint,
            access_key=settings.s3_access_key_id,
            secret_key=settings.s3_secret_access_key,
            secure=str(settings.s3_endpoint).startswith("https://"),
        )

        response = client.get_object(bucket, object_name)
        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()

        actual_sha256 = sha256_hex(data)
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(
                f"Artifact checksum mismatch for {artifact_uri}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        manifest = decode_manifest(data)
        if manifest.get("embedding_dim") != self.dimension:
            raise ValueError(
                f"Artifact dimension mismatch: expected {self.dimension}, "
                f"got {manifest.get('embedding_dim')}"
            )
        return manifest

    async def _load_onnx_session(
        self, manifest: dict[str, Any]
    ) -> tuple[ort.InferenceSession, str, str, str]:
        onnx_metadata = manifest.get("onnx")
        if not isinstance(onnx_metadata, dict):
            raise ValueError("Model manifest does not contain ONNX metadata.")

        onnx_uri = onnx_metadata.get("uri")
        onnx_sha256 = onnx_metadata.get("sha256")
        if not isinstance(onnx_uri, str) or not isinstance(onnx_sha256, str):
            raise ValueError("Model manifest ONNX metadata is incomplete.")

        # Checked before the artifact is fetched: an unservable manifest should not cost a
        # download. Absent metadata means a legacy projection artifact, which is what the
        # demo adapter produces; only PEFT exports declare themselves as encoders.
        kind = str(onnx_metadata.get("kind", "projection"))
        if kind not in {"projection", "encoder"}:
            raise ValueError(f"Unknown ONNX model kind {kind!r} in manifest")

        data = await self._download_s3_object(onnx_uri)
        actual_sha256 = sha256_hex(data)
        if actual_sha256 != onnx_sha256:
            raise ValueError(
                f"ONNX checksum mismatch for {onnx_uri}: "
                f"expected {onnx_sha256}, got {actual_sha256}"
            )

        with NamedTemporaryFile(suffix=".onnx", delete=False) as artifact_file:
            artifact_file.write(data)
            artifact_path = artifact_file.name

        try:
            session = ort.InferenceSession(artifact_path, providers=["CPUExecutionProvider"])
        finally:
            Path(artifact_path).unlink(missing_ok=True)

        return (
            session,
            str(onnx_metadata.get("input_name", "input")),
            str(onnx_metadata.get("output_name", "embeddings")),
            kind,
        )

    async def _download_s3_object(self, uri: str) -> bytes:
        bucket, object_name = parse_s3_uri(uri)
        endpoint = (
            str(settings.s3_endpoint).replace("http://", "").replace("https://", "").rstrip("/")
        )
        client = Minio(
            endpoint=endpoint,
            access_key=settings.s3_access_key_id,
            secret_key=settings.s3_secret_access_key,
            secure=str(settings.s3_endpoint).startswith("https://"),
        )

        response = client.get_object(bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()


engine = ModelEngine()


async def background_poller():
    while True:
        await asyncio.sleep(10)
        try:
            await engine.poll_active_model()
        except Exception as e:
            logger.error("Error polling active model", error=str(e))
