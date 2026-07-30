import asyncio
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import numpy as np
import onnx
import structlog
from continuum_shared.artifacts import (
    build_demo_artifact_manifest,
    build_peft_artifact_manifest,
    encode_manifest,
    parse_s3_uri,
    sha256_hex,
)
from continuum_shared.config import settings
from continuum_shared.embeddings import embed_texts
from continuum_shared.prisma import Json, Prisma
from continuum_shared.prisma.enums import ModelStatus, TrainingJobStatus
from minio import Minio

from continuum_trainer.eval import evaluate_encoder_model, evaluate_model
from continuum_trainer.peft_engine import run_peft_training_from_db

logger = structlog.get_logger()


# We use asyncio to interface with the async Prisma client inside a synchronous RQ task
def run_training_pipeline(model_id: str, training_job_id: str | None = None):
    asyncio.run(_async_run_training_pipeline(model_id, training_job_id))


async def _async_run_training_pipeline(model_id: str, training_job_id: str | None = None):
    db = Prisma()
    await db.connect()

    try:
        model_version = await db.modelversion.find_unique(where={"id": model_id})
        if not model_version:
            logger.error("Model not found", model_id=model_id)
            return

        logger.info("Starting training pipeline", version=model_version.version)

        # Mark as evaluating/training
        await db.modelversion.update(
            where={"id": model_id}, data={"status": ModelStatus.EVALUATING}
        )

        if training_job_id:
            await mark_training_job_running(db, training_job_id)

        if settings.trainer_backend == "peft":
            drift_window_id = await get_training_job_drift_window_id(db, training_job_id)
            peft_result = await run_peft_training_from_db(
                db,
                model_id=model_id,
                version=model_version.version,
                base_model=model_version.baseModel,
                drift_window_id=drift_window_id,
            )
            # The adapted encoder is scored against the base model on the same held-out
            # documents, so the comparison isolates what the adapter changed. Previously
            # this branch returned here, leaving every PEFT model at PENDING_EVAL: trained,
            # never measured, never served.
            examples = await load_training_examples(db)
            onnx_artifact = await _download_model_object(peft_result.artifacts.onnx_uri)
            passed, metrics, baseline_metrics, improvement = await evaluate_encoder_model(
                model_version.version, onnx_artifact, examples
            )

            artifact_uri, artifact_sha256, artifact_bytes = await _export_peft_artifact(
                version=model_version.version,
                base_model=model_version.baseModel,
                artifacts=peft_result.artifacts,
                telemetry=peft_result.telemetry,
                metrics=metrics,
                baseline_metrics=baseline_metrics,
                improvement_pct=improvement,
            )

            await db.modelversion.update(
                where={"id": model_id},
                data={
                    "status": ModelStatus.PASSED if passed else ModelStatus.REJECTED,
                    "artifactUri": artifact_uri,
                    "artifactSha256": artifact_sha256,
                    "artifactBytes": artifact_bytes,
                    "loraRank": 8,
                    "metrics": Json(metrics),
                    "baselineMetrics": Json(baseline_metrics),
                    "improvementPct": improvement,
                },
            )

            if passed:
                # Activation is all that is needed to trigger re-indexing. The embedding
                # worker claims documents whose vector was produced by a different model
                # version, so the corpus is re-encoded there: resumable across restarts,
                # and spread over however many worker replicas are running. Doing it here
                # blocked the training job for the length of a full re-index.
                await activate_model_version(db, model_id)

            if training_job_id:
                await db.trainingjob.update(
                    where={"id": training_job_id},
                    data={
                        "status": TrainingJobStatus.SUCCEEDED,
                        "finishedAt": datetime.now(UTC),
                        "sampleCount": peft_result.telemetry.sample_count,
                        "lossHistory": Json(peft_result.telemetry.loss_history),
                        "error": None,
                    },
                )
            logger.info(
                "PEFT training pipeline completed",
                version=model_version.version,
                domain=peft_result.artifacts.domain_tag,
                onnx_uri=peft_result.artifacts.onnx_uri,
                baseline_mrr=baseline_metrics["mrr"],
                candidate_mrr=metrics["mrr"],
                improvement_pct=improvement,
                activated=passed,
            )
            return

        telemetry, onnx_artifact, examples = await _run_corpus_adapter_training(db)
        passed, metrics, baseline_metrics, improvement = await evaluate_model(
            model_version.version, onnx_artifact, examples
        )

        # Determine status
        new_status = ModelStatus.PASSED if passed else ModelStatus.REJECTED

        artifact_uri, artifact_sha256, artifact_bytes = await _export_trained_artifact(
            version=model_version.version,
            base_model=model_version.baseModel,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            improvement_pct=improvement,
            onnx_artifact=onnx_artifact,
        )

        # Update database
        await db.modelversion.update(
            where={"id": model_id},
            data={
                "status": new_status,
                "artifactUri": artifact_uri,
                "artifactSha256": artifact_sha256,
                "artifactBytes": artifact_bytes,
                "loraRank": 8,
                "metrics": Json(metrics),
                "baselineMetrics": Json(baseline_metrics),
                "improvementPct": improvement,
            },
        )

        if passed:
            await activate_model_version(db, model_id)

        if training_job_id:
            await db.trainingjob.update(
                where={"id": training_job_id},
                data={
                    # The job ran to completion either way. A candidate that misses the
                    # improvement gate is a correct decision, not a failure, and the
                    # outcome is carried by ModelStatus.PASSED / REJECTED. Conflating the
                    # two made "rejected because it did not help" indistinguishable from
                    # "crashed", and left the worker's genuine failure path (which sets
                    # FAILED after exhausting retries) with no distinct signal.
                    "status": TrainingJobStatus.SUCCEEDED,
                    "finishedAt": datetime.now(UTC),
                    "sampleCount": telemetry["sample_count"],
                    "lossHistory": Json(telemetry["loss_history"]),
                    "error": None,
                },
            )

        logger.info("Training pipeline completed", version=model_version.version, status=new_status)

    finally:
        await db.disconnect()


async def activate_model_version(db: Prisma, model_id: str) -> None:
    active_models = await db.modelversion.find_many(where={"status": ModelStatus.ACTIVE})
    for active in active_models:
        if active.id != model_id:
            await db.modelversion.update(
                where={"id": active.id},
                data={"status": ModelStatus.PASSED},
            )

    await db.modelversion.update(
        where={"id": model_id},
        data={"status": ModelStatus.ACTIVE, "activatedAt": datetime.now(UTC)},
    )


async def mark_training_job_running(db: Prisma, training_job_id: str) -> None:
    training_job = await db.trainingjob.find_unique(where={"id": training_job_id})
    current_attempts = training_job.attempts if training_job else 0
    max_attempts = training_job.maxAttempts if training_job else 3
    await db.trainingjob.update(
        where={"id": training_job_id},
        data={
            "status": TrainingJobStatus.RUNNING,
            "startedAt": datetime.now(UTC),
            "attempts": min(current_attempts + 1, max_attempts),
        },
    )


async def get_training_job_drift_window_id(db: Prisma, training_job_id: str | None) -> str | None:
    if not training_job_id:
        return None
    training_job = await db.trainingjob.find_unique(where={"id": training_job_id})
    if not training_job:
        return None
    return training_job.driftWindowId


class EmbeddingAdapter:
    def __init__(self, dimension: int, weights: np.ndarray | None = None):
        self.dimension = dimension
        self.weights = (
            np.eye(dimension, dtype=np.float32) if weights is None else weights.astype(np.float32)
        )


async def _run_corpus_adapter_training(db: Prisma) -> tuple[dict, bytes, list[dict]]:
    """Train a small projection adapter from ingested embeddings and hard negatives."""

    examples = await load_training_examples(db)
    triplets = build_training_triplets(examples)
    model = train_adapter_from_examples(examples, settings.embedding_dim)

    if not triplets:
        return (
            {
                "sample_count": len(examples),
                "loss_history": [],
            },
            export_adapter_to_onnx(model, settings.embedding_dim),
            examples,
        )

    return (
        {
            "sample_count": len(triplets) * 3,
            # No trajectory: train_adapter_from_examples solves in closed form from the
            # between-source variance, so there are no optimisation steps to report. This
            # previously emitted a descending curve produced by walking a hardcoded margin
            # schedule against an unchanging model, which rendered on the dashboard as
            # training telemetry despite nothing having been trained.
            "loss_history": [],
        },
        export_adapter_to_onnx(model, settings.embedding_dim),
        examples,
    )


def train_adapter_from_examples(examples: list[dict], dimension: int) -> EmbeddingAdapter:
    """Build a small deterministic projection by emphasizing domain-separating dimensions."""

    by_source: dict[str, list[list[float]]] = {}
    for example in examples:
        by_source.setdefault(str(example["source"]), []).append(example["vector"])

    if len(by_source) < 2:
        return EmbeddingAdapter(dimension)

    centroids = np.array(
        [np.mean(np.array(vectors, dtype=np.float32), axis=0) for vectors in by_source.values()],
        dtype=np.float32,
    )
    between_source_variance = np.var(centroids, axis=0)
    max_variance = float(between_source_variance.max())
    if max_variance == 0:
        return EmbeddingAdapter(dimension)

    diagonal = 1.0 + (between_source_variance / max_variance)
    weights = np.diag(diagonal.astype(np.float32))
    return EmbeddingAdapter(dimension, weights)


def transform_vectors(model: EmbeddingAdapter, vectors: np.ndarray) -> np.ndarray:
    transformed = vectors @ model.weights
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (transformed / norms).astype(np.float32)


async def load_training_examples(db: Prisma, per_source: int = 250) -> list[dict]:
    """Take the most recent documents per source rather than overall.

    Retrieval is scored by whether a document's nearest neighbour shares its source, so an
    evaluation set drawn from a single domain cannot measure anything: it has no negatives
    and the scorer returns zeros. Ordering purely by embedding time made that reachable,
    because the drifted documents are ingested last and are therefore embedded last, so
    the most recent rows could be entirely one source. One run in three came back with
    baseline and candidate MRR both 0.0000 for exactly that reason.
    """
    rows = await db.query_raw(
        """
        SELECT source, text, vec_str FROM (
            SELECT
                d.source,
                d.text,
                e.vector::text AS vec_str,
                ROW_NUMBER() OVER (
                    PARTITION BY d.source ORDER BY e.created_at DESC
                ) AS source_rank
            FROM embeddings e
            JOIN documents d ON d.id = e.document_id
        ) ranked
        WHERE source_rank <= $1
        """,
        per_source,
    )
    examples = [
        {
            "source": row["source"],
            "text": row["text"],
            "vector": [float(value) for value in row["vec_str"].strip("[]").split(",")],
        }
        for row in rows
    ]
    if len(examples) >= 4 and len({example["source"] for example in examples}) >= 2:
        return examples

    fallback_texts = {
        "software": [
            "Refactoring a backend API handler for lower latency.",
            "Adding an index to improve PostgreSQL query performance.",
            "Debugging a Redis cache miss in a worker process.",
            "Deploying a TypeScript service through CI.",
        ],
        "healthcare": [
            "Patient presents with respiratory distress and elevated vitals.",
            "Cardiology follow-up after an abnormal echocardiogram.",
            "Prescribing Losartan for hypertension management.",
            "Blood test indicates elevated low-density lipoprotein.",
        ],
    }
    fallback_examples = []
    for source, texts in fallback_texts.items():
        for text, vector in zip(texts, embed_texts(texts, settings.embedding_dim)):
            fallback_examples.append({"source": source, "text": text, "vector": vector})
    return fallback_examples


def build_training_triplets(
    examples: list[dict], limit: int = 512
) -> list[tuple[list[float], ...]]:
    by_source: dict[str, list[dict]] = {}
    for example in examples:
        by_source.setdefault(str(example["source"]), []).append(example)

    triplets = []
    sources = sorted(by_source)
    for source in sources:
        positives = by_source[source]
        negatives = [
            candidate
            for other_source in sources
            if other_source != source
            for candidate in by_source[other_source]
        ]
        if len(positives) < 2 or not negatives:
            continue

        for index, anchor in enumerate(positives):
            positive = positives[(index + 1) % len(positives)]
            negative = negatives[index % len(negatives)]
            triplets.append((anchor["vector"], positive["vector"], negative["vector"]))
            if len(triplets) >= limit:
                return triplets
    return triplets


def _models_client() -> Minio:
    endpoint = str(settings.s3_endpoint).replace("http://", "").replace("https://", "").rstrip("/")
    return Minio(
        endpoint=endpoint,
        access_key=settings.s3_access_key_id,
        secret_key=settings.s3_secret_access_key,
        secure=str(settings.s3_endpoint).startswith("https://"),
    )


async def _download_model_object(uri: str) -> bytes:
    bucket, object_name = parse_s3_uri(uri)
    response = _models_client().get_object(bucket, object_name)
    try:
        return bytes(response.read())
    finally:
        response.close()
        response.release_conn()


async def _export_peft_artifact(
    *,
    version: str,
    base_model: str,
    artifacts: Any,
    telemetry: Any,
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    improvement_pct: float,
) -> tuple[str, str, int]:
    """Publish the manifest the serving engine loads for a LoRA-adapted encoder.

    The ONNX graph itself was already uploaded by the training step; this records where it
    lives and, critically, that it is an encoder rather than a projection.
    """
    manifest = build_peft_artifact_manifest(
        version=version,
        base_model=base_model,
        embedding_dim=settings.embedding_dim,
        domain_tag=artifacts.domain_tag,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        improvement_pct=improvement_pct,
        onnx_uri=artifacts.onnx_uri,
        onnx_sha256=artifacts.onnx_sha256,
        onnx_bytes=artifacts.onnx_bytes,
        adapter_config_uri=artifacts.adapter_config_uri,
        sample_count=telemetry.sample_count,
    )

    payload = encode_manifest(manifest)
    payload_sha256 = sha256_hex(payload)
    object_name = f"{version}/model-manifest.json"
    client = _models_client()
    if not client.bucket_exists(settings.s3_bucket_models):
        client.make_bucket(settings.s3_bucket_models)
    client.put_object(
        bucket_name=settings.s3_bucket_models,
        object_name=object_name,
        data=BytesIO(payload),
        length=len(payload),
        content_type="application/json",
        metadata={"sha256": payload_sha256},
    )

    return (
        f"s3://{settings.s3_bucket_models}/{object_name}",
        payload_sha256,
        len(payload),
    )


async def _export_trained_artifact(
    *,
    version: str,
    base_model: str,
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    improvement_pct: float,
    onnx_artifact: bytes,
) -> tuple[str, str, int]:
    manifest = build_demo_artifact_manifest(
        version=version,
        base_model=base_model,
        embedding_dim=settings.embedding_dim,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        improvement_pct=improvement_pct,
    )

    endpoint = str(settings.s3_endpoint).replace("http://", "").replace("https://", "").rstrip("/")
    client = Minio(
        endpoint=endpoint,
        access_key=settings.s3_access_key_id,
        secret_key=settings.s3_secret_access_key,
        secure=str(settings.s3_endpoint).startswith("https://"),
    )

    if not client.bucket_exists(settings.s3_bucket_models):
        client.make_bucket(settings.s3_bucket_models)

    onnx_sha256 = sha256_hex(onnx_artifact)
    onnx_object_name = f"{version}/adapter.onnx"
    client.put_object(
        bucket_name=settings.s3_bucket_models,
        object_name=onnx_object_name,
        data=BytesIO(onnx_artifact),
        length=len(onnx_artifact),
        content_type="application/octet-stream",
        metadata={"sha256": onnx_sha256},
    )

    manifest["onnx"] = {
        "uri": f"s3://{settings.s3_bucket_models}/{onnx_object_name}",
        "sha256": onnx_sha256,
        "bytes": len(onnx_artifact),
        "input_name": "input",
        "output_name": "embeddings",
    }

    artifact = encode_manifest(manifest)
    artifact_sha256 = sha256_hex(artifact)
    object_name = f"{version}/model-manifest.json"
    client.put_object(
        bucket_name=settings.s3_bucket_models,
        object_name=object_name,
        data=BytesIO(artifact),
        length=len(artifact),
        content_type="application/json",
        metadata={"sha256": artifact_sha256},
    )

    return f"s3://{settings.s3_bucket_models}/{object_name}", artifact_sha256, len(artifact)


def export_adapter_to_onnx(model: EmbeddingAdapter, dimension: int) -> bytes:
    weights = onnx.numpy_helper.from_array(model.weights, name="weights")
    input_tensor = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [None, dimension]
    )
    output_tensor = onnx.helper.make_tensor_value_info(
        "embeddings", onnx.TensorProto.FLOAT, [None, dimension]
    )
    matmul = onnx.helper.make_node("MatMul", ["input", "weights"], ["projected"])
    normalize = onnx.helper.make_node("LpNormalization", ["projected"], ["embeddings"], axis=1, p=2)
    graph = onnx.helper.make_graph(
        [matmul, normalize],
        "continuum-demo-adapter",
        [input_tensor],
        [output_tensor],
        [weights],
    )
    onnx_model = onnx.helper.make_model(
        graph,
        producer_name="continuum-trainer",
        opset_imports=[onnx.helper.make_opsetid("", 17)],
    )
    onnx_model.ir_version = 10

    buffer = BytesIO()
    buffer.write(onnx_model.SerializeToString())
    artifact = buffer.getvalue()
    onnx.checker.check_model(artifact)
    return artifact
