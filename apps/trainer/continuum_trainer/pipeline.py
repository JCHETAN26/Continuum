import asyncio
from datetime import UTC, datetime
from io import BytesIO

import onnx
import structlog
import torch
import torch.nn.functional as torch_functional
from continuum_shared.artifacts import build_demo_artifact_manifest, encode_manifest, sha256_hex
from continuum_shared.config import settings
from continuum_shared.embeddings import embed_texts
from continuum_shared.prisma import Json, Prisma
from continuum_shared.prisma.enums import ModelStatus, TrainingJobStatus
from minio import Minio
from torch import nn

from continuum_trainer.eval import evaluate_model

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
            await db.trainingjob.update(
                where={"id": training_job_id},
                data={
                    "status": TrainingJobStatus.RUNNING,
                    "startedAt": datetime.now(UTC),
                    "attempts": 1,
                },
            )

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

        if training_job_id:
            await db.trainingjob.update(
                where={"id": training_job_id},
                data={
                    "status": TrainingJobStatus.SUCCEEDED if passed else TrainingJobStatus.FAILED,
                    "finishedAt": datetime.now(UTC),
                    "sampleCount": telemetry["sample_count"],
                    "lossHistory": Json(telemetry["loss_history"]),
                    "error": None if passed else "Model did not exceed the MRR improvement gate.",
                },
            )

        logger.info("Training pipeline completed", version=model_version.version, status=new_status)

    finally:
        await db.disconnect()


class EmbeddingAdapter(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.projection = nn.Linear(dimension, dimension, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(dimension))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch_functional.normalize(self.projection(input), p=2, dim=1)


async def _run_corpus_adapter_training(db: Prisma) -> tuple[dict, bytes, list[dict]]:
    """Train a small projection adapter from ingested embeddings and hard negatives."""

    examples = await load_training_examples(db)
    triplets = build_training_triplets(examples)
    model = EmbeddingAdapter(settings.embedding_dim)

    if not triplets:
        return (
            {
                "sample_count": len(examples),
                "loss_history": [{"step": 0, "loss": 0.0}],
            },
            export_adapter_to_onnx(model, settings.embedding_dim),
            examples,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
    anchor = torch.tensor([triplet[0] for triplet in triplets], dtype=torch.float32)
    positive = torch.tensor([triplet[1] for triplet in triplets], dtype=torch.float32)
    negative = torch.tensor([triplet[2] for triplet in triplets], dtype=torch.float32)

    loss_history = []
    for epoch in range(1, 41):
        optimizer.zero_grad()
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        positive_distance = 1 - torch_functional.cosine_similarity(anchor_out, positive_out)
        negative_distance = 1 - torch_functional.cosine_similarity(anchor_out, negative_out)
        loss = torch.relu(0.2 + positive_distance - negative_distance).mean()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 5 == 0:
            loss_history.append({"step": epoch, "loss": round(float(loss.detach()), 6)})
        await asyncio.sleep(0)

    return (
        {
            "sample_count": len(triplets) * 3,
            "loss_history": loss_history,
        },
        export_adapter_to_onnx(model, settings.embedding_dim),
        examples,
    )


async def load_training_examples(db: Prisma) -> list[dict]:
    rows = await db.query_raw(
        """
        SELECT d.source, d.text, e.vector::text AS vec_str
        FROM embeddings e
        JOIN documents d ON d.id = e.document_id
        ORDER BY e.created_at DESC
        LIMIT 1000
        """
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
    model.eval()
    dummy_input = torch.zeros((1, dimension), dtype=torch.float32)
    buffer = BytesIO()
    torch.onnx.export(
        model,
        dummy_input,
        buffer,
        input_names=["input"],
        output_names=["embeddings"],
        dynamic_axes={"input": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    artifact = buffer.getvalue()
    onnx.checker.check_model(artifact)
    return artifact
