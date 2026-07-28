from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import structlog
from continuum_shared.artifacts import sha256_hex
from continuum_shared.config import settings
from continuum_shared.prisma import Prisma
from minio import Minio
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class TrainingText(BaseModel):
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    domain_tag: str = Field(min_length=1)


class PeftTrainingConfig(BaseModel):
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("query", "key", "value", "dense")
    task_type: str = "FEATURE_EXTRACTION"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-4
    temperature: float = 0.05
    max_length: int = 256
    output_dir: str | None = None


class PeftTrainingTelemetry(BaseModel):
    sample_count: int
    loss_history: list[dict[str, float]]
    domain_tag: str


class PeftArtifactSet(BaseModel):
    adapter_config_uri: str
    onnx_uri: str
    onnx_sha256: str
    onnx_bytes: int
    domain_tag: str


class PeftTrainingResult(BaseModel):
    telemetry: PeftTrainingTelemetry
    artifacts: PeftArtifactSet


@dataclass(frozen=True)
class PeftDependencies:
    datasets: Any
    peft: Any
    torch: Any
    transformers: Any


def load_peft_dependencies() -> PeftDependencies:
    try:
        import datasets
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "Real PEFT training requires torch, transformers, peft, and datasets. "
            "Install the trainer PEFT extra before setting TRAINER_BACKEND=peft."
        ) from exc

    return PeftDependencies(
        datasets=datasets,
        peft=peft,
        torch=torch,
        transformers=transformers,
    )


def build_contrastive_dataset(training_texts: Sequence[TrainingText], datasets_module: Any) -> Any:
    if len(training_texts) < 2:
        raise ValueError("PEFT training requires at least two training texts.")
    return datasets_module.Dataset.from_dict({"text": [item.text for item in training_texts]})


def infer_domain_tag(training_texts: Sequence[TrainingText]) -> str:
    counts: dict[str, int] = {}
    for item in training_texts:
        counts[item.domain_tag] = counts.get(item.domain_tag, 0) + 1
    if not counts:
        return "unknown"
    return max(sorted(counts), key=lambda key: counts[key])


def mean_pool_and_normalize(outputs: Any, attention_mask: Any, torch_module: Any) -> Any:
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch_module.sum(token_embeddings * mask, dim=1)
    counts = torch_module.clamp(mask.sum(dim=1), min=1e-9)
    embeddings = summed / counts
    return torch_module.nn.functional.normalize(embeddings, p=2, dim=1)


def build_in_batch_trainer_class(deps: PeftDependencies) -> type:
    class InBatchContrastiveTrainer(deps.transformers.Trainer):  # type: ignore[misc]
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            **_: Any,
        ) -> Any:
            labels = inputs.pop("labels", None)
            outputs = model(**inputs)
            embeddings = mean_pool_and_normalize(outputs, inputs["attention_mask"], deps.torch)
            logits = embeddings @ embeddings.T / self.args.temperature
            targets = deps.torch.arange(logits.shape[0], device=logits.device)
            loss = deps.torch.nn.functional.cross_entropy(logits, targets)
            if labels is not None:
                inputs["labels"] = labels
            return (loss, outputs) if return_outputs else loss

    return InBatchContrastiveTrainer


def build_lora_config(config: PeftTrainingConfig, peft_module: Any) -> Any:
    task_type = getattr(peft_module.TaskType, config.task_type)
    return peft_module.LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.target_modules),
        lora_dropout=config.lora_dropout,
        task_type=task_type,
    )


def tokenize_dataset(dataset: Any, tokenizer: Any, max_length: int) -> Any:
    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def train_peft_model(
    training_texts: Sequence[TrainingText],
    config: PeftTrainingConfig,
    deps: PeftDependencies | None = None,
    exporter: Callable[[Path, Path, str], None] | None = None,
) -> tuple[PeftTrainingTelemetry, Path, Path]:
    deps = deps or load_peft_dependencies()
    exporter = exporter or export_onnx_with_optimum
    dataset = build_contrastive_dataset(training_texts, deps.datasets)
    domain_tag = infer_domain_tag(training_texts)

    output_root = Path(config.output_dir) if config.output_dir else None
    root = output_root or Path(mkdtemp(prefix="continuum-peft-"))
    adapter_dir = root / "adapter"
    onnx_dir = root / "onnx"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = deps.transformers.AutoTokenizer.from_pretrained(config.base_model)
    base_model = deps.transformers.AutoModel.from_pretrained(config.base_model)
    lora_config = build_lora_config(config, deps.peft)
    model = deps.peft.get_peft_model(base_model, lora_config)
    tokenized = tokenize_dataset(dataset, tokenizer, config.max_length)
    tokenized = tokenized.with_format("torch")

    trainer_cls = build_in_batch_trainer_class(deps)
    training_args = deps.transformers.TrainingArguments(
        output_dir=str(root / "checkpoints"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )
    setattr(training_args, "temperature", config.temperature)
    trainer = trainer_cls(model=model, args=training_args, train_dataset=tokenized)
    train_output = trainer.train()

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    exporter(adapter_dir, onnx_dir, "feature-extraction")

    loss_history = extract_loss_history(trainer.state.log_history)
    if not loss_history and getattr(train_output, "training_loss", None) is not None:
        loss_history = [{"step": 0, "loss": float(train_output.training_loss)}]

    telemetry = PeftTrainingTelemetry(
        sample_count=len(training_texts),
        loss_history=loss_history,
        domain_tag=domain_tag,
    )
    return telemetry, adapter_dir, onnx_dir


def extract_loss_history(log_history: Sequence[dict[str, Any]]) -> list[dict[str, float]]:
    losses = []
    for entry in log_history:
        if "loss" in entry:
            losses.append(
                {
                    "step": float(entry.get("step", len(losses))),
                    "loss": float(entry["loss"]),
                }
            )
    return losses


def export_onnx_with_optimum(model_dir: Path, output_dir: Path, task: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "optimum-cli",
            "export",
            "onnx",
            "--model",
            str(model_dir),
            "--task",
            task,
            str(output_dir),
        ],
        check=True,
    )


def find_onnx_file(onnx_dir: Path) -> Path:
    files = sorted(onnx_dir.glob("*.onnx"))
    if not files:
        raise FileNotFoundError(f"No ONNX artifact was exported under {onnx_dir}")
    return files[0]


def minio_client() -> Minio:
    endpoint = str(settings.s3_endpoint).replace("http://", "").replace("https://", "").rstrip("/")
    return Minio(
        endpoint=endpoint,
        access_key=settings.s3_access_key_id,
        secret_key=settings.s3_secret_access_key,
        secure=str(settings.s3_endpoint).startswith("https://"),
    )


def upload_peft_artifacts(
    *,
    version: str,
    telemetry: PeftTrainingTelemetry,
    adapter_dir: Path,
    onnx_dir: Path,
    client: Minio | None = None,
) -> PeftArtifactSet:
    client = client or minio_client()
    if not client.bucket_exists(settings.s3_bucket_models):
        client.make_bucket(settings.s3_bucket_models)

    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        adapter_config = adapter_dir / "config.json"
    adapter_payload = adapter_config.read_bytes() if adapter_config.exists() else b"{}"
    adapter_object = f"{version}/peft-adapter-config.json"
    client.put_object(
        bucket_name=settings.s3_bucket_models,
        object_name=adapter_object,
        data=BytesIO(adapter_payload),
        length=len(adapter_payload),
        content_type="application/json",
        metadata={"sha256": sha256_hex(adapter_payload)},
    )

    onnx_file = find_onnx_file(onnx_dir)
    onnx_payload = onnx_file.read_bytes()
    onnx_sha256 = sha256_hex(onnx_payload)
    onnx_object = f"{version}/model.onnx"
    client.put_object(
        bucket_name=settings.s3_bucket_models,
        object_name=onnx_object,
        data=BytesIO(onnx_payload),
        length=len(onnx_payload),
        content_type="application/octet-stream",
        metadata={"sha256": onnx_sha256},
    )

    metadata = {
        "version": version,
        "domain_tag": telemetry.domain_tag,
        "created_at": datetime.now(UTC).isoformat(),
        "adapter_config_uri": f"s3://{settings.s3_bucket_models}/{adapter_object}",
        "onnx_uri": f"s3://{settings.s3_bucket_models}/{onnx_object}",
        "onnx_sha256": onnx_sha256,
        "sample_count": telemetry.sample_count,
        "loss_history": telemetry.loss_history,
    }
    metadata_payload = json.dumps(metadata, sort_keys=True).encode("utf-8")
    client.put_object(
        bucket_name=settings.s3_bucket_models,
        object_name=f"{version}/peft-training-metadata.json",
        data=BytesIO(metadata_payload),
        length=len(metadata_payload),
        content_type="application/json",
        metadata={"sha256": sha256_hex(metadata_payload)},
    )

    return PeftArtifactSet(
        adapter_config_uri=f"s3://{settings.s3_bucket_models}/{adapter_object}",
        onnx_uri=f"s3://{settings.s3_bucket_models}/{onnx_object}",
        onnx_sha256=onnx_sha256,
        onnx_bytes=len(onnx_payload),
        domain_tag=telemetry.domain_tag,
    )


async def load_drifted_training_texts(
    db: Prisma, drift_window_id: str | None, limit: int = 1000
) -> list[TrainingText]:
    if drift_window_id:
        rows = await db.query_raw(
            """
            SELECT d.text, d.source
            FROM documents d
            JOIN drift_windows w
              ON d.ingested_at >= w.window_start
             AND d.ingested_at < w.window_end
            WHERE w.id = $1::uuid
            ORDER BY d.ingested_at DESC
            LIMIT $2
            """,
            drift_window_id,
            limit,
        )
    else:
        rows = await db.query_raw(
            """
            SELECT text, source
            FROM documents
            ORDER BY ingested_at DESC
            LIMIT $1
            """,
            limit,
        )

    return [
        TrainingText(
            text=str(row["text"]),
            source=str(row["source"]),
            domain_tag=str(row["source"]),
        )
        for row in rows
    ]


async def mark_model_pending_eval(
    db: Prisma,
    *,
    model_id: str,
    artifacts: PeftArtifactSet,
    eval_mrr: float | None,
) -> None:
    await db.execute_raw(
        """
        UPDATE model_versions
        SET
            status = 'PENDING_EVAL'::"ModelStatus",
            domain_tag = $2,
            onnx_path = $3,
            eval_mrr = $4,
            artifact_uri = $3,
            artifact_sha256 = $5,
            artifact_bytes = $6,
            lora_rank = 8,
            updated_at = NOW()
        WHERE id = $1::uuid
        """,
        model_id,
        artifacts.domain_tag,
        artifacts.onnx_uri,
        eval_mrr,
        artifacts.onnx_sha256,
        artifacts.onnx_bytes,
    )


async def run_peft_training_from_db(
    db: Prisma,
    *,
    model_id: str,
    version: str,
    base_model: str,
    drift_window_id: str | None,
    config: PeftTrainingConfig | None = None,
) -> PeftTrainingResult:
    resolved_config = config or PeftTrainingConfig(base_model=base_model)
    training_texts = await load_drifted_training_texts(db, drift_window_id)
    telemetry, adapter_dir, onnx_dir = train_peft_model(training_texts, resolved_config)
    artifacts = upload_peft_artifacts(
        version=version,
        telemetry=telemetry,
        adapter_dir=adapter_dir,
        onnx_dir=onnx_dir,
    )
    await mark_model_pending_eval(
        db,
        model_id=model_id,
        artifacts=artifacts,
        eval_mrr=None,
    )
    return PeftTrainingResult(telemetry=telemetry, artifacts=artifacts)
