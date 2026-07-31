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

import numpy as np
import structlog
from continuum_shared.artifacts import sha256_hex
from continuum_shared.config import settings
from continuum_shared.pairs import build_pairs
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
    # Sequences are padded to exactly this length, so it is a direct multiplier on
    # training cost rather than a ceiling. Corpus documents run 72-83 words at the median
    # (~110 tokens), so 256 spent more than half of every batch on padding that the
    # attention mask then discards. Serving still embeds at MAX_SEQUENCE_LENGTH.
    max_length: int = 128
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
    """Build (query, document) positives from the drifted window.

    Dropout views of one text were the previous positives. They carry no information about
    which documents ought to be close, so the objective could only push everything apart,
    and four training runs moved retrieval by between -0.21% and +1.25%. The opening of a
    post against its own body is a positive with actual semantic content, and it matches
    how the model is scored.
    """
    if len(training_texts) < 2:
        raise ValueError("PEFT training requires at least two training texts.")

    pairs = build_pairs([item.text for item in training_texts])
    if len(pairs) < 2:
        raise ValueError(
            "PEFT training requires at least two documents long enough to split into a "
            f"query and a body; {len(training_texts)} texts yielded {len(pairs)} pairs."
        )

    return datasets_module.Dataset.from_dict(
        {
            "query": [query for query, _ in pairs],
            "document": [document for _, document in pairs],
        }
    )


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


def in_batch_contrastive_loss(
    view_a: Any,
    view_b: Any,
    temperature: float,
    torch_module: Any,
) -> Any:
    """Symmetric InfoNCE over two dropout views of the same batch.

    Row i of the similarity matrix pairs view A of document i against view B of every
    document, so the positive on the diagonal is a similarity the model has to actually
    maximise. Comparing a batch against itself instead puts self-similarity on the
    diagonal, which is 1.0 for any L2-normalised embedding: the target is satisfied
    before training starts and the only remaining gradient pushes every document apart.
    """
    logits = view_a @ view_b.T / temperature
    targets = torch_module.arange(logits.shape[0], device=logits.device)
    forward = torch_module.nn.functional.cross_entropy(logits, targets)
    backward = torch_module.nn.functional.cross_entropy(logits.T, targets)
    return (forward + backward) / 2


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
            # The query side and the document side are genuinely different text, so the
            # diagonal of the similarity matrix is a positive the model has to earn from
            # meaning. Encoding one text twice under dropout, as this did before, gave a
            # positive that carried no information about which documents belong together.
            query_mask = inputs["query_attention_mask"]
            document_mask = inputs["document_attention_mask"]
            query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=query_mask)
            document_outputs = model(
                input_ids=inputs["document_input_ids"], attention_mask=document_mask
            )
            queries = mean_pool_and_normalize(query_outputs, query_mask, deps.torch)
            documents = mean_pool_and_normalize(document_outputs, document_mask, deps.torch)
            loss = in_batch_contrastive_loss(queries, documents, self.args.temperature, deps.torch)
            if labels is not None:
                inputs["labels"] = labels
            return (loss, query_outputs) if return_outputs else loss

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
    """Tokenise both sides of each pair into separately named columns."""

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        queries = tokenizer(
            batch["query"], padding="max_length", truncation=True, max_length=max_length
        )
        documents = tokenizer(
            batch["document"], padding="max_length", truncation=True, max_length=max_length
        )
        return {
            "query_input_ids": queries["input_ids"],
            "query_attention_mask": queries["attention_mask"],
            "document_input_ids": documents["input_ids"],
            "document_attention_mask": documents["attention_mask"],
        }

    return dataset.map(tokenize, batched=True, remove_columns=["query", "document"])


def train_peft_model(
    training_texts: Sequence[TrainingText],
    config: PeftTrainingConfig,
    deps: PeftDependencies | None = None,
    exporter: Callable[[Path, Path, str], None] | None = None,
    validator: Callable[[Path, int], int] | None = None,
) -> tuple[PeftTrainingTelemetry, Path, Path]:
    deps = deps or load_peft_dependencies()
    exporter = exporter or export_onnx_with_optimum
    validator = validator or validate_onnx_export
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

    # Counted before merge_and_unload, which folds the adapter into the base weights and
    # leaves nothing marked trainable to count.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA parameter budget",
        trainable_parameters=trainable,
        total_parameters=total,
        trainable_percent=round(100.0 * trainable / max(total, 1), 4),
    )

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    exporter(adapter_dir, onnx_dir, "feature-extraction")
    validator(find_onnx_file(onnx_dir), int(merged_model.config.hidden_size))

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


def validate_onnx_export(onnx_path: Path, expected_dim: int) -> int:
    """Load the exported graph and confirm it runs and emits the expected width.

    Exporting only proves optimum-cli exited zero. It does not prove the graph loads, that
    its inputs are wired the way the serving engine calls them, or that the hidden width
    survived the adapter merge. A model that fails any of those still reaches the registry
    and is served, producing vectors that are incomparable with the baseline centroids
    drift is measured against — with no error anywhere along the way.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {}
    for model_input in session.get_inputs():
        # Symbolic dims come back as strings; a 1x8 probe is enough to exercise the graph.
        shape = [1 if isinstance(dim, str) or dim is None else dim for dim in model_input.shape]
        if len(shape) >= 2:
            shape[1] = 8
        feeds[model_input.name] = np.ones(shape, dtype=np.int64)

    outputs = session.run(None, feeds)
    if not outputs:
        raise ValueError(f"ONNX export at {onnx_path} produced no outputs")

    actual_dim = int(outputs[0].shape[-1])
    if actual_dim != expected_dim:
        raise ValueError(
            f"ONNX export at {onnx_path} emits {actual_dim}-dimensional vectors, "
            f"expected {expected_dim} from the merged model config"
        )
    return actual_dim


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


async def load_recent_document_rows(db: Prisma, limit: int) -> list[dict]:
    return await db.query_raw(
        """
        SELECT text, source
        FROM documents
        ORDER BY ingested_at DESC
        LIMIT $1
        """,
        limit,
    )


async def load_drifted_training_texts(
    db: Prisma, drift_window_id: str | None, limit: int = 1000
) -> list[TrainingText]:
    if drift_window_id:
        # Membership follows embeddings.created_at, matching how the drift service builds a
        # window's centroid. Selecting on documents.ingested_at instead described a
        # different set entirely: embedding runs well behind ingestion once a real model is
        # doing the work, so by the time a window breached, that slice held hundreds of
        # embeddings and no newly ingested documents, and training died on an empty corpus.
        rows = await db.query_raw(
            """
            SELECT d.text, d.source
            FROM embeddings e
            JOIN documents d ON d.id = e.document_id
            JOIN drift_windows w
              ON e.created_at >= w.window_start
             AND e.created_at < w.window_end
            WHERE w.id = $1::uuid
            ORDER BY e.created_at DESC
            LIMIT $2
            """,
            drift_window_id,
            limit,
        )
        if len(rows) < 2:
            # An unlucky window boundary should not kill an adaptation run. Falling back to
            # the most recent documents keeps the job alive, and the warning makes the
            # degraded selection visible rather than silently training on the wrong set.
            logger.warning(
                "Drift window yielded too few documents to train on, using recent documents",
                drift_window_id=drift_window_id,
                window_documents=len(rows),
            )
            rows = await load_recent_document_rows(db, limit)
    else:
        rows = await load_recent_document_rows(db, limit)

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
