from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from continuum_trainer.peft_engine import (
    PeftDependencies,
    PeftTrainingConfig,
    PeftTrainingTelemetry,
    TrainingText,
    build_contrastive_dataset,
    build_lora_config,
    in_batch_contrastive_loss,
    infer_domain_tag,
    load_drifted_training_texts,
    mark_model_pending_eval,
    mine_hard_negatives,
    train_peft_model,
    upload_peft_artifacts,
    validate_onnx_export,
)


class FakeDataset:
    def __init__(self, payload: dict[str, list[str]]):
        self.payload = payload
        self.mapped = False
        self.formatted = False

    @property
    def column_names(self) -> list[str]:
        return list(self.payload)

    def map(self, fn, batched: bool, remove_columns: list[str]):
        assert batched is True
        # Every raw text column is tokenised and then dropped, including the mined
        # negative when hard negatives are on.
        assert remove_columns == [
            column for column in ("query", "document", "negative") if column in self.payload
        ]
        tokenized = fn(self.payload)
        remaining = {k: v for k, v in self.payload.items() if k not in remove_columns}
        mapped = FakeDataset(remaining)
        mapped.payload.update(tokenized)
        mapped.mapped = True
        return mapped

    def with_format(self, format_name: str):
        assert format_name == "torch"
        self.formatted = True
        return self


class FakeDatasetsModule:
    class Dataset:
        @staticmethod
        def from_dict(payload: dict[str, list[str]]) -> FakeDataset:
            return FakeDataset(payload)


class FakeTokenizer:
    @classmethod
    def from_pretrained(cls, model_name: str):
        assert model_name == "prajjwal1/bert-tiny"
        return cls()

    seen_lengths: list[int] = []

    def __call__(self, texts, padding: str, truncation: bool, max_length: int):
        assert padding == "max_length"
        assert truncation is True
        # Queries are short and documents are not, so the two sides are tokenised to
        # different lengths. A single shared length truncated 46% of documents while
        # padding the median query to five times its own length.
        FakeTokenizer.seen_lengths.append(max_length)
        return {
            "input_ids": [[1, 2, 0] for _ in texts],
            "attention_mask": [[1, 1, 0] for _ in texts],
        }

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


class FakeBaseModel:
    @classmethod
    def from_pretrained(cls, model_name: str):
        assert model_name == "prajjwal1/bert-tiny"
        return cls()


class FakeParameter:
    def __init__(self, count: int, trainable: bool):
        self._count = count
        self.requires_grad = trainable

    def numel(self) -> int:
        return self._count


class FakePeftModel:
    config = SimpleNamespace(hidden_size=128)

    def parameters(self):
        # A LoRA run has a small trainable slice over a frozen base.
        return [FakeParameter(1_000, False), FakeParameter(50, True)]

    def merge_and_unload(self):
        return self

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text('{"r":8}', encoding="utf-8")


class FakeTrainingArguments:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeTrainer:
    def __init__(self, model, args, train_dataset):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.state = SimpleNamespace(log_history=[{"step": 1, "loss": 0.4}])

    def train(self):
        return SimpleNamespace(training_loss=0.4)


class FakeTransformersModule:
    AutoTokenizer = FakeTokenizer
    AutoModel = FakeBaseModel
    TrainingArguments = FakeTrainingArguments
    Trainer = FakeTrainer


class FakePeftModule:
    TaskType = SimpleNamespace(FEATURE_EXTRACTION="FEATURE_EXTRACTION")
    captured_config = None

    class LoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            FakePeftModule.captured_config = kwargs

    @staticmethod
    def get_peft_model(base_model, lora_config):
        assert isinstance(base_model, FakeBaseModel)
        assert lora_config.kwargs["r"] == 8
        return FakePeftModel()


class FakeMinio:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def bucket_exists(self, bucket: str) -> bool:
        assert bucket
        return True

    def make_bucket(self, bucket: str) -> None:
        raise AssertionError(f"unexpected make_bucket({bucket})")

    def put_object(
        self,
        *,
        bucket_name: str,
        object_name: str,
        data,
        length: int,
        content_type: str,
        metadata: dict[str, str],
    ) -> None:
        payload = data.read()
        assert len(payload) == length
        assert content_type
        assert metadata["sha256"]
        self.objects[f"{bucket_name}/{object_name}"] = payload


def fake_deps() -> PeftDependencies:
    return PeftDependencies(
        datasets=FakeDatasetsModule,
        peft=FakePeftModule,
        torch=SimpleNamespace(),
        transformers=FakeTransformersModule,
    )


class NumpyTorchStub:
    """Just enough of the torch surface for the loss, so this runs without the ML extra."""

    @staticmethod
    def arange(count: int, device=None):
        return np.arange(count)

    @staticmethod
    def cat(matrices, dim: int):
        return np.concatenate(matrices, axis=dim)

    # Names mirror the torch API the loss calls into, so they intentionally break CapWords.
    class nn:  # noqa: N801
        class functional:  # noqa: N801
            @staticmethod
            def cross_entropy(logits, targets):
                shifted = logits - logits.max(axis=1, keepdims=True)
                log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
                return -float(np.mean(log_probs[np.arange(len(targets)), targets]))


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def test_contrastive_loss_penalises_misaligned_views():
    """The loss has to measure alignment between two views, not self-similarity.

    Comparing a batch against itself puts 1.0 on the diagonal for any L2-normalised
    embedding, so the target is already satisfied and only a repulsive gradient remains.
    Pairing view A against view B makes the diagonal something the model must earn.
    """
    rng = np.random.default_rng(0)
    view_a = unit_rows(rng.normal(size=(8, 16)))
    aligned = unit_rows(view_a + 0.01 * rng.normal(size=(8, 16)))
    misaligned = aligned[::-1]

    aligned_loss = in_batch_contrastive_loss(view_a, aligned, 0.05, NumpyTorchStub)
    misaligned_loss = in_batch_contrastive_loss(view_a, misaligned, 0.05, NumpyTorchStub)

    assert aligned_loss < misaligned_loss


def test_contrastive_loss_is_not_trivially_satisfied_by_normalisation():
    """Regression: the objective must not be solved before training starts.

    The original implementation passed a single view in both positions. With unit-norm
    embeddings that puts an unbeatable 1.0 on every diagonal entry, and the loss sat at
    ~1e-6 at random initialisation while real training stayed flat.
    """
    view_a = unit_rows(np.random.default_rng(1).normal(size=(8, 16)))
    view_b = unit_rows(view_a + 0.5 * np.random.default_rng(7).normal(size=(8, 16)))

    self_paired = in_batch_contrastive_loss(view_a, view_a, 0.05, NumpyTorchStub)
    two_view = in_batch_contrastive_loss(view_a, view_b, 0.05, NumpyTorchStub)

    # Pairing a batch with itself is solved at initialisation; two genuine views are not.
    assert self_paired < 1e-3
    assert two_view > 0.1


def test_contrastive_loss_is_symmetric_in_its_views():
    rng = np.random.default_rng(2)
    view_a = unit_rows(rng.normal(size=(6, 12)))
    view_b = unit_rows(rng.normal(size=(6, 12)))

    forward = in_batch_contrastive_loss(view_a, view_b, 0.05, NumpyTorchStub)
    reverse = in_batch_contrastive_loss(view_b, view_a, 0.05, NumpyTorchStub)

    assert forward == pytest.approx(reverse, rel=1e-9)


def build_tiny_onnx(path: Path, output_dim: int) -> Path:
    """A minimal graph shaped like a feature extractor, for validating the validator."""
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "s"])
    out = helper.make_tensor_value_info(
        "last_hidden_state", TensorProto.FLOAT, ["b", "s", output_dim]
    )
    table = helper.make_tensor(
        "table", TensorProto.FLOAT, [4, output_dim], [0.1] * (4 * output_dim)
    )
    node = helper.make_node("Gather", ["table", "input_ids"], ["last_hidden_state"], axis=0)
    graph = helper.make_graph([node], "tiny", [ids], [out], initializer=[table])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    path.write_bytes(model.SerializeToString())
    return path


def test_validate_onnx_export_accepts_the_expected_width(tmp_path: Path):
    path = build_tiny_onnx(tmp_path / "good.onnx", output_dim=384)

    assert validate_onnx_export(path, expected_dim=384) == 384


def test_validate_onnx_export_rejects_a_mismatched_width(tmp_path: Path):
    """A merge that changes the hidden width would otherwise reach the registry silently."""
    path = build_tiny_onnx(tmp_path / "narrow.onnx", output_dim=128)

    with pytest.raises(ValueError, match="emits 128-dimensional vectors, expected 384"):
        validate_onnx_export(path, expected_dim=384)


def test_validate_onnx_export_rejects_a_graph_that_does_not_load(tmp_path: Path):
    corrupt = tmp_path / "corrupt.onnx"
    corrupt.write_bytes(b"not an onnx graph")

    with pytest.raises(Exception):  # noqa: B017 - onnxruntime raises its own Fail type
        validate_onnx_export(corrupt, expected_dim=384)


def test_build_contrastive_dataset_requires_at_least_two_texts():
    with pytest.raises(ValueError, match="at least two"):
        build_contrastive_dataset(
            [TrainingText(text="one", source="medical", domain_tag="medical")],
            FakeDatasetsModule,
        )


def test_infer_domain_tag_uses_majority_source():
    texts = [
        TrainingText(text="a", source="software", domain_tag="software"),
        TrainingText(text="b", source="medical", domain_tag="medical"),
        TrainingText(text="c", source="medical", domain_tag="medical"),
    ]

    assert infer_domain_tag(texts) == "medical"


def test_build_lora_config_matches_phase_one_contract():
    config = PeftTrainingConfig(base_model="prajjwal1/bert-tiny")

    build_lora_config(config, FakePeftModule)

    assert FakePeftModule.captured_config == {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["query", "key", "value", "dense"],
        "lora_dropout": 0.05,
        "task_type": "FEATURE_EXTRACTION",
    }


def test_train_peft_model_orchestrates_tiny_model_export(tmp_path: Path):
    # Long enough to split into a query and a body; training pairs the opening of a post
    # against its own remainder, so a two-word fixture yields no pairs at all.
    body = " ".join(f"detail{index}" for index in range(60))
    texts = [
        TrainingText(
            text=f"scsi termination problem on the quadra {body}",
            source="medical",
            domain_tag="medical",
        ),
        TrainingText(
            text=f"isa card irq conflict after the upgrade {body}",
            source="medical",
            domain_tag="medical",
        ),
    ]

    def fake_exporter(model_dir: Path, output_dir: Path, task: str) -> None:
        assert task == "feature-extraction"
        assert (model_dir / "adapter_config.json").exists()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "model.onnx").write_bytes(b"onnx")

    validated: list[tuple[Path, int]] = []
    FakeTokenizer.seen_lengths.clear()

    def recording_validator(path: Path, expected_dim: int) -> int:
        validated.append((path, expected_dim))
        return expected_dim

    telemetry, adapter_dir, onnx_dir = train_peft_model(
        texts,
        PeftTrainingConfig(
            base_model="prajjwal1/bert-tiny",
            epochs=1,
            batch_size=2,
            max_length=32,
            query_max_length=8,
            output_dir=str(tmp_path),
        ),
        deps=fake_deps(),
        exporter=fake_exporter,
        validator=recording_validator,
    )

    # The query side is tokenised to query_max_length and the document side to
    # max_length, rather than both to one shared number.
    assert FakeTokenizer.seen_lengths[0] == 8
    assert set(FakeTokenizer.seen_lengths[1:]) == {32}

    assert telemetry.sample_count == 2
    assert telemetry.loss_history == [{"step": 1.0, "loss": 0.4}]
    assert telemetry.domain_tag == "medical"
    assert (adapter_dir / "adapter_config.json").exists()
    assert (onnx_dir / "model.onnx").exists()

    # A successful optimum-cli exit says nothing about whether the graph loads or kept its
    # hidden width, so the export has to be checked against the merged model's config.
    assert validated == [(onnx_dir / "model.onnx", 128)]


def test_upload_peft_artifacts_writes_adapter_onnx_and_metadata(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    onnx_dir = tmp_path / "onnx"
    adapter_dir.mkdir()
    onnx_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"r":8}', encoding="utf-8")
    (onnx_dir / "model.onnx").write_bytes(b"onnx")
    client = FakeMinio()

    artifacts = upload_peft_artifacts(
        version="2026.07.28-test",
        telemetry=PeftTrainingTelemetry(
            sample_count=2,
            loss_history=[{"step": 1.0, "loss": 0.4}],
            domain_tag="medical",
        ),
        adapter_dir=adapter_dir,
        onnx_dir=onnx_dir,
        client=client,
    )

    assert artifacts.domain_tag == "medical"
    assert artifacts.onnx_uri.endswith("/2026.07.28-test/model.onnx")
    assert any(key.endswith("peft-adapter-config.json") for key in client.objects)
    assert any(key.endswith("peft-training-metadata.json") for key in client.objects)


@pytest.mark.asyncio
async def test_load_drifted_training_texts_queries_window_when_available():
    db = SimpleNamespace()
    # Two rows minimum: a window returning fewer now falls back to recent documents, which
    # is a different query and would defeat what this test is checking.
    db.query_raw = AsyncMock(
        return_value=[
            {"text": "isa card irq conflict", "source": "pc_hardware"},
            {"text": "mac quadra vram question", "source": "mac_hardware"},
        ]
    )

    texts = await load_drifted_training_texts(db, "11111111-1111-1111-1111-111111111111", limit=50)

    assert texts[0] == TrainingText(
        text="isa card irq conflict", source="pc_hardware", domain_tag="pc_hardware"
    )
    assert "drift_windows" in db.query_raw.call_args.args[0]


@pytest.mark.asyncio
async def test_mark_model_pending_eval_updates_phase_one_columns():
    db = SimpleNamespace()
    db.execute_raw = AsyncMock(return_value=1)

    await mark_model_pending_eval(
        db,
        model_id="11111111-1111-1111-1111-111111111111",
        artifacts=SimpleNamespace(
            domain_tag="medical",
            onnx_uri="s3://models/model.onnx",
            onnx_sha256="a" * 64,
            onnx_bytes=123,
        ),
        eval_mrr=0.91,
    )

    query = db.execute_raw.call_args.args[0]
    assert "PENDING_EVAL" in query
    assert "domain_tag" in query
    assert "onnx_path" in query
    assert "eval_mrr" in query


@pytest.mark.asyncio
async def test_training_set_follows_embedding_time_not_ingestion_time():
    """Window membership must match how the drift service builds a centroid.

    Two clocks have been wrong here. Selecting documents by ingested_at described a
    different set entirely, because embedding runs well behind ingestion once a real model
    does the work, and training died on an empty corpus. Selecting by created_at then broke
    a second way: a backfill moves it, so re-encoded documents would drift into whichever
    window was current. first_embedded_at is the arrival clock the drift service uses.
    """
    from continuum_trainer.peft_engine import load_drifted_training_texts

    captured: list[str] = []

    async def query_raw(sql: str, *args):
        captured.append(sql)
        return [{"text": "mac quadra vram question", "source": "mac_hardware"}] * 4

    db = SimpleNamespace(query_raw=query_raw)
    texts = await load_drifted_training_texts(db, "window-1", limit=10)

    assert len(texts) == 4
    statement = " ".join(captured[0].split())
    assert "FROM embeddings e" in statement
    assert "e.first_embedded_at >= w.window_start" in statement
    assert "e.created_at" not in statement
    assert "ingested_at" not in statement


@pytest.mark.asyncio
async def test_sparse_window_falls_back_to_recent_documents():
    """An unlucky window boundary should degrade the selection, not kill the run."""
    from continuum_trainer.peft_engine import load_drifted_training_texts

    calls: list[str] = []

    async def query_raw(sql: str, *args):
        calls.append(sql)
        if len(calls) == 1:
            return []  # window matched nothing
        return [{"text": f"recent document {index}", "source": "pc_hardware"} for index in range(6)]

    db = SimpleNamespace(query_raw=query_raw)
    texts = await load_drifted_training_texts(db, "window-1", limit=10)

    assert len(texts) == 6
    assert len(calls) == 2
    assert "ORDER BY ingested_at DESC" in " ".join(calls[1].split())


def encoder_from(table: dict[str, list[float]]):
    """Deterministic stand-in for the base encoder, so mining is exercised without ONNX."""

    def encode(texts: list[str]) -> np.ndarray:
        return np.array([table[text] for text in texts], dtype=np.float32)

    return encode


def test_mine_hard_negatives_picks_the_nearest_wrong_document():
    queries = ["q0", "q1", "q2"]
    documents = ["d0", "d1", "d2"]
    encode = encoder_from(
        {
            "q0": [1.0, 0.0],
            "q1": [0.0, 1.0],
            "q2": [1.0, 1.0],
            # d0 is q0's positive. d2 sits close to q0 without matching it, d1 is orthogonal.
            "d0": [1.0, 0.0],
            "d1": [0.0, 1.0],
            "d2": [0.9, 0.4],
        }
    )

    chosen = mine_hard_negatives(queries, documents, encode)

    assert chosen[0] == 2


def test_mine_hard_negatives_skips_probable_false_negatives():
    """Newsgroup posts quote each other, so the nearest neighbour is sometimes a match.

    Training against a document that genuinely answers the query teaches the model to
    separate things that belong together, which is worse than an easy negative.
    """
    queries = ["q0", "q1", "q2"]
    documents = ["d0", "d1", "d2"]
    encode = encoder_from(
        {
            "q0": [1.0, 0.0],
            "q1": [0.0, 1.0],
            "q2": [1.0, 1.0],
            # d0 is q0's positive at 0.8. d1 matches q0 perfectly, which makes it a missed
            # positive rather than a negative. d2 is the hardest genuine negative left.
            "d0": [0.8, 0.6],
            "d1": [1.0, 0.0],
            "d2": [0.5, 0.866],
        }
    )

    chosen = mine_hard_negatives(queries, documents, encode)

    # Without the filter this would be d1, the document scoring above the true positive.
    assert chosen[0] == 2


def test_mine_hard_negatives_falls_back_when_everything_is_filtered():
    """A negative is still required for the batch, so the filter cannot return nothing."""
    queries = ["q0", "q1"]
    documents = ["d0", "d1"]
    encode = encoder_from(
        {
            "q0": [1.0, 0.0],
            "q1": [0.0, 1.0],
            "d0": [0.7, 0.7],
            "d1": [1.0, 0.0],
        }
    )

    chosen = mine_hard_negatives(queries, documents, encode)

    assert chosen[0] == 1
    assert len(chosen) == 2


def test_hard_negatives_make_the_objective_harder():
    """A mined negative belongs in the denominator, so the loss must rise when one is added."""
    rng = np.random.default_rng(3)
    queries = unit_rows(rng.normal(size=(8, 16)))
    documents = unit_rows(queries + 0.05 * rng.normal(size=(8, 16)))
    # Distractors sitting almost on top of the queries: the hardest possible negatives.
    negatives = unit_rows(queries + 0.01 * rng.normal(size=(8, 16)))

    without = in_batch_contrastive_loss(queries, documents, 0.05, NumpyTorchStub)
    with_negatives = in_batch_contrastive_loss(queries, documents, 0.05, NumpyTorchStub, negatives)

    assert with_negatives > without


def test_dataset_carries_a_mined_negative_per_pair(monkeypatch):
    monkeypatch.setattr(
        "continuum_trainer.peft_engine.mine_hard_negatives",
        lambda queries, documents: list(reversed(range(len(documents)))),
    )
    texts = [
        TrainingText(
            text=" ".join(f"word{index}" for index in range(80)), source="s", domain_tag="d"
        )
        for _ in range(3)
    ]

    dataset = build_contrastive_dataset(texts, FakeDatasetsModule, use_hard_negatives=True)

    assert "negative" in dataset.payload
    assert len(dataset.payload["negative"]) == len(dataset.payload["query"])


def test_dataset_omits_negatives_when_disabled():
    texts = [
        TrainingText(
            text=" ".join(f"word{index}" for index in range(80)), source="s", domain_tag="d"
        )
        for _ in range(3)
    ]

    dataset = build_contrastive_dataset(texts, FakeDatasetsModule, use_hard_negatives=False)

    assert "negative" not in dataset.payload
