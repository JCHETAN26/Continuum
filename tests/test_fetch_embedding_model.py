"""The build-time fetch script must stay in step with the runtime loader.

`scripts/fetch_embedding_model.py` cannot import `continuum_shared`: that package builds a
Settings instance at import time, which needs DATABASE_URL and the rest of the environment,
and a docker build has none of it. So the model coordinates are duplicated there, and this
pins the copies together — otherwise the image could be baked with one model while the
services load another, and every vector would silently change meaning.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

from continuum_shared import embeddings


def load_fetch_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_embedding_model.py"
    spec = importlib.util.spec_from_file_location("fetch_embedding_model", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["fetch_embedding_model"] = module
    spec.loader.exec_module(module)
    return module


def test_model_coordinates_match_the_runtime_loader():
    script = load_fetch_script()

    assert script.MODEL_REPO == embeddings.MODEL_REPO
    assert script.ONNX_FILENAME == embeddings.ONNX_FILENAME
    assert script.TOKENIZER_FILENAME == embeddings.TOKENIZER_FILENAME


def test_script_does_not_import_continuum_shared():
    """Importing it would break the docker build, which has no environment configured."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_embedding_model.py"
    tree = ast.parse(path.read_text())

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    offenders = {name for name in imported if name.split(".")[0] == "continuum_shared"}
    assert not offenders, f"docker build would fail on Settings validation: {offenders}"


def test_fetch_writes_the_filenames_the_loader_expects(tmp_path, monkeypatch):
    """The loader looks for model.onnx and tokenizer.json, not the Hub's nested paths."""
    script = load_fetch_script()

    def fake_download(repo: str, filename: str) -> str:
        assert repo == embeddings.MODEL_REPO
        source = tmp_path / "hub" / filename
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"weights")
        return str(source)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    target = script.fetch(tmp_path / "models")

    assert (target / "model.onnx").exists()
    assert (target / embeddings.TOKENIZER_FILENAME).exists()
