"""Contract for the seed corpora.

The point of this module is that the documents are real and distinct. Tests that only
checked counts would have passed just as happily against twenty sentences repeated
seventy-five times, which is what it replaces.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

CORPUS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "corpus.py"
spec = importlib.util.spec_from_file_location("continuum_corpus", CORPUS_PATH)
assert spec and spec.loader
corpus = importlib.util.module_from_spec(spec)
sys.modules["continuum_corpus"] = corpus
spec.loader.exec_module(corpus)

# What the demo actually asks for; see DEFAULT_* in scripts/seed.py.
BASELINE_DOCUMENTS = 1_000
DRIFT_DOCUMENTS = 500


@pytest.fixture(scope="module")
def software():
    return corpus.software_documents(BASELINE_DOCUMENTS, seed=7)


@pytest.fixture(scope="module")
def medical():
    return corpus.medical_documents(DRIFT_DOCUMENTS, seed=7)


def test_supplies_enough_documents_for_the_demo(software, medical):
    assert len(software) == BASELINE_DOCUMENTS
    assert len(medical) == DRIFT_DOCUMENTS


def test_documents_are_distinct(software, medical):
    """No repetition: 1500 documents must mean 1500 different documents."""
    assert len(set(software)) == len(software)
    assert len(set(medical)) == len(medical)
    assert not set(software) & set(medical)


def test_documents_are_substantial_prose(software, medical):
    for document in (*software, *medical):
        assert corpus.MIN_WORDS <= len(document.split()) <= corpus.MAX_WORDS


def test_selection_is_deterministic_for_a_given_seed():
    """Drift compares windows over time, so a rerun must produce the same corpus."""
    assert corpus.medical_documents(50, seed=7) == corpus.medical_documents(50, seed=7)


def test_different_seeds_select_different_documents():
    assert corpus.medical_documents(50, seed=7) != corpus.medical_documents(50, seed=8)


def test_domains_are_lexically_distinguishable(software, medical):
    """A weak proxy for the domain shift, without paying for embeddings in a unit test."""
    clinical = {"patient", "doctor", "medical", "disease", "treatment", "symptoms"}
    technical = {"windows", "file", "software", "graphics", "hardware", "server"}

    def share(documents: list[str], vocabulary: set[str]) -> float:
        hits = sum(1 for text in documents if vocabulary & set(text.lower().split()))
        return hits / len(documents)

    assert share(medical, clinical) > share(software, clinical)
    assert share(software, technical) > share(medical, technical)


def test_unknown_category_is_an_error_rather_than_an_empty_corpus():
    with pytest.raises(RuntimeError, match="no documents matched"):
        corpus.load_documents(frozenset({"rec.knitting"}), limit=10, seed=7)
