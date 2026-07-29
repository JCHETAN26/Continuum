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
BASELINE_DOCUMENTS = 700
DRIFT_DOCUMENTS = 500


@pytest.fixture(scope="module")
def baseline():
    return corpus.load_baseline_documents(BASELINE_DOCUMENTS, seed=7)


@pytest.fixture(scope="module")
def drifted():
    return corpus.load_drift_documents(DRIFT_DOCUMENTS, seed=7)


def test_supplies_enough_documents_for_the_demo(baseline, drifted):
    assert len(baseline) == BASELINE_DOCUMENTS
    assert len(drifted) == DRIFT_DOCUMENTS


def test_documents_are_distinct(baseline, drifted):
    """No repetition: 1500 documents must mean 1500 different documents."""
    assert len(set(baseline)) == len(baseline)
    assert len(set(drifted)) == len(drifted)
    assert not set(baseline) & set(drifted)


def test_documents_are_substantial_prose(baseline, drifted):
    for document in (*baseline, *drifted):
        assert corpus.MIN_WORDS <= len(document.split()) <= corpus.MAX_WORDS


def test_selection_is_deterministic_for_a_given_seed():
    """Drift compares windows over time, so a rerun must produce the same corpus."""
    assert corpus.load_drift_documents(50, seed=7) == corpus.load_drift_documents(50, seed=7)


def test_different_seeds_select_different_documents():
    assert corpus.load_drift_documents(50, seed=7) != corpus.load_drift_documents(50, seed=8)


def test_domains_are_lexically_distinguishable(baseline, drifted):
    """A weak proxy for the shift, without paying for embeddings in a unit test.

    The two groups deliberately overlap — both are hardware support threads — so this only
    checks that platform vocabulary leans the right way, not that the domains are cleanly
    separable. Separability is measured with real embeddings, not here.
    """
    mac_words = {"mac", "apple", "macintosh", "quadra", "powerbook", "centris"}
    pc_words = {"dos", "isa", "ide", "motherboard", "bios", "jumper"}

    def share(documents: list[str], vocabulary: set[str]) -> float:
        hits = sum(1 for text in documents if vocabulary & set(text.lower().split()))
        return hits / len(documents)

    assert share(drifted, mac_words) > share(baseline, mac_words)
    assert share(baseline, pc_words) > share(drifted, pc_words)


def test_unknown_category_is_an_error_rather_than_an_empty_corpus():
    with pytest.raises(RuntimeError, match="no documents matched"):
        corpus.load_documents(frozenset({"rec.knitting"}), limit=10, seed=7)
