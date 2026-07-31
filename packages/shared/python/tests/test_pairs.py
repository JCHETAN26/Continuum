"""Pair construction is shared by training, the gate and the benchmark.

If they disagreed, the adapter would be trained on one task and measured on another.
"""

from __future__ import annotations

from continuum_shared.pairs import (
    MIN_DOCUMENT_WORDS,
    QUERY_WORDS,
    build_pairs,
    split_query_and_document,
)


def test_query_is_removed_from_its_own_document():
    """Overlap would let a model score by matching the literal string."""
    text = " ".join(f"w{i}" for i in range(80))
    query, document = split_query_and_document(text)

    assert query.split() == [f"w{i}" for i in range(QUERY_WORDS)]
    assert query not in document
    assert len(document.split()) == 80 - QUERY_WORDS


def test_text_too_short_to_split_is_rejected():
    """A body barely longer than the query leaves nothing to retrieve."""
    assert split_query_and_document("one two three") is None

    just_short = " ".join(["w"] * (QUERY_WORDS + MIN_DOCUMENT_WORDS - 1))
    assert split_query_and_document(just_short) is None

    just_long_enough = " ".join(["w"] * (QUERY_WORDS + MIN_DOCUMENT_WORDS))
    assert split_query_and_document(just_long_enough) is not None


def test_build_pairs_drops_texts_it_cannot_split():
    long_text = " ".join(f"w{i}" for i in range(100))

    pairs = build_pairs([long_text, "far too short", long_text])

    assert len(pairs) == 2
    assert all(query and document for query, document in pairs)
