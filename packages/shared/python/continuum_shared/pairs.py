"""Deriving (query, document) pairs from plain text.

Used for both training and evaluation, and shared so the two cannot diverge. A model
trained on pairs built one way and scored on pairs built another would be measured against
a task it was never taught.

The opening of a post becomes the query and the remainder becomes the document. That gives
a positive pair with real semantic content, derived from structure rather than annotation:
the opening of a support post states the problem, and the body elaborates on it.
"""

from __future__ import annotations

QUERY_WORDS = 15
MIN_DOCUMENT_WORDS = 40


def split_query_and_document(
    text: str,
    query_words: int = QUERY_WORDS,
    min_document_words: int = MIN_DOCUMENT_WORDS,
) -> tuple[str, str] | None:
    """Split text into a query and the document it should retrieve.

    The query is removed from the document rather than left in it. Overlapping text would
    let a model score well by matching the literal string, which measures nothing useful
    about retrieval and would train the adapter toward copying rather than understanding.
    """
    words = text.split()
    if len(words) < query_words + min_document_words:
        return None
    return " ".join(words[:query_words]), " ".join(words[query_words:])


def build_pairs(
    texts: list[str],
    query_words: int = QUERY_WORDS,
    min_document_words: int = MIN_DOCUMENT_WORDS,
) -> list[tuple[str, str]]:
    """Pairs for every text long enough to split; shorter ones are dropped."""
    pairs = []
    for text in texts:
        split = split_query_and_document(text, query_words, min_document_words)
        if split is not None:
            pairs.append(split)
    return pairs
