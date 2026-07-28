"""Real document corpora for seeding, drawn from the 20 Newsgroups dataset.

The seed corpus used to be twenty hand-written sentences sampled with replacement, so
"1500 documents" was twenty strings repeated seventy-five times each. That gives a drift
detector almost nothing to measure: every window is a permutation of the same twenty
vectors, and any separation it reports is an artifact of how those sentences were written.

These are real newsgroup posts written by people. `comp.*` stands in for the software
baseline and `sci.med` for the drifted clinical domain, so the domain shift the demo
narrates is a genuine shift in subject matter rather than a change of keyword list.

The dataset is fetched as plain JSONL through huggingface_hub, which the embedding model
already depends on. scikit-learn ships `fetch_20newsgroups` and is already a dependency,
but it downloads from figshare, which failed outright during development — not something
worth putting on CI's critical path.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

DATASET_REPO = "SetFit/20_newsgroups"
# Both splits are used as one pool. This is a document corpus for seeding a pipeline, not
# a supervised benchmark, so there is no train/test leak to protect against — and the
# train split alone yields only 403 usable sci.med posts against the 500 the demo needs.
DATASET_FILES = ("train.jsonl", "test.jsonl")

SOFTWARE_CATEGORIES = frozenset(
    {
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
    }
)
MEDICAL_CATEGORIES = frozenset({"sci.med"})

# Posts shorter than this are mostly quoted signatures and say little about the domain.
# The upper bound keeps a single rambling thread from dominating a window's centroid, and
# text beyond the model's 256-token window is truncated anyway.
MIN_WORDS = 25
MAX_WORDS = 200


@lru_cache(maxsize=1)
def _load_rows() -> tuple[dict[str, str], ...]:
    from huggingface_hub import hf_hub_download

    rows: list[dict[str, str]] = []
    for filename in DATASET_FILES:
        path = Path(hf_hub_download(DATASET_REPO, filename, repo_type="dataset"))
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return tuple(rows)


def load_documents(categories: frozenset[str], limit: int, seed: int) -> list[str]:
    """Return up to `limit` distinct real posts from the given newsgroup categories."""
    candidates = [
        text
        for row in _load_rows()
        if row.get("label_text") in categories
        for text in [str(row.get("text", "")).strip()]
        if MIN_WORDS <= len(text.split()) <= MAX_WORDS
    ]
    if not candidates:
        raise RuntimeError(f"no documents matched categories {sorted(categories)}")

    # Newsgroup posts are cross-posted between groups and repeated across the two splits,
    # so the raw pool holds exact duplicates. Left in, they would weight a window's
    # centroid toward whichever document happened to be posted twice. dict.fromkeys keeps
    # first-seen order so the shuffle below stays reproducible for a given seed.
    unique = list(dict.fromkeys(candidates))

    random.Random(seed).shuffle(unique)
    return unique[:limit]


def software_documents(limit: int, seed: int) -> list[str]:
    return load_documents(SOFTWARE_CATEGORIES, limit, seed)


def medical_documents(limit: int, seed: int) -> list[str]:
    return load_documents(MEDICAL_CATEGORIES, limit, seed)
