# ADR-006: Supervised (Query, Document) Pairs Over Dropout Views

## Status

Accepted. Supersedes the augmentation half of
[ADR-003](003-contrastive-in-batch-negative-loss.md).

## Context

ADR-003 chose dropout views, following SimCSE, because drift-time text carries no relevance
labels. That reasoning was sound and the objective trained without collapsing, but the
system could not tell whether it was working.

Two problems showed up once the pipeline ran end to end on real data.

The gate counted any document from the same newsgroup as relevant. Half the candidates
qualified, so both models scored MRR near 0.88 and four training runs moved it by between
−0.21% and +1.25%. Whatever the adapter learned, the measurement could not see it.

The objective and the gate also disagreed. Training pulled dropout views of one document
together; the gate asked whether a document ranked near other documents from its domain.
Nothing connected the two, so improving the training loss carried no expectation of
improving the score.

## Decision

Both sides now use the same construction, in `continuum_shared.pairs`: split each post into
its opening fifteen words as the query and the remainder as the document, requiring at
least forty words in the remainder.

Training encodes the query side and the document side separately and applies the same
symmetric InfoNCE from ADR-003, with the other documents in the batch as negatives. The
gate asks each held-out query to retrieve its own document out of every candidate.

This is supervised only in the weak sense that the pair is constructed rather than
observed. No human labels are involved, so the objection in ADR-003 — that labels are
unavailable at drift time — is still respected.

## Consequences

The task is much harder, and that is the point. Base-model MRR fell from about 0.88 to
between 0.41 and 0.48, which leaves room for an adapter to register.

Measured gains became legible. Across five runs the three that trained on 124 to 250
examples moved MRR by between −0.49% and +0.87%, while the two that trained on 400 and 450
gained +9.86% and +8.29%. Part of the earlier reading — an adapter that helps unpredictably
— was an artifact of a gate that could not resolve the difference, and part of it is that
the drift window captures different amounts of data on each run.

The query is removed from its own document, so a hit means the model matched meaning rather
than finding a copy of the query string.

The construction assumes a post's opening is representative of the rest of it. That holds
for newsgroup posts and would not hold for documents with boilerplate headers, so a
different corpus may need a different split.

All four runs were still rejected by the 10% activation gate. The gate is doing its job,
but the adapter has not yet earned a promotion, and hard-negative batching is the obvious
next thing to try.
