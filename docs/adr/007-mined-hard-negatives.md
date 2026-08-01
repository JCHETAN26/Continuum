# ADR-007: Mined Hard Negatives

## Status

Accepted, on grounds of construction rather than measured gain. See Consequences.

## Context

[ADR-006](006-supervised-pairs-over-dropout-views.md) made the training objective and the
gate agree on the same (query, document) task. What it did not fix is which documents the
query is scored against.

In-batch negatives are whatever the sampler happened to draw. The corpus is two hardware
newsgroups, so a randomly drawn negative is usually easy: the model tells a PC thread from
a Mac thread without learning anything that transfers to ranking one specific post above
another. The gate asks the harder question, so the objective was leaving the gate's actual
difficulty untouched.

## Decision

Each pair carries an additional negative, mined with the base model: the document it
currently ranks nearest the correct answer for that query. Those are the confusions the
adapter exists to correct.

The loss extends the candidate set from `B` to `2B` columns with targets still on the
diagonal. The backward direction continues to score documents against queries only — a
mined negative has no query of its own and therefore cannot be a target.

Candidates scoring at or above the true positive are skipped. Newsgroup posts quote each
other, so the nearest neighbour is sometimes a genuine match, and training against one
teaches the model to separate documents that belong together. A fallback keeps every batch
supplied with a negative when the filter removes all candidates.

`use_hard_negatives` defaults on and can be turned off, so both arms remain runnable on the
same corpus.

## Consequences

Cost is a third encode per training step.

The measured effect is unresolved. Four runs gained +1.17%, +4.22%, +7.02% and +10.44% at
119, 270, 400 and 400 training examples.

The two runs at 400 settle the question of whether these numbers can be compared at all:
identical code and identical training-set size produced +7.02% and +10.44%, a 3.4-point
spread straddling the activation gate. The within-condition noise is wider than any
difference between the two objectives, so the comparison against in-batch-only runs cannot
support a conclusion in either direction.

**No improvement is claimed.** Deciding this needs an experiment that pins the training-set
size and runs both arms several times, rather than the opportunistic sizes CI produces
while doing other work.

This ADR is accepted because mined hard negatives are the standard construction for this
objective and the implementation is correct, not because they have been shown to win here.
Reverting is a one-line configuration change if a controlled experiment later says they
should be.

The false-negative filter is a heuristic. It cannot detect a near-duplicate that scores
below the true positive, and on a corpus with heavier quoting it would need replacing with
something stronger than a similarity ceiling.
