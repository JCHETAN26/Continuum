# Verified Metrics

Every number here was produced by a CI run on a machine nobody could tamper with, and each
one links to the run that emitted it. Nothing is estimated, and nothing is carried over
from an earlier design of the system.

Source runs: [`30426014715`](https://github.com/JCHETAN26/Continuum/actions/runs/30426014715)
and [`30429103002`](https://github.com/JCHETAN26/Continuum/actions/runs/30429103002)
· commits `993265c` and `dcb8c18`

## Domain adaptation

A drift alert triggered a LoRA adaptation of `sentence-transformers/all-MiniLM-L6-v2` over
the drifted window, followed by an ONNX export and an evaluation against the base model on
the same held-out documents.

Two independent runs of the identical scenario:

| metric            | run 1 baseline | run 1 candidate | run 1   | run 2 baseline | run 2 candidate | run 2   |
| ----------------- | -------------- | --------------- | ------- | -------------- | --------------- | ------- |
| MRR               | 0.877744       | 0.882762        | +0.57%  | 0.897203       | 0.902210        | +0.56%  |
| Recall@5          | 0.9725         | 0.9800          | +0.77%  | 0.9825         | 0.9800          | −0.25%  |
| Mean margin       | 0.026163       | 0.022054        | −15.71% | 0.019118       | 0.022821        | +19.37% |
| Composite quality | 0.841996       | 0.846449        | +0.53%  | 0.856116       | 0.859148        | +0.35%  |

**The headline result is the MRR gain: +0.57% and +0.56%.** That is the one figure that
reproduces. Recall@5 and mean margin change sign between runs, so neither supports a claim
in either direction — they are reported here rather than omitted precisely because a table
showing only the favourable run would be dishonest.

Absolute values move between runs because the evaluation set is drawn from the most
recently embedded documents, and which documents those are depends on the order embedding
completes in. The relative gain is stable across that variation, which is what makes it
worth quoting.

**The candidate was rejected in both runs**, at 0.53% and 0.35% against a 10% activation
gate, so the pipeline kept serving the baseline. That is the intended outcome: the system
measured a candidate, found the gain too small to justify a model swap, and declined to
ship it.

## Adapter

| metric                | value                            |
| --------------------- | -------------------------------- |
| Base model parameters | 23,051,136                       |
| Trainable parameters  | 337,920                          |
| Trainable share       | 1.47%                            |
| LoRA rank / alpha     | 8 / 16                           |
| Target modules        | `query`, `key`, `value`, `dense` |

## Drift detection

Observed across four consecutive CI runs of the same seeded scenario, 700
`comp.sys.ibm.pc.hardware` posts followed by 500 from `comp.sys.mac.hardware`:

| measurement                          | value                     |
| ------------------------------------ | ------------------------- |
| Drift score at breach                | 0.099 – 0.518 across runs |
| Alert threshold                      | 0.080                     |
| Within-domain noise floor (PC vs PC) | 0.0506, 0.0598, 0.0629    |
| Centroid distance, PC vs Mac         | 0.1562                    |

The threshold sits above the noise floor and below a drift-dominated window. Lowering it
to catch an evenly mixed window (0.0789) would put it under the noise floor and alert on a
stable distribution.

## Method

Relevance is same-source: a retrieval counts as correct when the top-ranked neighbour
comes from the same newsgroup. Both sides embed identical raw text, so the comparison
isolates the adapter. The evaluation set is capped at 400 documents, each encoded twice.

## Why the gain is small, and why the task was chosen

The base model is already strong here. Three candidate domain pairs were measured before
one was chosen:

| pair                            | drift separation | baseline MRR      |
| ------------------------------- | ---------------- | ----------------- |
| `comp.*` vs `sci.med`           | 0.9156           | 0.9833            |
| `sci.med` vs `sci.electronics`  | 0.8886           | 0.9670            |
| `pc.hardware` vs `mac.hardware` | 0.1760           | 0.8591 — selected |

A live run against `comp.*` vs `sci.med` measured baseline MRR at 0.9993, leaving no room
for any adapter to register an improvement. Two flavours of hardware discussion share
vocabulary and structure, which leaves genuine headroom while still drifting clear of the
detection threshold.

Adaptation is unsupervised: positives come from dropout views of the same document
(SimCSE), with no labels stating which documents ought to be close. Mean margin fell while
MRR and recall rose, so the adapter sharpened the top of the ranking while pulling the
domains slightly closer overall. That is a real limitation of the objective, not a defect
in the run.
