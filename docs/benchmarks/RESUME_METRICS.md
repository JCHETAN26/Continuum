# Verified Metrics

Every number here was produced by a CI run on a machine nobody could tamper with, and each
one links to the run that emitted it. Nothing is estimated, and nothing is carried over
from an earlier design of the system.

Source runs: [`30436026193`](https://github.com/JCHETAN26/Continuum/actions/runs/30436026193)
and [`30437089093`](https://github.com/JCHETAN26/Continuum/actions/runs/30437089093)
· commit `9b111df`

## Domain adaptation

A drift alert triggers a LoRA adaptation of `sentence-transformers/all-MiniLM-L6-v2` over
the drifted window, followed by an ONNX export and an evaluation against the base model on
the same held-out documents.

| run | baseline MRR | candidate MRR | gain   |
| --- | ------------ | ------------- | ------ |
| 1   | 0.877859     | 0.886278      | +0.96% |
| 2   | 0.897169     | 0.904057      | +0.77% |

**Adaptation improves retrieval MRR by 0.77% to 0.96%**, mean +0.86%, across two runs of
the identical scenario.

Full metrics from both runs:

| metric            | run 1 baseline | run 1 candidate | run 1  | run 2 baseline | run 2 candidate | run 2   |
| ----------------- | -------------- | --------------- | ------ | -------------- | --------------- | ------- |
| MRR               | 0.877859       | 0.886278        | +0.96% | 0.897169       | 0.904057        | +0.77%  |
| Recall@5          | 0.9825         | 0.9850          | +0.25% | 0.9925         | 0.9875          | −0.50%  |
| Mean margin       | 0.019936       | 0.021402        | +7.35% | 0.018803       | 0.023058        | +22.63% |
| Composite quality | 0.843603       | 0.849686        | +0.72% | 0.858070       | 0.861866        | +0.44%  |

MRR is the only metric that moves consistently. Recall@5 rises in one run and falls in the
other, and mean margin swings by very different amounts, so neither supports a claim in
either direction. They are shown rather than omitted because a table carrying only the
favourable column would misrepresent what was measured.

**The candidate was rejected in both runs**, at 0.72% and 0.44% composite improvement
against a 10% activation gate, so the pipeline kept serving the baseline. That is the
intended outcome: the system measured a candidate, found the gain too small to justify
swapping the model an index was built with, and declined to ship it.

### Two earlier measurements are excluded

Runs recorded before commit `9b111df` showed +0.57% and +0.56%. They are not comparable
and are not averaged in. The evaluation set was drawn from the most recently embedded
documents overall, and because drifted documents are ingested last they are embedded last,
so that set could be skewed toward a single domain. Retrieval here is scored against
same-source neighbours, so a single-domain set has no negatives at all: one run under that
selection returned baseline and candidate MRR of exactly 0.0000. The set is now drawn per
source, and only runs under that selection are reported.

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

## Serving latency

Measured against the compose stack in CI, at the same images and resource limits the demo
runs under, using documents from the ingested corpus. Percentiles are nearest-rank over 50
requests per batch size, warm-up excluded.

Source run: [`30495237125`](https://github.com/JCHETAN26/Continuum/actions/runs/30495237125)
· commit `35b57de`

| batch | p50        | p95        | p99        | per document (p50) |
| ----- | ---------- | ---------- | ---------- | ------------------ |
| 1     | 201.9 ms   | 497.8 ms   | 503.3 ms   | 201.9 ms           |
| 8     | 2403.4 ms  | 3194.7 ms  | 3296.4 ms  | 300.4 ms           |
| 32    | 10895.9 ms | 11821.6 ms | 12097.2 ms | 340.5 ms           |

**The spec target of p99 under 50 ms at batch 32 is missed by a factor of 242.** It is
recorded here as measured rather than adjusted to fit, because the target was written
before anything had been measured.

Two causes, both configuration rather than model:

- The serving container runs under `x-api-limits`, which caps it at **0.50 CPU**. That is
  transformer inference on half a core, and it dominates everything else.
- Every request pads to `MAX_SEQUENCE_LENGTH` of 256 tokens while corpus documents run
  about 110, so more than half of each forward pass is padding that the attention mask then
  discards.

Per-document cost rises from 201.9 ms at batch 1 to 340.5 ms at batch 32. Batching normally
lowers per-item cost; that it rises here is the signature of CPU starvation, where the work
cannot spread across cores because there are none to spread across.

Raising the serving CPU limit and switching to dynamic padding are the obvious next steps.
Neither has been done, so neither is claimed.
