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
runs under, using documents from the ingested corpus. Nearest-rank percentiles over 50
requests per batch size, warm-up excluded.

Five runs, between
[`30495237125`](https://github.com/JCHETAN26/Continuum/actions/runs/30495237125) and
[`30524795102`](https://github.com/JCHETAN26/Continuum/actions/runs/30524795102).

| batch | p50 range       | p95 range       | slowest observed |
| ----- | --------------- | --------------- | ---------------- |
| 1     | 101 – 299 ms    | 200 – 604 ms    | 697 ms           |
| 8     | 1410 – 3307 ms  | 2377 – 4098 ms  | 4203 ms          |
| 32    | 7504 – 14208 ms | 8195 – 15300 ms | 15799 ms         |

**Run-to-run spread reaches 3.0x at batch 1 and 1.9x at batch 32.** These are shared CI
runners with no isolation, so the figures support an order-of-magnitude claim and nothing
finer. Any single run quoted alone would misrepresent them, and the ranges above were
widened twice as further runs landed outside earlier versions of this table.

### p99 is not reported, because 50 samples cannot support one

At 50 samples the nearest-rank p99 is rank `ceil(0.99 * 50) = 50`, which is the slowest
request observed. Reporting that as a 99th percentile would dress the maximum up as
something it is not, so the slowest request is labelled as such and p95 (rank 48) is the
tail figure. The benchmark prints the same caveat when run under a hundred samples.

### The spec target is missed by two orders of magnitude

The target is p99 under 50 ms at batch 32. Median latency across the five runs ranged from
7.5 s to 14.2 s, so the gap is 150x to 284x. That conclusion is the one thing the
measurements agree on, and it holds under the most generous reading of the spread.

Two causes, both configuration rather than model:

- The serving container inherits `x-api-limits` and runs on **0.50 CPU**. That is
  transformer inference on half a core, and it dominates everything else.
- Every request pads to `MAX_SEQUENCE_LENGTH` of 256 tokens while corpus documents run
  about 110, so most of each forward pass is padding that the attention mask then discards.

Per-document cost rises with batch size in every run, for example 202 to 340 ms, 299 to
437 ms, and 101 to 235 ms. Batching normally lowers per-item cost; that it rises is the signature of CPU
starvation, where the work cannot spread across cores because there are none to spread
across.

Raising the serving CPU limit and switching to dynamic padding are the obvious next steps.
Neither has been done, so neither is claimed.
