# Verified Metrics

Every number here was produced by a CI run on a machine nobody could tamper with, and each
one links to the run that emitted it. Nothing is estimated.

The gate and the objective were both replaced in
[#37](https://github.com/JCHETAN26/Continuum/pull/37). Figures from before that change are
not comparable to these and have been removed rather than carried forward — an earlier
version of this document reported baseline MRR near 0.88 from the previous, much easier
gate.

## Retrieval quality, measured independently

`eval/benchmark.py` scores 100 held-out queries against the serving API. Each query is the
opening fifteen words of a post, the document is the rest of that post, and it is the only
relevant result among all 100 candidates. The query is removed from its own document, so a
hit means the model matched meaning rather than finding a copy of the query string.

Run [`30589884101`](https://github.com/JCHETAN26/Continuum/actions/runs/30589884101):

| domain         | queries | candidates | MRR    | recall@1 | recall@5 |
| -------------- | ------- | ---------- | ------ | -------- | -------- |
| `pc_hardware`  | 50      | 100        | 0.6051 | 0.50     | 0.70     |
| `mac_hardware` | 50      | 100        | 0.4340 | 0.30     | 0.58     |
| overall        | 100     | 100        | 0.5196 |          |          |

**Retrieval on the drifted domain is 28% worse than on the baseline domain**, 0.434 against
0.605. That is the premise of the project appearing in a measurement: drift is not only a
centroid moving, it is a measurable loss of retrieval quality on the shifted distribution.
It is also the one number here produced by code that shares nothing with the pipeline it
measures.

## Domain adaptation

A drift alert triggers a LoRA adaptation of `sentence-transformers/all-MiniLM-L6-v2` over
the drifted window, followed by an ONNX export and an evaluation against the base model.
Each query must retrieve its own document out of every candidate, which is the same task
the adapter trains on, so the gate and the objective agree.

Four runs of the identical scenario:

| run                                                                              | training samples | baseline MRR | candidate MRR | change |
| -------------------------------------------------------------------------------- | ---------------- | ------------ | ------------- | ------ |
| [`30605700975`](https://github.com/JCHETAN26/Continuum/actions/runs/30605700975) | 124              | 0.4756       | 0.4801        | +0.62% |
| [`30658311041`](https://github.com/JCHETAN26/Continuum/actions/runs/30658311041) | 250              | 0.4429       | 0.4458        | +0.87% |
| [`30607125415`](https://github.com/JCHETAN26/Continuum/actions/runs/30607125415) | 400              | 0.4113       | 0.4474        | +9.86% |
| [`30654126895`](https://github.com/JCHETAN26/Continuum/actions/runs/30654126895) | 450              | 0.4146       | 0.4487        | +8.29% |

**The gain tracks how much drifted data the training window captured.** Ordered by sample
count, the improvement is monotonic up to the two large runs: below roughly 400 in-window
examples the adapter barely moves retrieval, above it the gain is eight to ten percent.

Three of these runs were originally read as an unreliable adapter producing scattered
results. They are not scattered against sample count, and the window size varies between
runs because it depends on which documents happened to land inside the drift window. Four
points cannot establish a threshold, and nothing here rules out other differences between
runs; the mechanism — more contrastive pairs, better adapter — is at least the ordinary
one.

**The candidate was rejected in all four runs**, against a 10% activation gate. Even the
best run fell 0.14 points short. That is the system working: it measured a candidate, found
no gain worth a model swap, and kept serving the baseline.

### NDCG@10 does not move with MRR

| run           | baseline NDCG@10 | candidate NDCG@10 |
| ------------- | ---------------- | ----------------- |
| `30607125415` | 0.6385           | 0.6466            |
| `30654126895` | 0.6554           | 0.6527            |

On the run that gained 8.29% MRR, graded NDCG@10 fell slightly. The adapter sharpens the
top of the ranking without improving the graded ordering below it. NDCG is reported but
deliberately excluded from the promotion decision, so this cannot change which models ship.

## Adapter

| metric                | value                            |
| --------------------- | -------------------------------- |
| Base model parameters | 23,051,136                       |
| Trainable parameters  | 337,920                          |
| Trainable share       | 1.47%                            |
| LoRA rank / alpha     | 8 / 16                           |
| Target modules        | `query`, `key`, `value`, `dense` |

## Drift detection

Observed across CI runs of the same seeded scenario, 700 `comp.sys.ibm.pc.hardware` posts
followed by 500 from `comp.sys.mac.hardware`:

| measurement                          | value                     |
| ------------------------------------ | ------------------------- |
| Drift score at breach                | 0.099 – 0.518 across runs |
| Alert threshold                      | 0.080                     |
| Within-domain noise floor (PC vs PC) | 0.0506, 0.0598, 0.0629    |
| Centroid distance, PC vs Mac         | 0.1562                    |

The threshold sits above the noise floor and below a drift-dominated window. Lowering it
to catch an evenly mixed window (0.0789) would put it under the noise floor and alert on a
stable distribution.

## Linguistic drift

Run [`30658311041`](https://github.com/JCHETAN26/Continuum/actions/runs/30658311041), real
spaCy named-entity extraction over ingested documents:

| measurement        | value |
| ------------------ | ----- |
| Documents analysed | 350   |
| Composite score    | 0.903 |
| Threshold          | 0.650 |
| New entities       | 10    |
| Emerging terms     | 10    |

## Serving latency

Measured against the compose stack in CI, at the same images and resource limits the demo
runs under, using documents from the ingested corpus. Nearest-rank percentiles over 50
requests per batch size, warm-up excluded.

Three runs: [`30605700975`](https://github.com/JCHETAN26/Continuum/actions/runs/30605700975),
[`30607125415`](https://github.com/JCHETAN26/Continuum/actions/runs/30607125415) and
[`30658311041`](https://github.com/JCHETAN26/Continuum/actions/runs/30658311041).

| batch | p50 range      | p95 range      | slowest observed |
| ----- | -------------- | -------------- | ---------------- |
| 1     | 27 – 71 ms     | 38 – 137 ms    | 186 ms           |
| 8     | 346 – 788 ms   | 503 – 1116 ms  | 1289 ms          |
| 32    | 1603 – 3356 ms | 2235 – 3790 ms | 3888 ms          |

**Run-to-run spread reaches 2.6x at batch 1 and 2.1x at batch 32.** These are shared CI
runners with no isolation, so the figures support an order-of-magnitude claim and nothing
finer.

### Raising the CPU limit and dropping fixed padding cut batch-32 latency about 3x

Earlier runs measured 7504 – 14208 ms at batch 32. Two configuration changes account for
the difference: the serving container ran on **0.50 CPU** and now runs on 2.00, and every
request padded to `MAX_SEQUENCE_LENGTH` of 256 tokens while corpus documents run about 110,
so most of each forward pass was padding the attention mask then discarded. Padding is now
dynamic to a multiple of 8.

### p99 is not reported, because 50 samples cannot support one

At 50 samples the nearest-rank p99 is rank `ceil(0.99 * 50) = 50`, which is the slowest
request observed. Reporting that as a 99th percentile would dress the maximum up as
something it is not, so the slowest request is labelled as such and p95 (rank 48) is the
tail figure. The benchmark prints the same caveat when run under a hundred samples.

### The spec target is still missed

The target is p99 under 50 ms at batch 32, and the best observed p50 is 1603 ms — a gap of
roughly 32x after the 3x improvement, against 150x to 284x before it. Batch 1 now sits at
27 – 71 ms p50, within the target range, so the remaining gap is throughput under batching
rather than per-request cost.

## Why the task was chosen

The base model is already strong on easy domain pairs. Three candidates were measured
before one was chosen:

| pair                            | drift separation | baseline MRR      |
| ------------------------------- | ---------------- | ----------------- |
| `comp.*` vs `sci.med`           | 0.9156           | 0.9833            |
| `sci.med` vs `sci.electronics`  | 0.8886           | 0.9670            |
| `pc.hardware` vs `mac.hardware` | 0.1760           | 0.8591 — selected |

A live run against `comp.*` vs `sci.med` measured baseline MRR at 0.9993, leaving no room
for any adapter to register an improvement. Two flavours of hardware discussion share
vocabulary and structure, which leaves genuine headroom while still drifting clear of the
detection threshold.

## Method

Adaptation is supervised in the weak sense: positives are (query, document) pairs built by
splitting each post, with no human labels. The gate scores the same construction on
held-out documents, capped at 300 pairs, each encoded by both models. Both sides embed
identical raw text, so the comparison isolates the adapter.
