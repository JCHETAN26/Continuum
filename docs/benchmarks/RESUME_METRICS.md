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

Run [`30877829855`](https://github.com/JCHETAN26/Continuum/actions/runs/30877829855):

| domain         | queries | candidates | MRR    | recall@1 | recall@5 | NDCG@10 |
| -------------- | ------- | ---------- | ------ | -------- | -------- | ------- |
| `pc_hardware`  | 50      | 100        | 0.6214 | 0.52     | 0.72     | 0.6434  |
| `mac_hardware` | 50      | 100        | 0.4972 | 0.38     | 0.60     | 0.5980  |
| overall        | 100     | 100        | 0.5593 |          |          |         |

**Retrieval on the drifted domain is 20.0% worse than on the baseline domain**, 0.497
against 0.621. That is the premise of the project appearing in a measurement: drift is not
only a centroid moving, it is a measurable loss of retrieval quality on the shifted
distribution. It is also the one number here produced by code that shares nothing with the
pipeline it measures.

This step runs after the smoke test, so the model answering it is the **adapted** one the
run just promoted, not the base encoder. The gap above is therefore what survives
adaptation, and the pre-adaptation gap is wider. An earlier version of this document
reported 28% from run `30589884101`, which GitHub no longer serves — the figure could not
be re-derived from a deleted run, so it was replaced by a measurement that links.

## Domain adaptation

A drift alert triggers a LoRA adaptation of `sentence-transformers/all-MiniLM-L6-v2` over
the drifted window, followed by an ONNX export and an evaluation against the base model.
Each query must retrieve its own document out of every candidate, which is the same task
the adapter trains on, so the gate and the objective agree.

Five runs of the identical scenario, with in-batch negatives only. This was the shipped
objective through commit `0dc89a0`; mined hard negatives replaced it in
[#42](https://github.com/JCHETAN26/Continuum/pull/42) and are measured in the next section.

| run                                                                              | training samples | baseline MRR | candidate MRR | change |
| -------------------------------------------------------------------------------- | ---------------- | ------------ | ------------- | ------ |
| [`30605700975`](https://github.com/JCHETAN26/Continuum/actions/runs/30605700975) | 124              | 0.4756       | 0.4801        | +0.62% |
| [`30659803585`](https://github.com/JCHETAN26/Continuum/actions/runs/30659803585) | 230              | 0.4455       | 0.4439        | −0.49% |
| [`30658311041`](https://github.com/JCHETAN26/Continuum/actions/runs/30658311041) | 250              | 0.4429       | 0.4458        | +0.87% |
| [`30607125415`](https://github.com/JCHETAN26/Continuum/actions/runs/30607125415) | 400              | 0.4113       | 0.4474        | +9.86% |
| [`30654126895`](https://github.com/JCHETAN26/Continuum/actions/runs/30654126895) | 450              | 0.4146       | 0.4487        | +8.29% |

**Gains appear only in the runs that captured 400 or more examples.** The three runs under
250 examples land between −0.49% and +0.87%, which is noise around zero in both directions;
the two larger runs gain eight to ten percent.

An earlier version of this table had four runs and called the relationship monotonic. The
fifth run, at 230 examples, scored below the run at 124 and broke that ordering. What
survives is the separation between the two clusters, not a smooth trend, and five points
cannot locate where between 250 and 400 the behaviour changes.

The window size varies between runs because it depends on which documents happened to land
inside the drift window, which is why these results first read as an unreliable adapter.
Nothing here rules out other differences between runs; the mechanism — more contrastive
pairs, better adapter — is at least the ordinary one.

**The candidate was rejected in all five runs**, against a 10% activation gate. Even the
best run fell 0.14 points short. That is the system working: it measured a candidate, found
no gain worth a model swap, and kept serving the baseline.

### Hard negatives, and why they are not claimed as an improvement

Each pair now also carries a negative mined from the base model's own confusions. Four runs
of the same scenario, this time with mined negatives in the denominator:

| run                                                                              | training samples | baseline MRR | candidate MRR | change      |
| -------------------------------------------------------------------------------- | ---------------- | ------------ | ------------- | ----------- |
| [`30664356329`](https://github.com/JCHETAN26/Continuum/actions/runs/30664356329) | 119              | 0.4626       | 0.4661        | +1.17%      |
| [`30663295295`](https://github.com/JCHETAN26/Continuum/actions/runs/30663295295) | 270              | 0.4324       | 0.4494        | +4.22%      |
| [`30664356329`](https://github.com/JCHETAN26/Continuum/actions/runs/30664356329) | 400              | 0.4242       | 0.4508        | +7.02%      |
| [`30672478186`](https://github.com/JCHETAN26/Continuum/actions/runs/30672478186) | 400              | 0.4402       | 0.4851        | **+10.44%** |

Set against the runs above at comparable sizes, the comparison does not settle anything:

| training samples | in-batch only  | with hard negatives |
| ---------------- | -------------- | ------------------- |
| ~120             | +0.62%         | +1.17%              |
| ~230–270         | −0.49%, +0.87% | +4.22%              |
| 400              | +9.86%         | +7.02%, +10.44%     |

**The two runs at 400 examples on identical code differ by 3.4 points**, +7.02% against
+10.44%, one below the activation gate and one above it. That is the clearest available
statement of how noisy this measurement is: the within-condition spread at a fixed
training-set size is wider than any difference between the two objectives. No improvement
from hard negatives is claimed, and none of the numbers in the comparison table above
survive that spread.

Answering this properly needs a dedicated experiment that pins the training-set size and
runs both arms several times, rather than the opportunistic sizes CI happens to produce.
That has not been done. Hard negatives are shipped because they are the standard
construction for this objective and they are correct, not because they have been shown to
beat the alternative here.

### The loop has closed once, end to end

Run [`30672478186`](https://github.com/JCHETAN26/Continuum/actions/runs/30672478186) is the
first in which a candidate cleared the gate. Nine runs, eight rejections, one promotion.

| stage              | evidence                                                        |
| ------------------ | --------------------------------------------------------------- |
| Drift detected     | `score=0.181, threshold=0.080, breached=True`                   |
| Training triggered | `status=SUCCEEDED, samples=400`                                 |
| Gate passed        | `improvement=+10.44%, gate=+10%`                                |
| Model promoted     | `version=2026.07.31-c9dfab84, status=ACTIVE`                    |
| Serving switched   | `served_by=2026.07.31-c9dfab84`                                 |
| Re-indexing began  | `Embedded batch documents=50 model_version=2026.07.31-c9dfab84` |

The last line is the one worth reading closely. On activation the embedding worker starts
re-encoding the existing corpus under the new version, because vectors from two different
encoders are not comparable. It claims documents whose `model_version_id` differs from the
active model, so the work is resumable and spreads across replicas.

**The end-to-end test does not wait for that backfill to finish.** It observed re-indexing
in progress, at 50 documents per batch against a corpus of 1200, and then tore the stack
down. That a full backfill completes is therefore not demonstrated; that activation triggers
one, and that the re-encoding uses the promoted model, is.

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

Four runs: [`30605700975`](https://github.com/JCHETAN26/Continuum/actions/runs/30605700975),
[`30607125415`](https://github.com/JCHETAN26/Continuum/actions/runs/30607125415),
[`30658311041`](https://github.com/JCHETAN26/Continuum/actions/runs/30658311041) and
[`30877829855`](https://github.com/JCHETAN26/Continuum/actions/runs/30877829855).

| batch | p50 range      | p95 range      | slowest observed |
| ----- | -------------- | -------------- | ---------------- |
| 1     | 27 – 82 ms     | 38 – 145 ms    | 186 ms           |
| 8     | 346 – 920 ms   | 503 – 1537 ms  | 1580 ms          |
| 32    | 1603 – 3485 ms | 2235 – 4085 ms | 4318 ms          |

**Run-to-run spread reaches 3.0x at batch 1 and 2.2x at batch 32.** These are shared CI
runners with no isolation, so the figures support an order-of-magnitude claim and nothing
finer. The fourth run widened every band: batch 1 p50 came in at 82 ms against a previous
worst of 71 ms. Any single-run figure quoted from this table will be contradicted by the
next run, so quote the range.

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
roughly 32x after the 3x improvement, against 150x to 284x before it.

Batch 1 spans 27 – 82 ms p50 across four runs, so it **straddles the 50 ms target rather than
meeting it**: two runs land inside, two outside. The remaining gap is mostly throughput under
batching rather than per-request cost, but "batch 1 meets the target" is not a claim these
measurements support.

## Retrieval against the whole corpus

The section above ranks each query against 100 candidates. That pool size flatters the
numbers: with 100 candidates a random guess lands at MRR ≈ 0.05, and the task gets steadily
harder as the pool grows. `eval/benchmark.py --full-corpus` ranks 500 queries against every
post that survives the corpus filters, which is 8,525 of them.

Run [`30877829855`](https://github.com/JCHETAN26/Continuum/actions/runs/30877829855):

| queries | candidates | MRR    | recall@1 | recall@5 | NDCG@10 |
| ------- | ---------- | ------ | -------- | -------- | ------- |
| 500     | 8,525      | 0.3063 | 0.208    | 0.412    | 0.4756  |

**One in five queries retrieves its own document first out of 8,525.** The drop from 0.56 to
0.31 MRR between the two pool sizes is the pool, not a regression — quoting either number
without its candidate count says nothing.

The corpus funnel that produces 8,525: 18,846 raw posts, 18,331 non-empty, 12,414 within the
25–200 word bounds, 12,380 after dedup, and 8,525 long enough to split into a 15-word query
plus a 40-word document.

## Data plane at 100k vectors

The demo runs on 600 documents, which says nothing about how the storage layer behaves at a
size worth worrying about. `bench/load_vectors.py` loads 100,000 synthetic vectors into the
real schema with the HNSW index live, then times and `EXPLAIN`s the queries the services
actually run.

Run [`30877829855`](https://github.com/JCHETAN26/Continuum/actions/runs/30877829855):

| query                              | time      | plan                     |
| ---------------------------------- | --------- | ------------------------ |
| ANN search (HNSW, top 10)          | 1.20 ms   | —                        |
| claim, stale branch (backlog)      | 0.59 ms   | SEQ SCAN, 646 buffers    |
| claim, stale branch (settled)      | 65.80 ms  | **index, 31 buffers**    |
| claim, unembedded branch (settled) | 61.36 ms  | SEQ SCAN, 78,691 buffers |
| drift centroid over window         | 297.53 ms | —                        |
| re-index UPDATE                    | 16.3 s    | 5,000 rows, 307 rows/sec |

### The benchmark caught a sequential scan on the worker's poll

The embedding worker claimed documents with one query:

```sql
LEFT JOIN embeddings e ON e.document_id = d.id
WHERE e.id IS NULL OR e.model_version_id IS DISTINCT FROM $1
```

`IS DISTINCT FROM` has no btree strategy and neither does `<>`, so no index could serve that
`OR` however the tables were indexed. Postgres hashed both tables and filtered. The settled
state was the expensive one: once every vector carries the active version the query matches
nothing, `LIMIT 50` cannot stop early, and the worker pays a full scan of both tables once a
second forever.

[#49](https://github.com/JCHETAN26/Continuum/pull/49) split it into the two claims it always
was and spelled the version test as `IS NULL OR < $1 OR > $1` — the same predicate, in a form
the planner answers from the **existing** `embeddings(model_version_id)` index with a
BitmapOr. **Pages touched fell from 49,040 to 31**, about 1,580x. No index was added; the one
that was needed already existed and the query shape was keeping it unreachable.

### Buffers, not milliseconds, are the signal in that table

Two things distort the wall-clock and both are visible above.

Every claim query ends in `LIMIT 50`, so a sequential scan that finds fifty rows and stops is
cheap — the backlog row reads `SEQ SCAN` at 0.59 ms. Access method alone does not separate a
healthy plan from a sick one; pages touched does.

The three settled measurements run after a statement that rewrites all 100,000 rows, and they
cluster at 61–67 ms regardless of whether they touch 31 pages or 78,691. Everything measured
before that rewrite is fast (ANN at 1.20 ms), so the floor is contention from the rewrite —
dead tuples and the autovacuum they trigger on a ~380 MB table with an HNSW index — not the
queries. Forcing a `CHECKPOINT` first did not move it. On an idle local instance the fixed
query runs in **0.15 ms**; CI cannot show that, so the buffer counts carry the claim.

### The planner needs statistics before its plans mean anything

The first run of the fix still reported `SEQ SCAN`. The rewrite was correct; the benchmark
was not. `COPY` does not update `pg_stats`, neither does the bulk `UPDATE` that reaches the
settled state, and nothing waited for autovacuum — `pg_stats` returned **zero rows** for
`model_version_id`. A planner that cannot tell how selective a predicate is will not risk an
index, so it scanned. `ANALYZE` now runs after the load and again after the re-version.

The un-analysed case is kept as its own row in the benchmark rather than deleted. It is real:
it is what the worker meets in the minutes after a re-index, before autovacuum catches up.

### What is still slow

The unembedded branch stays a sequential scan at 78,691 buffers and no index fixes it:
proving that no document is missing a vector means visiting every document. The drift
centroid at 297 ms is the same shape — an aggregate over a window that legitimately reads
what it covers.

Re-indexing at ~307 rows/sec means a 100,000-document corpus takes about five minutes to
re-encode after an activation. That is why re-encoding is the worker's job and resumable
rather than a step inside the training pipeline.

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
