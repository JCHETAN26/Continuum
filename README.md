# Continuum 🌌

**Continuum** is a local real-time embedding drift detection and adaptive model registry demo.

Every production RAG and semantic search system suffers from semantic drift over time. The distribution of data flowing through the system shifts away from the distribution the embedding model was originally trained on. Continuum demonstrates the control loop: ingest documents, compute embeddings, score centroid drift, trigger an adaptation job, register a candidate model, and activate it from the dashboard.

## Architecture

```mermaid
graph TD
    Client[Client Apps] -->|POST /v1/ingest| Ingest[Ingestion API]
    Ingest -->|Kafka Topic| Kafka((Redpanda))

    Kafka -->|Consume Stream| IngestWorker[Ingest Worker]
    IngestWorker --> PG[(pgvector)]
    Embedder[Embedding Worker] --> PG
    Drift -->|Compute Centroids| PG
    Drift -->|Alert Threshold breached| Kafka

    Kafka -->|Consume Alert| Trainer[Trainer Engine]
    Trainer -->|Demo Adaptation Job| MinIO[(MinIO S3)]
    Trainer -->|Register Model| PG

    Client -->|POST /v1/embed| Server[Serving Engine]
    Server -->|Poll for ACTIVE| PG
    Server -.->|Poll Active Model| PG

    Dashboard[Next.js Dashboard] -.->|SSE + REST| Drift
    Dashboard -.->|SSE + REST| Trainer
```

## Production Ops

Continuum has two migration tracks. Prisma owns the product schema in
`packages/shared/prisma/migrations`; Alembic owns operational tables for hashed API keys,
request metrics, and rollback audit events in `packages/shared/alembic`.

```bash
pnpm ops:migrate
pnpm ops:migrate:test
```

Docker Compose runs the same Alembic upgrade through the `migrations` one-shot service before app
containers become eligible to serve traffic.

Long-running services declare `restart: unless-stopped` so a crash does not silently remove a
service from the loop. The one-shot containers (`redpanda-init`, `minio-init`, `migrations`)
deliberately have no restart policy, because dependents wait on them via
`service_completed_successfully` and a restart loop would block startup forever.

`DATABASE_URL` pins `connection_limit=10&pool_timeout=30`. Left implicit, Prisma sizes its pool
from the CPUs the container can see (`num_cpus * 2 + 1`), which the per-service `cpus` limits
reduce to 3 — small enough that the drift services exhaust it under sustained load and exit on
`Timed out fetching a new connection from the connection pool`. Postgres runs with
`max_connections=200` to leave headroom above the summed pools for migrations and seeding.
Alembic strips the Prisma-only parameters before psycopg sees the URL
(`continuum_shared.db_url`), since libpq rejects connection options it does not recognise.

Every FastAPI service emits structured JSON logs with an `x-correlation-id` value, and local
OpenTelemetry spans are exported to stdout. Set `API_KEY_BCRYPT_HASH` instead of `API_KEY` in
real deployments; plaintext `API_KEY` exists only for local development compatibility.

`retention-worker` runs the cleanup policy on a schedule:

- embeddings older than `RETENTION_EMBEDDINGS_DAYS` (default 90)
- drift and linguistic windows older than `RETENTION_DRIFT_WINDOWS_DAYS` (default 30)
- training jobs older than `RETENTION_TRAINING_JOBS_DAYS` (default 365)

`DRIFT_TRIGGER_MIN_EMBEDDING_DRIFT` is on the same scale as `DRIFT_THRESHOLD`: centroid cosine
distance, where a domain shift reads around 0.10. Keep it at or below the alert threshold, or the
throttler suppresses every alert the drift service raises and only linguistic drift can trigger
training.

The serving engine tracks request outcomes per model version. If the active model exceeds
`ROLLBACK_ERROR_RATE_THRESHOLD` over `ROLLBACK_WINDOW_SECONDS` with at least
`ROLLBACK_MIN_REQUESTS`, it archives the failing version, restores the previous active model,
and records a rollback event.

## Running it in a Codespace

`.devcontainer/` configures a Codespace with Docker-in-Docker and the ports the stack
publishes. It requests **8 CPUs and 32 GB**: the app services alone declare 9.2 GB of
memory limits and 8 CPUs, the infrastructure containers are unbounded on top of that, and
serving is CPU-bound even at rest.

Create the Codespace, wait for setup, then follow the quickstart below. Port 3000 forwards
automatically once the dashboard is healthy.

Codespaces bill by core-hour while running, storage while stopped, and nothing once
deleted, so stop or delete it when finished.

## Quickstart (E2E Demo)

Boot up the entire infrastructure and microservices with Docker Compose:

```bash
docker compose up --build -d
```

Compose healthchecks gate the app startup path: APIs wait for Redpanda/Postgres/Redis/MinIO as needed, and the dashboard waits for the drift and trainer APIs to become healthy.

To verify the exposed services from the host after startup:

```bash
pnpm stack:health
```

1. Open the Dashboard at `http://localhost:3000`
2. Run the seed script to inject baseline data and then induce a drift event:
   ```bash
   uv run scripts/seed.py
   ```
3. Watch the Dashboard as the drift score spikes, the training job triggers, and the new model version is produced!
4. Inspect retrieval quality for the active model on the new domain:
   ```bash
   pnpm bench:retrieval
   ```
5. Or verify the full demo narrative end-to-end:
   ```bash
   pnpm demo:verify
   ```

See [DEMO.md](DEMO.md) for a detailed walkthrough.

### Trainer Backends

`TRAINER_BACKEND` defaults to `peft`: a drift alert trains
`sentence-transformers/all-MiniLM-L6-v2` with a LoRA adapter under a symmetric in-batch
contrastive objective, merges the adapter, exports ONNX through Optimum, and uploads the
artifacts to MinIO. Torch is installed only into the trainer image, through the
`INSTALL_EXTRAS` build arg, so the services that infer through ONNX Runtime stay lean.

The candidate is then scored against the base model on the same held-out documents and
promoted only if it clears `ACTIVATION_MIN_IMPROVEMENT`. Activation re-indexes the corpus:
vectors from two different encoders are not comparable, so an adapted model serving against
an index built by its predecessor would degrade retrieval and register as drift that never
happened.

The re-encoding is done by the embedding worker, which claims documents whose stored vector
carries a model version other than the active one. Activation is the only signal needed —
the mismatch is the work queue. That makes re-indexing resumable across restarts and spread
over however many worker replicas are running, rather than blocking the training job for the
length of a full pass.

`TRAINER_BACKEND=demo_adapter` selects the original deterministic projection instead. It
trains nothing, and exists for a fast laptop demo.

### Linguistic Drift

`continuum-linguistic-drift` adds a second drift signal over raw document text. It compares
rolling windows against the baseline corpus for entity movement, topic movement, and
vocabulary shift, stores results in `linguistic_windows`, publishes `linguistic-drift-alerts`,
and streams live dashboard updates from `http://localhost:8004/v1/linguistic/events`.

## Seed Corpus

`uv run scripts/seed.py` ingests real documents, not generated text: 700 posts from
`comp.sys.ibm.pc.hardware` as the baseline, then 500 from `comp.sys.mac.hardware` as the
drifted window, drawn from the
[20 Newsgroups](https://huggingface.co/datasets/SetFit/20_newsgroups) dataset. Every document
is distinct — the corpus is deduplicated and walked rather than sampled with replacement.

The pair was chosen by measurement. A domain shift only demonstrates adaptation if the base
model has somewhere left to improve, and MiniLM already separates unrelated subjects almost
perfectly:

| candidate pair                             | drift separation | baseline MRR         |
| ------------------------------------------ | ---------------- | -------------------- |
| `comp.*` vs `sci.med`                      | 0.9156           | 0.9833 — no headroom |
| `sci.med` vs `sci.electronics`             | 0.8886           | 0.9670 — no headroom |
| `rec.sport.baseball` vs `rec.sport.hockey` | 0.3457           | 0.9352 — usable      |
| **`pc.hardware` vs `mac.hardware`**        | **0.1760**       | **0.8591 — chosen**  |

A live run against `comp.* vs sci.med` measured baseline MRR at 0.9993, so no adapter could
register an improvement and the pipeline correctly rejected the candidate. Two flavours of
hardware support discussion share vocabulary and structure, which leaves real headroom.

Because that drift is subtler, `DRIFT_THRESHOLD` sits in a narrow band. Measured:

| window                                         | drift score |
| ---------------------------------------------- | ----------- |
| within-domain (PC vs PC, false-positive floor) | 0.0629      |
| 50% Mac                                        | 0.0789      |
| 75% Mac                                        | 0.1184      |
| 100% Mac                                       | 0.1728      |
| `DRIFT_THRESHOLD`                              | 0.08        |

0.08 clears the noise floor and trips on a drift-dominated window. Lowering it below ~0.065
would alert on a stable distribution.

## Scaling the workers

`embedding` and `ingest-worker` declare no `container_name` and publish no ports, so they
can be run with replicas:

```bash
docker compose --env-file .env -f infra/docker-compose.yml up -d --scale embedding=3
```

Both are safe to replicate. The embedding worker claims rows with
`FOR UPDATE ... SKIP LOCKED`, so replicas take disjoint batches without coordinating, and
the ingest worker is a Kafka consumer group over three partitions, so the broker assigns
each partition to one member.

Everything else keeps a fixed name deliberately. `redpanda-init`, `minio-init` and
`migrations` must run exactly once for `service_completed_successfully` to mean anything,
and a second `retention-worker` would duplicate the deletions the first is making.

## Retrieval Benchmark

`pnpm bench:retrieval` scores 100 held-out queries, 50 per domain, against the serving API.
Each query is the opening fifteen words of a post; the document is the rest of that post,
and it is the only relevant result among all 100 candidates. The split keeps the query out
of its own document, so a hit means the model matched meaning rather than finding a copy of
the query string.

This is deliberately harder than the gate the trainer applies to itself, which counts any
document from the same newsgroup as relevant. Half the candidates qualify under that rule,
which is why it reports MRR around 0.88 — a number that says more about the task than the
model. Finding one specific post among a hundred is something a retrieval model can be
wrong about.

CI runs it after the latency benchmark and publishes per-domain MRR, recall@1 and recall@5
to the job summary.

## Serving Latency

`pnpm bench:latency` measures p50/p95/p99 against a running stack at batch sizes 1, 8 and
32, using documents from the same corpus the demo ingests. Payload length matters: latency
on a transformer scales with sequence length, so measuring with short synthetic strings
understates it substantially.

CI runs it after the E2E smoke against the same images and resource limits, publishes the
table to the job summary, and uploads the raw JSON as an artifact.

Percentiles are nearest-rank rather than interpolated, so every figure is a latency some
request actually took.

## Embeddings

Documents are embedded with `sentence-transformers/all-MiniLM-L6-v2`, run through ONNX Runtime
rather than torch: the published ONNX export plus a tokenizers vocabulary keeps the service
images small, and the serving engine already loads ONNX sessions for adapted models.

The weights are baked into the image at build time (`scripts/fetch_embedding_model.py`), so no
container reaches the network at startup and every replica serves byte-identical vectors. Drift
is measured by comparing centroids over time, so replicas on different revisions of the model
would register as drift that never happened.

Pooling — attention-masked mean, then L2 normalisation — is duplicated in
`continuum_trainer.peft_engine`. The two must stay in step: an adapter trained under one
pooling strategy and served under another yields vectors that are silently incomparable with
the baseline centroids.

## Measured Results

A LoRA adaptation of `all-MiniLM-L6-v2` over the drifted window improves retrieval MRR by
0.77% to 0.96% across two CI runs, tuning 337,920 parameters — 1.47% of the model.
Composite improvement stayed below the 10% activation gate in both, so the pipeline kept
serving the baseline rather than shipping a marginal model.

Serving latency is measured in CI: median 101–299 ms at batch 1 and 7.5–14.2 s at batch 32
across five runs, against a spec target of p99 under 50 ms. The serving container is capped
at 0.50 CPU and every request pads to 256 tokens, which accounts for it; both are recorded
rather than tuned away.

Full figures, method, and the CI run that produced them:
[docs/benchmarks/RESUME_METRICS.md](docs/benchmarks/RESUME_METRICS.md).

## Validation

The default checks do not require Docker:

```bash
pnpm lint
pnpm type-check
pnpm test
pnpm build
docker compose config -q
```

To verify the complete compose stack from a clean local state:

```bash
docker compose --env-file .env.example -f infra/docker-compose.yml down -v
docker compose --env-file .env.example -f infra/docker-compose.yml up --build -d --wait
pnpm stack:health
pnpm demo:verify
```

Docker-backed Testcontainers checks are opt-in:

```bash
pnpm test:integration
```

## Services Overview

- **`apps/ingest`**: FastAPI service that validates document payloads and produces idempotently to Kafka.
- **`apps/embedding`**: Embeds documents with `all-MiniLM-L6-v2` via ONNX Runtime and stores the vectors in pgvector.
- **`apps/drift`**: Computes rolling centroids of the embedding space and fires alerts based on cosine distance.
- **`apps/trainer`**: Runs a deterministic demo adaptation/evaluation job before registering models.
- **`apps/server`**: REST + gRPC embedding service with active-model polling and hot-swap state.
- **`apps/dashboard`**: Next.js 15 UI for monitoring drift, training telemetry, and the model registry.
