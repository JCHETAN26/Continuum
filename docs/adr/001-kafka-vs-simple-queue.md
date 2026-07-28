# ADR 001: Kafka (Redpanda) over a Simple Queue

## Status

Accepted

## Context

Continuum needs to process a high-volume stream of incoming documents to compute embeddings, maintain running drift statistics, and potentially trigger fine-tuning jobs. The ingestion endpoint (FastAPI) must acknowledge receipt quickly and hand off the document to asynchronous workers. We considered using a simple Redis queue (e.g., BullMQ or RQ) or a Postgres-based queue, which we are already using for tracking training jobs.

## Decision

We chose to use Kafka (specifically Redpanda for local development and simplified deployment) for the document stream rather than a simple queue.

## Rationale

1. **Exactly-Once Processing**: Kafka allows us to maintain idempotency keys and use transactional offset commits to guarantee exactly-once processing, which is critical for accurate drift statistics.
2. **Replayability**: If we need to recalculate embeddings due to a model change, or if a bug is found in the drift detection math, Kafka allows us to easily replay the event stream from a specific point in time. Simple queues typically delete messages upon consumption.
3. **High Throughput**: Kafka is designed for high-throughput streaming and can handle the expected load better than Postgres-based queues.
4. **Consumer Groups**: As the load increases, Kafka's consumer groups allow us to seamlessly scale out the drift detection workers horizontally without complex locking mechanisms.

## Consequences

- **Operational Complexity**: Introducing Kafka (Redpanda) adds another piece of infrastructure to manage.
- **Learning Curve**: The team needs to be familiar with Kafka concepts (topics, partitions, offsets, consumer groups).
- **Tooling**: We will use Redpanda which is API-compatible with Kafka but operationally simpler (no ZooKeeper).
