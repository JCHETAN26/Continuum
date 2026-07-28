# ADR-004: Linguistic Drift Sidecar

## Status

Accepted

## Context

Centroid drift catches embedding-space movement, but it does not explain why a domain shift
happened. The demo narrative also needs a human-readable signal: new entities, new topics, and
vocabulary movement when healthcare documents enter a software-doc baseline.

## Decision

Add `continuum-linguistic-drift` as a separate Python service. It reads document text from
Postgres, compares a rolling window against the earlier baseline corpus, persists scored windows
to `linguistic_windows`, and emits `linguistic-drift-alerts` when the composite score breaches its
threshold.

The analyzer uses three signals:

- Entity distribution movement, scored with Jensen-Shannon distance and exposed as
  `entity_kl_divergence`.
- Topic share movement, scored with Wasserstein distance. BERTopic is loaded lazily when available;
  a deterministic TF-IDF topic fallback keeps local tests fast and offline.
- Vocabulary shift, scored with a chi-square p-value and emerging term ratios.

The dashboard consumes `/v1/linguistic/events` with the same SSE snapshot loop as semantic drift.

## Consequences

The service can evolve independently from embedding drift and can later weight training jobs through
`training_linguistic_signals`. Runtime deployments may install the optional `nlp` extras and a spaCy
model for richer extraction, while CI can validate the deterministic fallback without model
downloads.
