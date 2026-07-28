# ADR-003: Contrastive In-Batch Negative Loss For Domain Adaptation

## Status

Accepted

## Context

Continuum adapts embedding models when incoming documents drift into a new domain. The
training signal available at drift time is usually unlabeled text from the shifted window,
not curated query/document relevance labels. A supervised fine-tuning objective would
therefore require either manual labels or synthetic labels that risk teaching the model the
wrong task.

## Decision

The Phase 1 PEFT trainer uses LoRA with an in-batch contrastive objective. Each batch is
encoded with mean pooling over transformer token embeddings, L2-normalized, then scored
with a cosine-similarity matrix. Cross-entropy over the diagonal treats each example as its
own positive and every other in-batch example as a negative, with temperature `0.05`.

The trainer uses:

- Base model: `sentence-transformers/all-MiniLM-L6-v2`
- LoRA rank: `8`
- LoRA alpha: `16`
- Target modules: `query`, `key`, `value`, `dense`
- Dropout: `0.05`
- Epochs: `3`
- Batch size: `16`
- Learning rate: `2e-4`

## Consequences

This objective works with raw drifted text, keeps the adaptation job cheap, and improves
domain separation without requiring a labeling pipeline. It also keeps the LoRA adapter
small enough to merge and export to ONNX for later serving.

The tradeoff is that it is not a replacement for task-specific supervised ranking data.
When high-quality relevance judgments exist, we should add a later evaluation/promotion
stage that gates on those labels before activation.
