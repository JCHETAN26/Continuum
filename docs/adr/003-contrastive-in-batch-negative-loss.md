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
encoded twice with dropout active, producing two views of every document. Both views are
mean-pooled over transformer token embeddings and L2-normalized, then scored against each
other as a cosine-similarity matrix. Cross-entropy treats view B of a document as the
positive for view A, and every other document in the batch as a negative, with temperature
`0.05`. The loss is symmetrised over both directions of the matrix.

The two views are what make the objective learnable, and this is not incidental. Scoring a
batch against itself puts self-similarity on the diagonal, which is exactly `1.0` for any
L2-normalized embedding. The positive is then unbeatable and already attained at
initialization: measured loss was `1e-6` on random embeddings, and real LoRA runs stayed
flat across three epochs (`1.9294 → 1.9678`). All that remains is a repulsive gradient
pushing every document away from every other, which is the opposite of what retrieval
needs. Dropout supplies the augmentation, following SimCSE, so no labels are required.

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

Two forward passes per step roughly doubles training compute, which is an acceptable cost
at Phase 1 batch sizes and is what buys a non-degenerate objective.

Because dropout is the only source of augmentation, the objective silently collapses back
to self-similarity if the encoder is ever put in eval mode or configured with zero dropout.
`transformers.Trainer` keeps the model in training mode during `train()`, and both the base
model's `hidden_dropout_prob` and the adapter's `lora_dropout` contribute. Unit tests pin
the property directly: pairing a batch with itself must score near zero, while two genuine
views must not.
