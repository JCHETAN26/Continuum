# ADR 002: LoRA over Full Fine-Tuning

## Status
Accepted

## Context
When semantic drift is detected, Continuum needs to adapt the embedding model to the new domain. This requires fine-tuning the base model (e.g., `all-MiniLM-L6-v2`) on a sample of recently drifted documents and hard negatives. Full fine-tuning of all model weights is computationally expensive, requires significant memory, and produces a large artifact for every model version.

## Decision
We chose to use Low-Rank Adaptation (LoRA) via Hugging Face PEFT for fine-tuning the models instead of full fine-tuning.

## Rationale
1. **Memory Efficiency**: LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture. This drastically reduces the number of trainable parameters (often by 10,000x) and the VRAM required for training.
2. **Fast Adaptation**: Training only a small set of parameters is faster, allowing Continuum to react more quickly to detected drift.
3. **Hot-Swapping and Storage**: LoRA adapters are small (typically a few megabytes). We can export them alongside the base model into ONNX format. This significantly reduces the storage footprint in the model registry (MinIO) compared to storing full model checkpoints for every version. It also makes background loading and hot-swapping during inference much cheaper.
4. **Performance**: Empirical results show that LoRA achieves comparable performance to full fine-tuning on domain adaptation tasks for embedding models.

## Consequences
- **Dependency**: We rely on Hugging Face PEFT for training and ONNX Runtime for inference of the adapted models.
- **Complexity in Serving**: The serving layer needs to correctly load the base model and apply the specific LoRA adapter for the requested version. We will bake the adapter into the ONNX graph during export for simpler serving.
