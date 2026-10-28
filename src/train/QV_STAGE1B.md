# Quartz Stage 1B — frozen-prefix, last-layer Q/V LoRA (2026-10-28)

The first real output-head-only LoRA was a separate Stage 1A milestone.
This stage trains **Q and V adapters inside the LAST dense transformer block**
with a genuine end-to-end last-token cross-entropy derivative.

## Maths and integration

- Use native Quartz mmap GGUF loader and tokenizer, no Python/PyTorch
  inference or training calls. Base GGUF remains frozen and read-only.
- Cache output of layers 0..(n-2) for each token using
  embed_and_layers_up_to. Q/K/V frozen projections are cached for the
  last layer, including bias and optional QK norm/RoPE for K.
- For each optimizer step, recompute the LAST layer: Q/V adapters on
  frozen-prefix RMSNorm inputs, GQA attention including sink/window policy,
  frozen attention output projection, frozen residual+FFN SwiGLU+RMSNorm,
  final RMSNorm and full-vocabulary frozen output head.
- Backprop exactly from one target token's cross-entropy via the frozen
  W_head transpose, final RMSNorm, residual, FFN-down transpose,
  SwiGLU derivatives, FFN-up/gate transposes, FFN RMSNorm,
  attention-output transpose, grouped-query softmax derivative,
  inverse RoPE/QK RMSNorm, then Q and V LoRA A/B gradients.
- Base gradients are not stored. No gradients are propagated into the
  earlier frozen prefix, which has no trainable parameters.
- Transpose GEMV dequantizes 32 GGUF weight rows at a time, not an entire
  f32 model. Real 0.5B GGUF: Q matrix [896x896], V [128x896].
- Optimizer: gradient-clipped Adam updating Q/V A/B together.
- Safetensors metadata binds the adapter to its exact base GGUF SHA-256
  and last layer. Stores four F32 tensors (Q A/B, V A/B).
- Verify baseline logit parity at B=0 with original Quartz generation
  forward before any update. Verify save/reload, frozen base hash and
  both nonzero Q/V B weights after training.

## CLI

  quartz train-qv-last model.gguf dataset.jsonl last_qv.safetensors \
      --rank 2 --steps 8 --lr 0.01 --max-tokens 16

  quartz eval-qv-last model.gguf fresh-heldout.jsonl last_qv.safetensors \
      --max-tokens 16

  quartz train-qv-last ... --gradient-check

The optional last switch numerically compares analytical end-to-end
gradients for Q.A, Q.B, V.A and V.B with nonzero A/B on real GGUF data.

## Correctness and phone evidence

- Five new finite-difference gradient tests passed for RMSNorm,
  optional per-head QK norm, NeoX/adjacent-pair RoPE including YaRN,
  SiLU and GQA attention (including sink and causal-window cases).
- Portable adapter round-trip, wrong-base/wrong-layer rejection and
  optimizer Q+V update tests passed.
- Trainer test suite: 16/16 passing.
- 2026-10-28 desktop test: Qwen2.5-0.5B-Instruct (24 layers),
  rank 2, layer 23. Normal Quartz forward parity exactly matched
  (relative maximum logit error reported 0), both Q/V weights updated.
  One optimizer step improved training CE 0.9410058 -> 0.8646344;
  two held-out examples 0.3764773 -> 0.3510118.
- 2026-10-28 S26 ARM64 smoke: rank 2, 8 steps, two CPU workers:
  10/28 epoch run training CE 0.9410000 -> 0.0264473,
  two held-out CE 0.3764815 -> 0.0534430.
  End-to-end process 87.355 seconds; 57% battery start/end,
  thermal status 0 start/end, frozen GGUF unchanged.
  Adapter SHA-256:
  86392168326c1b4c271fa1b72a729671084e1c5fe225a0e206e733078f28a4c8
- Full source log recorded in experiments/android-lora-smoke/results/.

## Limits — important

This is NOT last-eight-layer LoRA. Only Q/V of the LAST block change.
Only the final token of each short prompt has a target; it is not a
general teacher-forced full-sequence trainer. No Q/K/O adapters,
cross-layer reverse pass, GGUF merge, resumeable optimizer checkpoints,
production inference integration or Vulkan training is claimed.
The narrow six-example two-label dataset is insufficient to establish
robust generalization. The independent fresh-example evaluation
is a separate acceptance gate. These are experimental prototypes.
