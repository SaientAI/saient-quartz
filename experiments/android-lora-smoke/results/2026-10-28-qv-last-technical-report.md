# Quartz Stage 1B — verified Samsung Q/V LoRA training, 28 October 2026

Device: SM-S948B, Android 17, Qualcomm SM8850. Model:
Qwen2.5-0.5B Instruct GGUF, 24 layers, 896 hidden, 151936 vocab.
Base SHA-256: 6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653
ARM64 binary SHA-256: d09a3fc91b7be8dd6c7ec37ccaa9de72047581a3a9f5773af0e478dd9d08f609

Trainable parameters: Q and V only in final transformer layer 23,
LoRA rank 2, alpha 4. Original quantized GGUF frozen and unchanged.
Frozen prefix layers 0..22 run through the native Quartz forward path;
the final block executes and backpropagates through frozen output
projection, SwiGLU FFN, RMSNorm, output head, GQA attention and RoPE
into Q/V LoRA parameters.

Phone training — Forge request quartz-qv-s26-genuine-last-layer-training-20261028-01
  4 training + 2 held-out sentiment examples; 8 Adam steps, lr 0.01.
  Original Quartz zero-adapter logits: exact parity (reported max relative 0).
  Training CE: 0.9410000 -> 0.0264473.
  Held-out CE: 0.3764815 -> 0.0534430.
  Both adapters changed: Q.B L2=1.7755568, V.B L2=0.7285355.
  Total runtime 87.355 seconds. Battery 57% -> 57%; thermal 0 -> 0.
  Base GGUF unchanged; adapter serialized and reloaded successfully.
  S26 output adapter 23 KB, SHA-256:
    86392168326c1b4c271fa1b72a729671084e1c5fe225a0e206e733078f28a4c8

Separate S26 evaluation — request quartz-qv-independent-s26-unseen-eval-20261028-01
  6 new prompts unseen during training or original heldout test.
  Base CE average 1.0799055, adapted CE average 0.0712443.
  Each individual example improved. Evaluated 56.979 seconds,
  Android thermal 0. No additional training updates.

Real-model numerical gradient check — host job 0e72b17d40e64c2c9a44d737c988aabf
  Tested with deliberately nonzero Q/V LoRA A and B:
  Q.A analytic -4.50476980 vs numeric -4.48281765 (relative 0.2442%)
  Q.B analytic  3.68963361 vs numeric  3.65677691 (relative 0.4472%)
  V.A analytic  0.89475381 vs numeric  0.89485645 (relative 0.0057%)
  V.B analytic -0.48939282 vs numeric -0.48921108 (relative 0.0186%)
  All four passed. Native trainer tests 16/16 passed.

LIMITATIONS: This is genuine last-layer Q/V fine-tuning, not last-eight-
layer or full-sequence training. One target token per prompt, tiny
sentiment-only datasets, CPU reference path. No production inference
binding to Q/V adapters, long-sequence SFT, resumeable checkpoints,
Vulkan training or GGUF merge yet. Improvement on six new examples
is not evidence of broad/general capability.

Original raw logs are retained beside this report.
