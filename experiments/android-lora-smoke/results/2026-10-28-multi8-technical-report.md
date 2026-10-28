# Quartz Stage 2 — Real internal transformer LoRA on Samsung S26

**Verified 28 October 2026.** Standalone native Rust code; frozen GGUF
with genuine multi-block causal reverse-mode. No cloud training.

## Setup

- Device: Samsung SM-S948B, Android 17, Qualcomm SM8850.
- Model: Qwen2.5-0.5B Instruct GGUF, 24 dense transformer blocks,
  hidden 896, 14 query heads, two KV heads, vocab 151936.
- Frozen base model SHA-256:
  6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653
- Native phone executable SHA-256:
  1cf7855d19e17df14b874debad00c2be7cb99aa62204b595daee4459e395a260
- Train Q/V LoRA in layers 16 through 23 (8 interconnected
  transformer blocks), rank 2, alpha 4; 45,056 trainable parameters.
- One target next token per short prompt. Training data: 2 examples;
  original holdout 1; independent evaluation 2 new examples.
  All are binary sentiment classification. This is a tiny, controlled
  experiment, not a general language-learning benchmark.
- Optimization: CPU, 2 Rayon threads, clipped Adam (per block), lr 0.01.
  Other 16 blocks and all base GGUF matrices are frozen.

## Correctness

- Native zero-adapter logits match ordinary Quartz inference
  (reported relative maximum logit error 0.000000000).
- A two-layer desktop real-GGUF experiment passed all eight
  numerical derivative checks (Q.A/Q.B/V.A/V.B at layers 22 and 23).
- An eight-layer desktop experiment passed another eight real-model
  numerical gradient checks (same four matrices at layers 16 and 23),
  thus validating paths across seven internal layer boundaries.
  Largest reported symmetric relative disagreement: 1.0521%
  (layer 16 V.A, finite-difference approximation).
- Full Rust test suite: 117 passed, 0 failed, 11 ignored.
- Both Q and V adapter tensors updated in each of the eight layers.
- The original GGUF file SHA-256 was unchanged; saved adapters
  reloaded with identical heldout-loss values.

## First native S26 checkpoint: one step

Forge request quartz-eight-internal-layers-s26-train-20261028-01

- Training cross-entropy: 1.0487998 -> 0.4235311.
- Original heldout cross-entropy: 0.6603754 -> 0.2987503.
- 55.796 seconds, battery 54% at both checkpoints, Android thermal
  status 0 at both checkpoints.
- Safetensors: 182,936 bytes, SHA-256:
  5f98c09934c3f0479a04a31479328c4abb65f1a26b22c0bfcf6c531b36ad1074
- Independent new S26 process, two previously unseen prompts:
  base CE 1.1034586, adapted CE 0.3857160.
  Both examples improved. 20.607 seconds; thermal status 0.

## Sustained S26 checkpoint: four steps

Forge request quartz-eight-layer-four-step-s26-learning-20261028-01

| Step | Train CE | Original heldout CE |
|---:|---:|---:|
| 0 | 1.0487998 | 0.6603754 |
| 1 | 0.4235311 | 0.2987503 |
| 2 | 0.0516241 | 0.1719357 |
| 3 | 1.0714211 | 0.0026584 |
| 4 | 0.0111069 | 0.0062530 |

- Non-monotonic training loss at step 3: optimizer is not yet
  stable enough for unattended training at this learning rate.
- 98.539 seconds, battery 53% at both checkpoints; thermal status 0.
- Safetensors: 182,936 bytes; SHA-256:
  adfb838b23efefadf9423df0d3b6574920b378173c6ef026a31b5ec4a56c710d
- Independent new S26 process on the same two unseen prompts:
  base CE 1.1034586, adapted CE 0.0191992.
  Each individual example improved; 17.889 seconds, thermal 0.

## Stored artifacts

- Phone:
  Download/quartz-lora/adapter-qwen25-qv16-23-r2-s26.safetensors
  Download/quartz-lora/adapter-qwen25-qv16-23-r2-step4-s26.safetensors
- PC local, gitignored:
  experiments/android-lora-smoke/output/
  Both phone adapters transferred to the PC with exact SHA-256
  validation. The one-shot LAN receiver was closed.
- Source and original logs:
  src/train/multi*.rs
  experiments/android-lora-smoke/results/2026-10-28-multi*
- Isolated branch: feat/multilayer-qv-backprop. No production merge.

## What is NOT proved

- No original GGUF base weight was modified. This is LoRA on real
  internal transformer Q/V projections, not full-parameter training.
- No full-sequence teacher-forced supervision, no layers 0-15
  training, no persistent Adam optimizer-state checkpoints, no
  Vulkan backward, and no production adapter-aware inference.
- The tiny binary-label dataset makes the independent improvement
  meaningful as a pipeline check but NOT evidence of broad semantic
  learning, robust generalization or deployment readiness.

The next engineering gates are safer learning-rate scheduling and
validation-based best-checkpoint selection; then real sequence targets
and inference integration, while keeping thermal/battery safeguards.
