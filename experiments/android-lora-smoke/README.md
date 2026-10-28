# Samsung S26 Ultra: on-device LoRA smoke test (2026-10-28)

Reproducible code from Forge phone project gpt_quartz_lora_smoke.
Package: uk.co.saient.generated.gpt_quartz_lora_smoke
Java source SHA-256: 1b8d110b7671c5ef6e5890f27c3b9c3dbb572ce27aef00373a0071e353e269a7
APK SHA-256: 67a8b5d3fdd283021d6576a84e5859a16a076819d276d70b73b5ff69cc5ab2d5
Build: on-phone Forge ECJ + D8; signed APK, zero Android permissions.
Installed and launched on device SM-S948B (Android 17, Snapdragon SM8850).
No inference server, external API, network permission, or training cloud.

Experiment:
- Frozen synthetic 6x8 base layer (NOT a GGUF model).
- Rank-2 LoRA, scaling alpha/r = 2.
- Random nonzero A; B initialized exactly zero.
- Analytical adapter/input derivatives checked against central differences
  with both A and B nonzero.
- Adam optimizer updated ONLY A/B on 96 synthetic training examples.
- 48 unseen synthetic holdout examples tested before and after.
- Trained adapter saved to private Android app storage.

Phone runtime log:
  RESULT: PASS
  Gradient finite-difference max relative error: 2.3e-09
  Base weights unchanged: YES
  Held-out loss: 1.054733 -> 0.000000000 (display rounding)
  Training loss: 0.000000000 (display rounding)
  Optimizer steps: 600
  Duration: 44 ms
  Android thermal status: 0 (normal)

Limits:
- This is real on-device LoRA math/training, but of a tiny synthetic
  linear layer using Java CPU arithmetic; NO transformer fine-tuning.
- The duration is *not* a throughput forecast for 0.6B–2B models.
- No Vulkan compute, no native Quartz, no GGUF, no safetensors export.
- The main Rust Stage 0 tests live in src/train; connecting them to
  GGUF projection kernels is the next substantial engineering step.
