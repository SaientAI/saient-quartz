# Quartz: reproducible Android GGUF LoRA training experiment

**Status:** experimental CPU implementation, observed on a Samsung Galaxy S26
(SM-S948B) on 28 October 2026. MIT-licensed source. Not integrated into the
production Saient app.

## The tested claim

Quartz ran a **real first-order LoRA training step inside eight connected
transformer blocks** of a **frozen, quantized Qwen2.5-0.5B-Instruct GGUF**
directly on Android ARM64. It did **not** retrain the base weights, invoke
a desktop training service, use Python/PyTorch on the phone, or run
eight disconnected approximation modules.

The forward pass uses the local model and its original GGUF quantized tensors.
The backward pass computes the causal dependencies across tokens and the
interconnected adapted layers, including frozen-weight transposes, GQA
softmax, residuals, RoPE, RMSNorm and SwiGLU. Only low-rank A/B matrices for
attention **query and value** projections are updated.

The tested suffix comprises layers **16–23** of a 24-layer model, rank 2,
**45,056 FP32 trainable LoRA parameters**. The original GGUF remains
byte-for-byte unchanged.

This is a proof of technical feasibility, **not a claim that smartphone
fine-tuning was previously impossible**, and not a demonstration of
full-parameter training or broadly useful personalisation.

## First-hand Samsung measurements

| Measurement | One optimisation step | Four optimisation steps |
|:--|--:|--:|
| Qwen model | Qwen2.5-0.5B Instruct | Same frozen GGUF |
| Adapted layers | 16–23 | 16–23 |
| Rank / optimiser | 2 / clipped Adam | 2 / clipped Adam |
| Training loss before | 1.048800 | 1.048800 |
| Training loss after | 0.423531 | 0.011107 |
| Small original held-out loss before | 0.660375 | 0.660375 |
| Original held-out loss after | 0.298750 | 0.006253 |
| Whole run (incl. prefix and verification) | 55.8 s | 98.5 s |
| Final adapter | 182,936 bytes | 182,936 bytes |
| Android thermal status | 0 → 0 | 0 → 0 |
| Battery reading | 54% → 54% | 53% → 53% |

These are **real device measurements**, not projected runtimes.
The tiny original training experiment had only **two training prompts
and one internal held-out prompt**. Training loss briefly increased
at step 3 of the four-step run; optimisation is not sufficiently
stable for unattended deployment.

On the same two unseen prompts, fresh independent evaluation of the
four-step checkpoint reported mean cross-entropy **1.103459 → 0.019199**.

### Adversarial verification (six additional fresh prompts)

A separate verification harness tested the four-step adapter and did not
hide the cases that became worse:

| Gate | Observed outcome |
|:--|:--|
| Model SHA-256 / frozen weight integrity | PASS |
| 32 safetensors tensors with correct dimensions and nonzero Q/V updates | PASS |
| Numerical gradients at both ends of the eight-layer stack | PASS |
| Six *unseen* sentiment prompts | 5/6 improved |
| Mean fresh-prompt cross-entropy | 0.607243 → 0.314402 (PC) |
| Inverted-label negative control | Mean loss increased as expected |
| Correct-vs-incorrect label preference | 4/6 improved — **warning** |
| Negation/composition challenge | 5/6 improved — **warning** |
| Wrong-base and truncated adapter rejection | PASS |
| Independent S26 evaluation of the same six prompts | 5/6 improved |
| PC ↔ S26 adapted-CE mean difference | 0.0000045 |
| Rust regression tests | 117 passed, 11 ignored |
| Model and adapter SHA-256 before/after | Unchanged |

**Result: PASS_WITH_WARNINGS.** The adapted model handled
"Review: not terrible" **worse** on the specified positive-label
target. Some unrelated completion targets also changed. Readers should
inspect the per-example logs instead of inferring general language
learning from these tiny datasets.

Two repeat CPU training runs were **byte-identical for a two-layer**
configuration. The initial full verification run tested the full
eight-layer adapter as its target but ran the retraining determinism
gate in a two-layer configuration; these are intentionally not
presented as proof of eight-layer repeat-training. An independent
eight-layer numerical-gradient test passed before this release.

Raw on-device training logs, native gradient comparisons and
the original result tables are retained in
[experiments/android-lora-smoke/results/](../experiments/android-lora-smoke/results/).
The adversarial harness, pinned fixtures and its self-tests are in
[verification/](../verification/).

## Rebuild the engine from source

Required on the build machine: Rust stable supporting edition 2024,
and for Android, Android NDK r27 or a compatible LLVM toolchain and
the ARM64 Rust target.

    git clone https://github.com/SaientAI/saient-quartz.git
    cd saient-quartz

    # CPU inference, trainer and test utilities
    cargo build --release
    cargo test --bin quartz
    python3 -m unittest discover -s verification -p 'test_*.py' -v

Android ARM64 build with the pre-existing script:

    rustup target add aarch64-linux-android
    export ANDROID_NDK_HOME=/path/to/android-ndk
    QUARTZ_ANDROID_FEATURES="" scripts/build-android-arm64.sh
    file target/aarch64-linux-android/release/quartz

**The Qwen GGUF is not bundled.** The measured experiment used an
Apache-2.0-licensed Qwen2.5-0.5B Instruct model in GGUF form, with
SHA-256:

    6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653

The frozen checkpoint and any quantised model with different bytes
are **not interchangeable**: a saved adapter is bound to the exact
GGUF hash. Do not rename a different model and pretend this is the
same reproducibility test. To reproduce independently with a different
GGUF conversion, train a *new* adapter with the real native trainer.

## Training on a Linux PC first

    # Supply your own compatible quantized GGUF; example uses Qwen2.5-0.5B.
    MODEL=/absolute/path/to/qwen05.gguf
    cargo build --release

    # Two-layer numerical gradient check (Q.A/Q.B/V.A/V.B in both blocks)
    target/release/quartz train-qv-suffix "$MODEL" \
      experiments/android-lora-smoke/multi-3-samples.jsonl \
      /tmp/quartz-two-layer-test.safetensors \
      --layers 2 --steps 1 --rank 2 --lr 0.01 \
      --max-tokens 16 --gradient-check

    # Actual eight interconnected transformer layers
    target/release/quartz train-qv-suffix "$MODEL" \
      experiments/android-lora-smoke/multi-3-samples.jsonl \
      /tmp/quartz-eight-layer-test.safetensors \
      --layers 8 --steps 4 --rank 2 --lr 0.01 --max-tokens 16

    # Run in an independent process with new prompts
    target/release/quartz eval-qv-suffix "$MODEL" \
      experiments/android-lora-smoke/multi-unseen-2.jsonl \
      /tmp/quartz-eight-layer-test.safetensors \
      --layers 8 --max-tokens 16

The regular model-loaded baseline is compared against **zero-B
LoRA inference**, and the trainer refuses to save if the frozen-base
hash changes or Q/V updates are nonfinite. It also checks
save/reload parity.

## Training on Android

Build the Android ARM64 binary on a desktop with the NDK, then
transfer it, the GGUF and the JSONL fixture to the phone. For example,
with Android Debug Bridge enabled on a phone you own:

    adb push target/aarch64-linux-android/release/quartz /data/local/tmp/quartz-trainer
    adb shell chmod 700 /data/local/tmp/quartz-trainer
    adb push /absolute/path/to/qwen05.gguf /sdcard/Download/qwen05.gguf
    adb push experiments/android-lora-smoke/multi-3-samples.jsonl /sdcard/Download/qv-training.jsonl

    adb shell 'RAYON_NUM_THREADS=2 /data/local/tmp/quartz-trainer train-qv-suffix /sdcard/Download/qwen05.gguf /sdcard/Download/qv-training.jsonl /sdcard/Download/qv-r2-eight.safetensors --layers 8 --steps 1 --rank 2 --lr 0.01 --max-tokens 16'

These are Android *shell* commands, not a consumer APK and not a
background service. No rooting or GPU is required by this experimental
CPU path. Memory headroom, execution permissions and supported GGUF
formats vary by device. Monitor thermal/battery status and do not run
unattended. Standard Java app sandbox constraints still apply if you
later integrate Quartz into an APK.

## Verification harness

The harness intentionally distinguishes **code correctness** from
**semantic generalisation**:

    python3 verification/verify.py --mode audit --model "$MODEL"
    python3 verification/verify.py --mode verify --model "$MODEL"
    python3 verification/verify.py --mode full --model "$MODEL" \
      --repeat-train --repeat-evaluation --timeout-seconds 220

For exact apples-to-apples reports the base model, adapter and all
four dataset fixtures are pinned in verification/manifest.json.
The complete verifier and mobile companion are described in
[verification/README.md](../verification/README.md).

The original 182,936-byte demonstration adapter is included at
verification/sample/adapter-qwen25-qv16-23-r2-step4-s26.safetensors,
but the 0.5B Qwen base GGUF is **NOT** included. Its SHA-256 must
exactly match the value above. A different GGUF version requires
a new adapter and a separately pinned experimental manifest.
The sample only contains LoRA matrices trained on the included
authored sentiment fixture — no personal messages or user files.

## Limitations and openness

- **Only Q/V LoRA**, not a full-weight Qwen retrain.
- **Last eight blocks only**, not all 24.
- **One supervised next-token label per short prompt**, not proper
  sequence-wide teacher-forced fine-tuning.
- **Tiny and strongly templated sentiment dataset**, no statistically
  convincing broad personalisation benchmark.
- **CPU-first experimental implementation**, not Vulkan or NPU
  accelerated training.
- **Adapter-aware evaluation CLI** works, but normal production chat
  inference does not yet dynamically load this LoRA format.
- A four-step run is not sufficient for claims about long-run
  thermal behaviour or energy efficiency.

If you find a numerical error, an unsupported quantisation issue, an
unexpected train/eval mismatch, or a stronger negative control,
please open an issue with the model hash, compiler/device details and
raw outputs. The goal is **falsifiable, reproducible work**, not a
marketing claim.

### Prior work (to avoid misleading first-ever claims)

Mobile fine-tuning was studied before Quartz:

- [PocketLLM (Peng et al., 2024)](https://aclanthology.org/2024.privatenlp-1.10/)
  demonstrated derivative-free on-device fine-tuning on OPPO hardware.
- [MobileFineTuner (Geng et al., 2025)](https://arxiv.org/abs/2512.08211)
  explored on-phone full-parameter and parameter-efficient training.
- [Fine-Tuning a 3B-Parameter LLM on a Smartphone (Geyko et al., 2026)](https://arxiv.org/abs/2610.06325)
  quantified sustained training on an iPhone.
- [First-Order LoRA Training with Phone NPUs (Zheng et al., 2026)](https://arxiv.org/abs/2610.24816)
  studied frozen-weight products on Snapdragon NPUs.

Quartz's contribution in this release is this **specific Rust,
GGUF-backed, connected-eight-block CPU training implementation and
open adversarial Android/desktop verification**. Novelty relative to
other systems is subject to independent assessment.

Code: MIT (root LICENSE).
The underlying Qwen2.5 base model: Apache-2.0; **not distributed** here.
