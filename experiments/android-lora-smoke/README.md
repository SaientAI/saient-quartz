# Samsung S26 Ultra: Quartz LoRA experiments (2026-10-28)

## 1. Initial native Java/Android training arithmetic smoke test

Forge phone project gpt_quartz_lora_smoke, package:
uk.co.saient.generated.gpt_quartz_lora_smoke
Java source SHA-256:
1b8d110b7671c5ef6e5890f27c3b9c3dbb572ce27aef00373a0071e353e269a7
APK SHA-256:
67a8b5d3fdd283021d6576a84e5859a16a076819d276d70b73b5ff69cc5ab2d5

On the Samsung (Android 17, Qualcomm SM8850) the synthetic 6x8
frozen linear layer trained a rank-2 LoRA for 600 Adam updates:
- Finite-difference gradient max relative error: 2.3e-9
- Held-out synthetic loss: 1.054733 -> ~0.0 (display rounding)
- Frozen base unchanged; adapter saved privately
- 44 milliseconds. This does NOT estimate real model training throughput.

## 2. Real GGUF head-only training

Dataset: real-qwen-sentiment.jsonl (10 train + 4 held-out)
Model already available from user's own Saient desktop installation:
Qwen2.5-0.5B Instruct, architecture qwen2, GGUF v3, Apache-2.0.
24 layers, hidden 896, vocab 151936; file 397,808,192 bytes.
Base model SHA-256:
6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653

Model and dataset transferred to /sdcard/Download/quartz-lora/
on the S26 over the local network. Hashes verified on phone.
The ARM64 quartz executable was cross-compiled on the PC and SHA-256
verified on phone; native code runs as Android shell uid=2000, not the
Java Forge APK. Model tokenizer verified token ' positive' => ID 6785.

Run:
  /data/local/tmp/quartz-lora/quartz train-head \
    /sdcard/Download/quartz-lora/qwen05.gguf \
    /sdcard/Download/quartz-lora/real-qwen-sentiment.jsonl \
    /sdcard/Download/quartz-lora/adapter-qwen25-r4-stage1.safetensors \
    --steps 80 --rank 4 --lr 0.015 --max-tokens 16

The output-head LoRA trains on real frozen Qwen hidden states/logits but
is not an attention/transformer Q/V LoRA. Refer to src/train/README.md
for limitations. Phone results must be recorded before calling Stage 1
accepted; successful compilation alone is not sufficient.

## 3. Actual S26 outcome — verified

The full shell output is archived in results/ (two raw logs).

- REAL GGUF training (rank 4, 80 steps, 10 train/4 held-out):
  train loss 0.423226 -> <0.000001 (rounded output)
  held-out loss 0.695651 -> <0.000001 (rounded output)
  14 real-model forward passes took 94,552 ms;
  80 adapter steps were approximately 4,220 ms;
  full native training process 102,728 ms;
  battery 86% -> 85%; Android thermal status 0.
- The frozen Qwen GGUF SHA-256 matched before and after.
  Adapter save/reload numerical parity passed.
- Portable safetensors output size 2,445,728 bytes (2.33 MiB);
  SHA-256 8a76650f2c55374f529f277566b837c03dcd96984563c045708b2e6c3cf14bf0.
- Independent second native process, no training: six previously
  unobserved held-out prompts, base CE 1.079906 -> <0.000001
  (rounded adapter display). Each of six individually improved.
  Eval took 51,703 ms, thermal status 0.
- Tested phone: SM-S948B (Snapdragon SM8850, Android 17).
- Android binary 1 SHA-256:
  caa2bc2884aa053ac59c36b5d8c0de70c3765ff3f95c646c9927c5775497379a
- Android binary 2 (new independent eval command) SHA-256:
  b8e2c3ed31d1eebc839ad985eefe0687e0d39c8440393beb9b886156ea6db180

The model, trained adapter and binaries are on the S26. The exact
safetensors adapter was also uploaded back over the private LAN and
SHA-256 verified under output/ on the PC (gitignored, not published).
The temporary transfer server was shut down after verification.

**Interpretation:** this is an important implementation milestone
for real-model on-device output-head adaptation. The tiny train and
held-out sets are restricted to two sentiment labels. Lower loss on
six fresh examples DOES NOT imply broad transfer, long-sequence SFT,
or attention/Q/V LoRA capability. No full transformer backward,
checkpoint/resume, or production lifecycle guarantee is claimed.


## 4. Q/V last-block S26 training (separate from head-only experiment)

The native ARM64 Quartz CPU binary trained Q/V adapters on layer 23 of
the real Qwen2.5-0.5B GGUF for eight steps (rank 2, alpha 4).
Training CE 0.9410000 -> 0.0264473; two-example heldout CE
0.3764815 -> 0.0534430. Independent reload in a new phone
process on six fresh examples: 1.0799055 -> 0.0712443.
Total training 87.355s, Android thermal 0 at both checkpoints,
57% battery indicated at both checkpoints. Saved Q/V adapter:
  Download/quartz-lora/adapter-qwen25-qv23-r2-s26.safetensors
Base GGUF unchanged. See the raw training/eval logs and
2026-10-28-qv-last-technical-report.md under results/.
The production inference CLI does not yet accept these Q/V adapters.


## 5. Multi-block Q/V training on the Samsung

The verified eight-block suffix runs across real Qwen2.5-0.5B
transformer layers 16..23 with rank-2 Q and V LoRA in EACH block.
Full reverse-mode gradients propagate through the eight interconnected
blocks and earlier causal-token key/value dependencies. Original
GGUF base remains immutable. Original inference parity and actual
Q/V parameter updates pass. See the new source modules in src/train/
and evidence logs in results/2026-10-28-multi8*.
