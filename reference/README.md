# Wan2.1 reimplementation — reference parity harness

Quartz is being extended to run Wan2.1 text-to-video itself, replacing the
bundled `stable-diffusion.cpp` fork that currently does it.

A video diffusion model cannot be debugged by looking at output frames. A
wrong RoPE, a wrong AdaLN modulation and a wrong VAE scale factor all
produce indistinguishable garbage, and chasing that by eye is how you end
up guessing. So every stage is diffed numerically against the engine that
ships today, and no stage is called done until it matches.

## Capturing a reference set

`reference-dump.patch` (repo root) adds `SAIENT_DUMP` hooks to the pinned
`stable-diffusion.cpp` working tree. It is deliberately **not** part of
`saient-progress.patch`, so the engine binary shipped in the Android app
is unaffected by any of this.

```bash
mkdir -p /tmp/saient_ref
cd ~/projects/stable-diffusion.cpp
git apply /path/to/reference-dump.patch      # if not already applied
cmake --build build-cuda128 -j"$(nproc)" --target sd-cli

PACK=~/projects/saient/models/wan2.1-t2v-1.3b-mobile-pack
SAIENT_DUMP=1 ./build-cuda128/bin/sd-cli --mode vid_gen \
  --diffusion-model $PACK/wan2.1_t2v_1.3B_Q4_K.gguf \
  --vae $PACK/wan_2.1_vae.safetensors \
  --t5xxl $PACK/umt5-xxl-encoder-Q4_K_M.gguf \
  --prompt "a red fox" --negative-prompt "" \
  --cfg-scale 6.0 --sampling-method euler --steps 1 \
  --width 416 --height 240 --video-frames 5 --fps 8 \
  --flow-shift 8.0 --seed 12345 --diffusion-fa --output /tmp/ref.webm
```

Dump format `SQD1`: magic `"SQD1"`, `u32` ndim, `i64` dims, then f32 data.

## What is verified so far

| Stage | Status | Verified by |
|---|---|---|
| Flow-matching sigma schedule | **done** | `src/wan_scheduler.rs` tests, exact against dumped sigmas |
| T5 tokenizer | reference captured | `t5_ids_*.txt` below |
| UMT5-XXL encoder | reference captured | `cond_crossattn` fingerprint below |
| Wan DiT | not started | needs per-step latent dumps |
| 3D causal VAE | not started | needs pre-decode latent dump |

### Sigmas — matching (steps=8, shift=8)

```
1, 0.979615152, 0.952444434, 0.914422750,
0.857428312, 0.762538671, 0.573137879, 0.00794487074, 0
```

### Tokenizer

`t5_ids_a_red_fox.txt` — `"a red fox"` → `289, 4062, 273, 56209, 1` then
`0` padding to 512. `1` is EOS, `0` is PAD, no BOS. Matches the GGUF
metadata (`add_bos_token=false`, `add_eos_token=true`).

`t5_ids_empty.txt` — the empty prompt, which is what the unconditional
branch encodes.

### Encoder output

Shape `[4096, 512, 1]` — d_model 4096 by the full 512-token context, i.e.
padded, not truncated to the real token count.

| | cond (`"a red fox"`) | uncond (`""`) |
|---|---|---|
| sha256 (first 16B) | `0e0048d556a2ec0ab6ce5f924e43a7e0` | `882fa505298733c22cbd185b47873925` |
| min | -0.805374 | -0.812546 |
| max | 0.748705 | 0.990059 |
| mean | 0.000009 | 0.000001 |
| first 8 | 0.001735, -0.034512, 0.100163, 0.043965, -0.035091, 0.061932, 0.013535, 0.096772 | 0.002188, -0.041350, 0.019414, -0.009941, 0.053085, 0.003397, 0.020289, 0.005043 |

The full 8 MB blobs are not committed — regenerate them with the command
above when needed.

## UMT5-XXL architecture (from the GGUF metadata)

| | |
|---|---|
| architecture | `t5encoder`, 24 blocks |
| d_model | 4096 |
| d_ff | 10240 (gated) |
| heads | 64 × 64 head dim |
| norm | RMSNorm, ε = 1e-6 |
| relative buckets | 32 |
| context | 512 |
| vocab | 256,384 |

Per block: `attn_norm`, `attn_q/k/v/o` (4096²), **`attn_rel_b` [64, 32]**,
`ffn_norm`, `ffn_gate`/`ffn_up` (4096×10240), `ffn_down`.
Plus `token_embd.weight [4096, 256384]` and `enc.output_norm.weight`.

**This is UMT5, not standard T5.** `attn_rel_b` is present in all 24
blocks; standard T5 computes the relative-position bias once in layer 0
and shares it. Implementing the shared-bias variant would produce subtly
wrong embeddings everywhere.

## Non-negotiable memory constraint

`token_embd.weight` is 256,384 × 4096 — 1.05 B parameters. Dequantised to
f32 that is **4.2 GB**, and materialising it is precisely the bug that
OOM-killed the Android app today: the shipped ggml only listed `Q2_K` in
`support_get_rows`, so `Q4_K`/`Q6_K` fell through to a full dequant of
this matrix and cost 2.45 GB of peak RSS.

The Quartz implementation must read embedding rows **directly from the
quantised weights** from the first commit. Do not "make it work first and
optimise later" here — that path ends at an OOM on an 8 GB phone.

## Suggested order

1. ~~Scheduler~~ — done.
2. **T5 SentencePiece unigram tokenizer** — self-contained, exact-match
   testable against `t5_ids_*.txt`, no model execution needed.
3. **UMT5 encoder** — clears the whole conditioning half, which is where
   the ghost-subject bug lived. Reuses GGUF loading, dequant and the
   attention/RMSNorm machinery from the LLM path.
4. **Wan DiT** — largest piece; diff block-by-block against per-step
   latents (needs a new dump hook in the sampler loop).
5. **3D causal VAE** — most new kernel work (conv3d, temporal causality),
   but the only stage whose output you can actually eyeball.
