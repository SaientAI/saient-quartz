# Wan 2.1 scalar-reference baseline

This file pins the environment and artifacts used to establish the native Rust
Wan 2.1 correctness baseline. The baseline contains the scheduler, tokenizer,
UMT5 encoder, 3D RoPE, 30-block DiT, and stateful causal 3D VAE decoder.

## Quartz revision

- VAE correctness implementation: `a21e801f198f6f996db8ecd03535b772ddf346fd`
- Baseline tag: `wan-2.1-scalar-reference`
- Rust: `rustc 1.93.0 (254b59607 2026-01-19)`
- Cargo: `cargo 1.93.0 (083ac513 2025-12-15)`
- LLVM: `21.1.8`
- Host: Linux x86_64, kernel `7.0.0-28-generic`
- CPU: Intel Core i5-7400, four cores/four threads

## Reference engine

- Source: `stable-diffusion.cpp`
- Revision: `e31a86ce9110b11a98bd5990c329093244c2d1e3`
- Reported version: `master-805-e31a86c`
- Capture binary: `build-cuda128/bin/sd-cli`
- Capture binary SHA-256:
  `7b1b2183d02c22d968dd86fcc4903f487b6bc74e4c8842aec89783c3b09f6701`
- Capture hooks: local `SAIENT_DUMP` changes described by
  `reference-dump.patch`; the binary hash above identifies the exact resulting
  executable because the reference working tree contains those uncommitted
  hooks.
- C compiler: `/usr/bin/cc`
- C++ compiler: `/usr/bin/c++`, GCC `13.3.0`
- CUDA compiler: `/usr/local/cuda-12.8/bin/nvcc`, CUDA `12.8.93`
- CUDA build flags: `GGML_CUDA_FA=ON`,
  `GGML_CUDA_COMPRESSION_MODE=size`
- Reference GPU: NVIDIA GeForce RTX 5060 Ti, 16,311 MiB
- NVIDIA driver: `595.84`

The exact full-size reference capture command is recorded in
[`README.md`](README.md#capturing-a-reference-set).

## Model package

- Package ID: `wan2.1-t2v-1.3b-q4-v1`
- Package manifest SHA-256:
  `cc1d0f6ac7270e0be200a3fd68aa1a2f0636d3d09d5fbd27d3e80a774f1bf97d`
- Wan DiT Q4_K:
  `65181afff758fba25cf311399ccb0638a746f8e0e6533e07a84a5a23e0c12318`
- UMT5-XXL Q4_K_M:
  `17cf97a5bbbc60a646d6105b832b6f657ce904a8a1ad970e4b59df0c67584a40`
- Wan VAE BF16:
  `2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b`

## VAE captures

| Artifact | SHA-256 |
| --- | --- |
| `vae_in_small.bin` | `62ecd4384e6e3b081400daec9412f992e91018fec216de74783d1096febd80b1` |
| `vae_out_small.bin` | `25c11b76c501b6ce0edda2a24b5bc6a2846405768958a23c0f5786cdc0795d42` |
| `vae_in_full.bin` | `323ccd735929cdb160e1c11c5583223b6ea12aa223e7c81ae9b96d1a5adf3aa9` |
| `vae_out_full.bin` | `30a070aa52d0e398afdb5e38812349d5cade553f5e3f4b0bdc0b9e9fdf2d60ea` |

The SQD1-to-NCTHW shape mapping, decoder graph, and feature-cache rules are
recorded in [`WAN_VAE.md`](WAN_VAE.md).

## Baseline verification commands

Run from the Quartz repository root:

```bash
cargo test
cargo test --release wan_vae::parity::small_decode_matches_reference -- --ignored --nocapture
cargo test --release wan_vae::parity::full_decode_matches_reference -- --ignored --nocapture
git diff --check
```

The full VAE parity test is intentionally not part of routine iteration. Its
verified scalar runtime is approximately 61 minutes; rerun it when a change
can affect model semantics or the scalar backend. Use the small case for normal
backend parity development.

## Recorded VAE metrics

| Case | Input NCTHW | Output NCTHW | Cosine | Max abs | Mean abs |
| --- | --- | --- | ---: | ---: | ---: |
| Small | `[1,16,2,8,8]` | `[1,3,5,64,64]` | 0.999992503 | 0.0027391 | 0.0004328 |
| Full | `[1,16,2,30,52]` | `[1,3,5,240,416]` | 0.999999012 | 0.0043769 | 0.0003166 |
