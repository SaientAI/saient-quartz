#!/usr/bin/env python3
"""
Generate tiny_model.gguf — a minimal valid GGUF file with F32 weights.

Config: vocab=64, embed=32, n_heads=4, n_kv_heads=2, layers=1, ffn=128
This matches the default main.rs Config::from_gguf output for this file.
"""

import struct
import numpy as np

VOCAB      = 64
EMBED      = 32
N_HEADS    = 4
N_KV_HEADS = 2      # GQA: 2 kv heads for 4 query heads
HEAD_DIM   = EMBED // N_HEADS   # 8
N_LAYERS   = 1
FFN_DIM    = 128
SEED       = 42
GGUF_MAGIC = 0x46554747

rng = np.random.default_rng(SEED)

# ── helpers ───────────────────────────────────────────────────────────────────

def pack_str(s: str) -> bytes:
    """GGUF string: u64 length + utf-8 bytes."""
    b = s.encode()
    return struct.pack('<Q', len(b)) + b

def pack_kv_str(key: str, val: str) -> bytes:
    return pack_str(key) + struct.pack('<I', 8) + pack_str(val)

def pack_kv_u32(key: str, val: int) -> bytes:
    return pack_str(key) + struct.pack('<II', 4, val)

def pack_tensor_info(name: str, arr: np.ndarray, offset: int) -> bytes:
    """Tensor info entry (ggml_type=0 for F32)."""
    b  = pack_str(name)
    b += struct.pack('<I', arr.ndim)
    for s in arr.shape[::-1]:   # GGUF: dims[0] = fastest-varying (last numpy axis)
        b += struct.pack('<Q', s)
    b += struct.pack('<I', 0)   # ggml_type F32
    b += struct.pack('<Q', offset)
    return b

# ── build tensor dict (GGUF column-major: shape is [out_dim, in_dim]) ─────────
#
# GGUF convention: ne[0] = in_features, ne[1] = out_features
# In numpy (row-major): we store as shape (out_features, in_features)
# so that arr.flatten() gives the correct byte order.
# Embedding is stored as (vocab, embed) → ne[0]=embed, ne[1]=vocab.

tensors = {}

tensors['token_embd.weight'] = rng.standard_normal((VOCAB, EMBED)).astype(np.float32) * 0.02

for l in range(N_LAYERS):
    tensors[f'blk.{l}.attn_norm.weight']    = np.ones(EMBED, dtype=np.float32)
    # weights: (out_dim, in_dim) → ne[0]=in_dim (EMBED), ne[1]=out_dim
    tensors[f'blk.{l}.attn_q.weight']       = rng.standard_normal((N_HEADS*HEAD_DIM, EMBED)).astype(np.float32) * 0.02
    tensors[f'blk.{l}.attn_k.weight']       = rng.standard_normal((N_KV_HEADS*HEAD_DIM, EMBED)).astype(np.float32) * 0.02
    tensors[f'blk.{l}.attn_v.weight']       = rng.standard_normal((N_KV_HEADS*HEAD_DIM, EMBED)).astype(np.float32) * 0.02
    tensors[f'blk.{l}.attn_output.weight']  = rng.standard_normal((EMBED, EMBED)).astype(np.float32) * 0.02
    tensors[f'blk.{l}.ffn_norm.weight']     = np.ones(EMBED, dtype=np.float32)
    tensors[f'blk.{l}.ffn_gate.weight']     = rng.standard_normal((FFN_DIM, EMBED)).astype(np.float32) * 0.02
    tensors[f'blk.{l}.ffn_up.weight']       = rng.standard_normal((FFN_DIM, EMBED)).astype(np.float32) * 0.02
    tensors[f'blk.{l}.ffn_down.weight']     = rng.standard_normal((EMBED, FFN_DIM)).astype(np.float32) * 0.02

tensors['output_norm.weight'] = np.ones(EMBED, dtype=np.float32)
tensors['output.weight']      = rng.standard_normal((VOCAB, EMBED)).astype(np.float32) * 0.02

# ── build metadata KV pairs ───────────────────────────────────────────────────

kv_bytes = b''
kv_bytes += pack_kv_str('general.architecture',    'llama')
kv_bytes += pack_kv_str('general.name',            'quartz-test')
kv_bytes += pack_kv_u32('general.file_type',       0)   # F32
kv_bytes += pack_kv_u32('llama.embedding_length',  EMBED)
kv_bytes += pack_kv_u32('llama.head_count',        N_HEADS)
kv_bytes += pack_kv_u32('llama.head_count_kv',     N_KV_HEADS)
kv_bytes += pack_kv_u32('llama.block_count',       N_LAYERS)
kv_bytes += pack_kv_u32('llama.feed_forward_length', FFN_DIM)
kv_bytes += pack_kv_u32('llama.vocab_size',        VOCAB)
kv_bytes += pack_kv_u32('general.alignment',       32)

n_kv = 10

# ── compute tensor data offsets (each tensor aligned to 32 bytes) ─────────────

tensor_data_parts = []
cur_offset = 0
tensor_offsets = []

for arr in tensors.values():
    tensor_offsets.append(cur_offset)
    data = arr.flatten().astype(np.float32).tobytes()
    tensor_data_parts.append(data)
    # align next tensor to 32 bytes
    aligned = (len(data) + 31) & ~31
    cur_offset += aligned

# ── tensor info section ───────────────────────────────────────────────────────

info_bytes = b''
for (name, arr), offset in zip(tensors.items(), tensor_offsets):
    info_bytes += pack_tensor_info(name, arr, offset)

# ── build header to find data_offset ─────────────────────────────────────────

header  = struct.pack('<I', GGUF_MAGIC)
header += struct.pack('<I', 3)                      # version 3
header += struct.pack('<Q', len(tensors))           # tensor_count
header += struct.pack('<Q', n_kv)                   # kv_count
header += kv_bytes
header += info_bytes

# align header to 32 bytes
pad = (32 - len(header) % 32) % 32
header += b'\x00' * pad

# ── write file ────────────────────────────────────────────────────────────────

out_path = 'tiny_model.gguf'
with open(out_path, 'wb') as f:
    f.write(header)
    for data in tensor_data_parts:
        f.write(data)
        # pad to next 32-byte boundary
        pad = (32 - len(data) % 32) % 32
        if pad: f.write(b'\x00' * pad)

import os
size = os.path.getsize(out_path)
print(f"Wrote {out_path}  ({size//1024} KB, {len(tensors)} tensors)")
for name, arr in tensors.items():
    print(f"  {name:50s}  {arr.shape}")
