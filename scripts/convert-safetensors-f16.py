#!/usr/bin/env python3
"""Convert a safetensors file's floating tensors to F16 atomically."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    if not args.source.is_file():
        raise SystemExit(f"source is not a file: {args.source}")
    if args.destination.exists():
        raise SystemExit(f"destination already exists: {args.destination}")

    with safe_open(args.source, framework="pt", device="cpu") as source:
        metadata = source.metadata()
    tensors = load_file(args.source, device="cpu")
    converted = {
        name: tensor.to(dtype=torch.float16) if tensor.is_floating_point() else tensor
        for name, tensor in tensors.items()
    }
    non_f16 = [name for name, tensor in converted.items() if tensor.dtype != torch.float16]
    if non_f16:
        raise SystemExit(f"non-F16 tensors remain: {', '.join(non_f16[:10])}")

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    partial = args.destination.with_name(f".{args.destination.name}.part")
    save_file(converted, partial, metadata=metadata)

    with safe_open(partial, framework="pt", device="cpu") as output:
        output_keys = list(output.keys())
        output_non_f16 = [name for name in output_keys if output.get_tensor(name).dtype != torch.float16]
    if output_non_f16 or set(output_keys) != set(converted):
        raise SystemExit("converted safetensors verification failed")
    os.replace(partial, args.destination)
    print(f"Converted {len(converted)} tensors to F16: {args.destination}")


if __name__ == "__main__":
    main()
