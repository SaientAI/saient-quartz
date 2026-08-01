#!/usr/bin/env bash
set -euo pipefail

api="${QUARTZ_ANDROID_API:-24}"
sdk="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-/usr/lib/android-sdk}}"
ndk="${ANDROID_NDK_HOME:-}"

if [[ -z "$ndk" ]]; then
  if [[ ! -d "$sdk/ndk" ]]; then
    echo "Android NDK not found. Set ANDROID_NDK_HOME or ANDROID_SDK_ROOT." >&2
    exit 2
  fi
  ndk="$(find "$sdk/ndk" -mindepth 1 -maxdepth 1 -type d -print | sort -V | tail -n 1)"
fi

toolchain="$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin"
linker="$toolchain/aarch64-linux-android${api}-clang"
archiver="$toolchain/llvm-ar"
if [[ ! -x "$linker" || ! -x "$archiver" ]]; then
  echo "Android ARM64 toolchain is incomplete under $toolchain" >&2
  exit 2
fi

export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$linker"
export CC_aarch64_linux_android="$linker"
export AR_aarch64_linux_android="$archiver"
export ANDROID_NDK_HOME="$ndk"

features="${QUARTZ_ANDROID_FEATURES:-vulkan}"
cargo build --release --target aarch64-linux-android --features "$features" "$@"
echo "Built target/aarch64-linux-android/release/quartz"
