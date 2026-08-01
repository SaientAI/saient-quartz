#!/usr/bin/env bash
set -euo pipefail

destination="${1:?usage: fetch-sdxl-pack.sh DESTINATION}"
base_revision="462165984030d82259a11f4367a4eed129e94a7b"
vae_revision="207b116dae70ace3637169f1ddd2434b91b3a8cd"
base_url="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/${base_revision}"
vae_url="https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/${vae_revision}"

download() {
  local url="$1"
  local output="$2"
  local expected_bytes="$3"
  mkdir -p "$(dirname "$output")"
  if [[ -f "$output" ]]; then
    local current_bytes
    current_bytes="$(stat -c %s "$output")"
    if [[ "$current_bytes" == "$expected_bytes" ]]; then
      echo "Already downloaded: $output ($current_bytes bytes)"
      return
    fi
    if (( current_bytes > expected_bytes )); then
      echo "Refusing oversized partial download: $output ($current_bytes > $expected_bytes)" >&2
      exit 2
    fi
  fi
  curl \
    --fail \
    --location \
    --continue-at - \
    --retry 8 \
    --retry-all-errors \
    --retry-delay 2 \
    --output "$output" \
    "$url"
  local downloaded_bytes
  downloaded_bytes="$(stat -c %s "$output")"
  if [[ "$downloaded_bytes" != "$expected_bytes" ]]; then
    echo "Wrong size for $output: $downloaded_bytes != $expected_bytes" >&2
    exit 2
  fi
}

while read -r path expected_bytes
do
  download "${base_url}/${path}" "${destination}/${path}" "$expected_bytes"
done <<'FILES'
model_index.json 609
scheduler/scheduler_config.json 479
text_encoder/config.json 565
text_encoder/model.fp16.safetensors 246144152
text_encoder_2/config.json 575
text_encoder_2/model.fp16.safetensors 1389382176
tokenizer/merges.txt 524619
tokenizer/vocab.json 1059962
tokenizer_2/merges.txt 524619
tokenizer_2/vocab.json 1059962
unet/config.json 1680
unet/diffusion_pytorch_model.fp16.safetensors 5135149760
FILES

download "${base_url}/LICENSE.md" "${destination}/LICENSE-SDXL.md" 14109
download "${base_url}/README.md" "${destination}/README-SDXL.md" 8668
download "${vae_url}/config.json" "${destination}/vae/config.json" 631
download "${vae_url}/README.md" "${destination}/README-VAE.md" 3143
download \
  "${vae_url}/diffusion_pytorch_model.safetensors" \
  "${destination}/vae/diffusion_pytorch_model.f32.safetensors" \
  334643238

echo "SDXL source files downloaded to ${destination}"
