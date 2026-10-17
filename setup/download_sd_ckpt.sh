#!/bin/bash
# Stable Diffusion v1.4 original checkpoint (CompVis/stable-diffusion-v-1-4-original, sd-v1-4.ckpt, ~4GB) -> <repo>/models/
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$REPO/models"
TOK=${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}
curl -L -C - --retry 5 ${TOK:+-H "Authorization: Bearer $TOK"} -o "$REPO/models/sd-v1-4.ckpt" \
  https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
ls -la "$REPO/models/sd-v1-4.ckpt"; sha256sum "$REPO/models/sd-v1-4.ckpt"
