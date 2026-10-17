#!/bin/bash
# Fails fast (non-zero exit) if the env, data or SD checkpoint are missing, so dependent SLURM jobs never start half-configured.
REPO=${SBOG_REPO:-$(cd "$(dirname "$0")/.." && pwd)}; D=$REPO/data; ok=1
chk () { if eval "$2"; then echo "ok    $1"; else echo "MISSING $1"; ok=0; fi; }
chk "python env with torch"             "python -c 'import torch, torchvision, transformers, omegaconf, pytorch_lightning, kornia, einops, clip' 2>/dev/null"
chk "models/sd-v1-4.ckpt (4265380512 bytes)" "[[ \$(stat -c %s $REPO/models/sd-v1-4.ckpt 2>/dev/null) == 4265380512 ]]"
chk "assets/token_embed_c100.npy"    "[[ -s $REPO/assets/token_embed_c100.npy ]]"
chk "CIFAR-100 train/test/meta"      "[[ -s $D/cifarpy/cifar-100-python/train && -s $D/cifarpy/cifar-100-python/test && -s $D/cifarpy/cifar-100-python/meta ]]"
chk "SVHN test_32x32.mat"            "[[ -s $D/svhn/test_32x32.mat ]]"
chk "DTD images (5640)"              "[[ \$(find $D/dtd/images -type f 2>/dev/null | wc -l) -ge 5640 ]]"
chk "LSUN_C (10000)"                 "[[ \$(find $D/LSUN_C/unknown -type f 2>/dev/null | wc -l) -ge 10000 ]]"
chk "iSUN (8925)"                    "[[ \$(find $D/iSUN/unknown -type f 2>/dev/null | wc -l) -ge 8925 ]]"
chk "Places365 test (328500)"        "[[ \$(find $D/places365_test/unknown -type f 2>/dev/null | wc -l) -ge 328500 ]]"
[[ $ok == 1 ]] && echo "PREFLIGHT OK" || { echo "PREFLIGHT FAILED"; exit 1; }
