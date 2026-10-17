#!/bin/bash
# Fails fast if the env, ImageNet-100 data, OOD test sets or SD checkpoint are missing.
REPO=${SBOG_REPO:-$(cd "$(dirname "$0")/.." && pwd)}; D=$REPO/data; ok=1
chk () { if eval "$2"; then echo "ok    $1"; else echo "MISSING $1"; ok=0; fi; }
chk "python env with torch"             "python -c 'import torch, torchvision, transformers, omegaconf, pytorch_lightning, kornia, einops, clip, xformers, taming' 2>/dev/null"
chk "models/sd-v1-4.ckpt (4265380512 bytes)" "[[ \$(stat -c %s $REPO/models/sd-v1-4.ckpt 2>/dev/null) == 4265380512 ]]"
chk "imagenet100/train (117000, 100 classes)" "[[ \$(find $D/imagenet100/train -type f 2>/dev/null | wc -l) -ge 117000 && \$(ls $D/imagenet100/train 2>/dev/null | wc -l) == 100 ]]"
chk "imagenet100/val (13000)"        "[[ \$(find $D/imagenet100/val -type f 2>/dev/null | wc -l) -ge 13000 ]]"
chk "imagenet100/metadata/token_embed_custom100.npy" "[[ -s $D/imagenet100/metadata/token_embed_custom100.npy && -s $D/imagenet100/metadata/class_order.txt ]]"
chk "iNaturalist (10000)"            "[[ \$(find -L $D/imagenet_ood/iNaturalist_imagefolder/ood -type f 2>/dev/null | wc -l) -ge 10000 ]]"
chk "SUN (10000)"                    "[[ \$(find -L $D/imagenet_ood/SUN_imagefolder/ood -type f 2>/dev/null | wc -l) -ge 10000 ]]"
chk "Places (10000)"                 "[[ \$(find -L $D/imagenet_ood/Places_imagefolder/ood -type f 2>/dev/null | wc -l) -ge 10000 ]]"
chk "DTD images (5640)"              "[[ \$(find $D/dtd/images -type f 2>/dev/null | wc -l) -ge 5640 ]]"
[[ $ok == 1 ]] && echo "PREFLIGHT OK" || { echo "PREFLIGHT FAILED"; exit 1; }
