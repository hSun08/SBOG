#!/bin/bash
# Cross-entropy-pretrained ResNet-34 used to initialise the ImageNet-100 outlier-exposure fine-tuning (step 4).
# This is the "pretrained model for initialization, trained using the cross-entropy loss" released by DreamOOD
# (https://github.com/deeplearning-wisc/dream-ood, Google Drive id 1x3QpmPuW3u6Gh61l1AdE_8l7-8ibUhsO, 85,457,229 bytes).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); mkdir -p "$REPO/models/dreamood_release"; cd "$REPO/models/dreamood_release"
python -m pip install -q gdown
gdown "https://drive.google.com/uc?id=1x3QpmPuW3u6Gh61l1AdE_8l7-8ibUhsO" -O in100_initialized_model_for_ood_det.pt
echo "eeff13d6043cfae227602aea9da8a1cf10cceaa2ee933ce9e8e2a31e50a075e0  in100_initialized_model_for_ood_det.pt" | sha256sum -c
