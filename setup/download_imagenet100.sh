#!/bin/bash
# ImageNet-100 ID data + ImageNet OOD test sets, in the layout the ImageNet-100 scripts expect:
#   data/imagenet100_hf/                 raw parquet snapshot of HF dataset ilee0022/ImageNet100 (= Kaggle ambityga/imagenet100, 100 classes)
#   data/imagenet100/{train,val,test}/<synset>/*.JPEG   ImageFolder export (train 117,000 / val 13,000 / test 5,000)
#   data/imagenet100/metadata/           class_order.txt, synset_to_text.json, token_embed_custom100.npy (CLIP text anchors)
#   data/imagenet_ood/{iNaturalist,SUN,Places}/images/*  (MOS release, 10,000 images each) and *_imagefolder/ood -> images
#   data/dtd/images                      shared with the CIFAR benchmark
# Stage 1 (--download-only) needs only wget + huggingface_hub; stage 2 (export + metadata + token embeddings) needs the conda env.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA_DIR:-$REPO/data}; ARCH=$DATA/_downloads; mkdir -p "$ARCH" "$DATA/imagenet_ood"
if [[ "${1:-}" != "--export-only" ]]; then
  for d in iNaturalist SUN Places; do
    echo "[$d]"; wget -nv -c -O "$ARCH/$d.tar.gz" "http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/$d.tar.gz"
    tar -xzf "$ARCH/$d.tar.gz" -C "$DATA/imagenet_ood"
    mkdir -p "$DATA/imagenet_ood/${d}_imagefolder" && ln -sfn "../$d/images" "$DATA/imagenet_ood/${d}_imagefolder/ood"
    echo "$(find "$DATA/imagenet_ood/$d/images" -type f | wc -l) files in $d"
  done
  echo "[ilee0022/ImageNet100 parquet, ~17GB]"
  python - <<'PY'
from huggingface_hub import snapshot_download
import os
p = snapshot_download(repo_id='ilee0022/ImageNet100', repo_type='dataset',
                      local_dir=os.environ.get('DATA_DIR', '') + '/imagenet100_hf' if os.environ.get('DATA_DIR') else None)
print("snapshot at", p)
PY
fi
[[ "${1:-}" == "--download-only" ]] && { echo "download stage done"; exit 0; }
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"; conda activate "${SBOG_ENV:-$REPO/env}"
export PYTHONPATH="$REPO:$REPO/sbog"
python "$REPO/setup/export_hf_parquet_to_imagefolder.py" --dataset_root "$DATA/imagenet100_hf" --output_root "$DATA/imagenet100" --splits train validation test
python "$REPO/setup/prepare_imagenet100_metadata.py" --image_root "$DATA/imagenet100" --output_dir "$DATA/imagenet100/metadata" --manifest_path "$DATA/imagenet100/label_manifest.json"
HF_HOME="$REPO/cache/hf" python "$REPO/setup/generate_token_embed.py" --metadata_dir "$DATA/imagenet100/metadata" --output_path "$DATA/imagenet100/metadata/token_embed_custom100.npy"
for s in train val test; do echo "$(find "$DATA/imagenet100/$s" -type f | wc -l) files in $s"; done
echo DONE
