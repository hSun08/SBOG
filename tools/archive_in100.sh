#!/bin/bash
# Collects the final ImageNet-100 artifacts of one branch into <repo>/checkpoints/imagenet100/<branch>/ with sha256 sums.
# Usage: tools/archive_in100.sh normal|robust [--fortress]
set -euo pipefail
REPO=${SBOG_REPO:-$(cd "$(dirname "$0")/.." && pwd)}; OUT=${SBOG_OUT:-$REPO/outputs/imagenet100}; IN100=${IN100:-$REPO/data/imagenet100}
b=$1; shift || true; DST=$REPO/checkpoints/imagenet100/$b; mkdir -p "$DST/logs"
feat=id_feat_custom100_full.npy; [[ $b == robust ]] && feat=id_feat_custom100_full_ra2rw020.npy
cp -pv "$OUT/metadata/$feat" "$DST/" || true
# SOE files get short names: htar/tar omit members whose path exceeds 100 characters
for f in "$OUT/metadata/soe_$b"/outlier_energy_semantic_embed*.npy; do cp -pv "$f" "$DST/soe_latents.npy"; done
for f in "$OUT/metadata/soe_$b"/outlier_energy_semantic_embed*.json; do cp -pv "$f" "$DST/soe_latents.json"; done
pre=pretrain_normal; [[ $b == robust ]] && pre=pretrain_robust
cp -pv "$OUT/runs/$pre"/*/*.csv "$OUT/runs/$pre"/*/*_epoch_79.pt "$DST/" 2>/dev/null || true
cp -pv "$OUT/runs/vanilla_ce"/*.csv "$OUT/runs/vanilla_ce"/*_epoch_9.pt "$DST/" 2>/dev/null || true
cp -pv "$OUT/runs/detector_$b"/*/*_epoch_29*.pt "$OUT/runs/detector_$b"/*/*.csv "$DST/" || true
cp -pv "$OUT/bootstrap_eval/${b}_bootstrap.json" "$DST/" || true
cp -pv "$OUT/logs/i100_s"*"_$b"_*.out "$DST/logs/" 2>/dev/null || true
cp -pv "$IN100/metadata/class_order.txt" "$IN100/metadata/synset_to_text.json" "$IN100/metadata/token_embed_custom100.npy" "$DST/" 2>/dev/null || true
(cd "$DST" && find . -type f ! -name SHA256SUMS -exec sha256sum {} \; | sort -k2 > SHA256SUMS && wc -l SHA256SUMS); du -sh "$DST"
[[ "${1:-}" == "--fortress" ]] && htar -cvf "sbog_imagenet100_${b}_$(date +%F).tar" -C "$REPO/checkpoints/imagenet100" "$b" && echo "pushed to Fortress" || true
