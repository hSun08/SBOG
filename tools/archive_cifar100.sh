#!/bin/bash
# Collects every artifact needed to re-evaluate / audit a CIFAR-100 branch into <repo>/checkpoints/cifar100/<branch>/
# (feature bank, SOE latents, final detector checkpoint + bundle, bootstrap JSON, training CSVs, SLURM logs) and writes sha256 sums.
# Usage: tools/archive_cifar100.sh normal|robust [--fortress]   (--fortress also pushes a tar to Fortress via htar)
set -euo pipefail
REPO=${SBOG_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
OUT=${SBOG_OUT:-$REPO/outputs/cifar100}
b=$1; shift || true
DST=$REPO/checkpoints/cifar100/$b; mkdir -p "$DST"
feat=id_feat_cifar100_199epoch.npy; [[ $b == robust ]] && feat=id_feat_cifar100_ra2rw010.npy
cp -pv "$OUT/metadata/$feat" "$DST/" || true
# SOE files get short names: htar/tar omit members whose path exceeds 100 characters
for f in "$OUT/metadata/soe_$b"/outlier_energy_semantic_embed*.npy; do cp -pv "$f" "$DST/soe_latents.npy"; done
for f in "$OUT/metadata/soe_$b"/outlier_energy_semantic_embed*.json; do cp -pv "$f" "$DST/soe_latents.json"; done
cp -pv "$OUT/runs/pretrain_"$([[ $b == robust ]] && echo robust || echo normal)/*/*.csv "$DST/" 2>/dev/null || true
cp -pv "$OUT/runs/pretrain_"$([[ $b == robust ]] && echo robust || echo normal)/*/*_epoch_*.pt "$DST/" 2>/dev/null || true
cp -pv "$OUT/runs/detector_$b/energy_ft_sd/"*_epoch_199*.pt "$DST/" || true
cp -pv "$OUT/runs/detector_$b/energy_ft_sd/"*.csv "$DST/" || true
cp -pv "$OUT/bootstrap_eval/${b}_bootstrap.json" "$DST/" || true
mkdir -p "$DST/logs" && cp -pv "$OUT/logs/c100_s"*"_$b"_*.out "$DST/logs/" 2>/dev/null || true
(cd "$DST" && find . -type f ! -name SHA256SUMS -exec sha256sum {} \; | sort -k2 > SHA256SUMS && cat SHA256SUMS)
du -sh "$DST"
if [[ "${1:-}" == "--fortress" ]]; then
  htar -cvf "sbog_cifar100_${b}_$(date +%F).tar" -C "$REPO/checkpoints/cifar100" "$b" && echo "pushed to Fortress"
fi
