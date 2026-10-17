#!/bin/bash
#SBATCH -J c100_s3_robust
#SBATCH -p a100-40gb       # <- edit: a partition with 40 GB+ GPUs
#SBATCH -A liu334          # <- edit: your SLURM account
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 4-00:00:00
#SBATCH -o outputs/cifar100/logs/%x_%j.out
#SBATCH -e outputs/cifar100/logs/%x_%j.err
# Submit from the repo root (submit_*.sh does this): common.sh is located via SLURM_SUBMIT_DIR, or set SBOG_REPO.
set -euo pipefail
source "${SBOG_REPO:-$SLURM_SUBMIT_DIR}/slurm/cifar100/common.sh"
EMB=$(find "$OUT/metadata/soe_robust" -maxdepth 1 -name 'outlier_energy_semantic_embed*.npy' | head -n 1)
[[ -f "$EMB" ]] || { echo "missing step2 embedding in $OUT/metadata/soe_robust"; exit 1; }
echo "embedding: $EMB"
run_step3 "$EMB" "$OUT/generated/cifar100_soe_robust"
echo "generated images: $(find "$OUT/generated/cifar100_soe_robust/samples" -type f | wc -l)"
