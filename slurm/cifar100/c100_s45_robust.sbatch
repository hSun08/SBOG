#!/bin/bash
#SBATCH -J c100_s45_robust
#SBATCH -p a100-40gb       # <- edit: a partition with 40 GB+ GPUs
#SBATCH -A liu334          # <- edit: your SLURM account
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 1-00:00:00
#SBATCH -o outputs/cifar100/logs/%x_%j.out
#SBATCH -e outputs/cifar100/logs/%x_%j.err
# Submit from the repo root (submit_*.sh does this): common.sh is located via SLURM_SUBMIT_DIR, or set SBOG_REPO.
set -euo pipefail
source "${SBOG_REPO:-$SLURM_SUBMIT_DIR}/slurm/cifar100/common.sh"
N=$(find "$OUT/generated/cifar100_soe_robust/samples" -type f 2>/dev/null | wc -l)
echo "generated images available: $N"; [[ $N -ge 90000 ]] || { echo "step3 output incomplete ($N < 90000)"; exit 1; }
run_step4 cifar100_soe_robust "$OUT/runs/detector_robust"
CKPT=$(latest_ckpt "$OUT/runs/detector_robust/energy_ft_sd" 'cifar100_wrn_s1_energy_ft_sd_*cifar100_soe_robust*_epoch_*.pt')
echo "detector checkpoint: $CKPT"; [[ -f "$CKPT" ]] || exit 1
run_step5 "$CKPT" cifar100_soe_robust "$OUT/bootstrap_eval/robust_bootstrap.json"
python -u tools/collect_cifar100_table.py --json "$OUT/bootstrap_eval/robust_bootstrap.json" --branch robust

# keep the final artifacts in one place (checkpoints/ is git-ignored; upload it to Drive/HF for the release)
bash tools/archive_cifar100.sh robust
