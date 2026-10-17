#!/bin/bash
#SBATCH -J i100_s45_normal
#SBATCH -p a100-40gb       # <- edit: a partition with 40 GB+ GPUs
#SBATCH -A liu334          # <- edit: your SLURM account
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 1-12:00:00
#SBATCH -o outputs/imagenet100/logs/%x_%j.out
#SBATCH -e outputs/imagenet100/logs/%x_%j.err
# Submit from the repo root (submit_*.sh does this): common.sh is located via SLURM_SUBMIT_DIR, or set SBOG_REPO.
set -euo pipefail
source "${SBOG_REPO:-$SLURM_SUBMIT_DIR}/slurm/imagenet100/common.sh"
N=$(find "$OUT/generated/custom100_soe_normal/samples" -type f 2>/dev/null | wc -l)
echo "generated images available: $N"; [[ $N -ge 90000 ]] || { echo "step3 output incomplete ($N < 90000)"; exit 1; }
[[ -f "$VANILLA_INIT" ]] || { echo "missing vanilla CE init $VANILLA_INIT (produced by i100_s12_normal)"; exit 1; }
run_step4 custom100_soe_normal "$OUT/runs/detector_normal" "$VANILLA_INIT"
CKPT=$(latest_ckpt "$OUT/runs/detector_normal" 'in100_wrn_s1_energy_ft_sd_*custom100_soe_normal*_epoch_*.pt')
echo "detector checkpoint: $CKPT"; [[ -f "$CKPT" ]] || exit 1
run_step5 "$CKPT" custom100_soe_normal "$OUT/bootstrap_eval/normal_bootstrap.json"
python -u tools/collect_in100_table.py --json "$OUT/bootstrap_eval/normal_bootstrap.json" --branch normal
bash tools/archive_in100.sh normal
