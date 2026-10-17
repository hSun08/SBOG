#!/bin/bash
#SBATCH -J i100_s12_robust
#SBATCH -p a100-40gb       # <- edit: a partition with 40 GB+ GPUs
#SBATCH -A liu334          # <- edit: your SLURM account
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 2-12:00:00
#SBATCH -o outputs/imagenet100/logs/%x_%j.out
#SBATCH -e outputs/imagenet100/logs/%x_%j.err
# Submit from the repo root (submit_*.sh does this): common.sh is located via SLURM_SUBMIT_DIR, or set SBOG_REPO.
set -euo pipefail
source "${SBOG_REPO:-$SLURM_SUBMIT_DIR}/slurm/imagenet100/common.sh"
# step1 (Sinkhorn-DRO robust alignment): 80 epochs, bs 48, 2 views, robust_weight 0.20 ramped over epochs 10-20, radius 0.03 (~40 h).
mkdir -p "$OUT/runs/pretrain_robust"
python -u sbog/pretrain_encoder_in100_robust.py \
  --data_root "$IN100" --token_embed_path "$TOKEN_EMBED" \
  --id_feat_output "$OUT/metadata/id_feat_custom100_full_ra2rw020.npy" \
  --save "$OUT/runs/pretrain_robust/" --my_info custom100_full_ra2rw020 \
  --epochs 80 --batch_size 48 --learning_rate 0.08 --momentum 0.9 --decay 0.0005 \
  --ngpu 1 --prefetch 4 --max_features_per_class 1000 \
  --num_views 2 --robust_weight 0.20 --robust_start_epoch 10 --robust_ramp_epochs 10 \
  --robust_temperature 1.0 --robust_radius 0.03 --logit_temperature 0.1
# step2
run_step2 "$OUT/metadata/id_feat_custom100_full_ra2rw020.npy" "$OUT/metadata/soe_robust"
ls -la "$OUT/metadata/soe_robust"
