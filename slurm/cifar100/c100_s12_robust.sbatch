#!/bin/bash
#SBATCH -J c100_s12_robust
#SBATCH -p a100-40gb       # <- edit: a partition with 40 GB+ GPUs
#SBATCH -A liu334          # <- edit: your SLURM account
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-08:00:00
#SBATCH -o outputs/cifar100/logs/%x_%j.out
#SBATCH -e outputs/cifar100/logs/%x_%j.err
# Submit from the repo root (submit_*.sh does this): common.sh is located via SLURM_SUBMIT_DIR, or set SBOG_REPO.
set -euo pipefail
source "${SBOG_REPO:-$SLURM_SUBMIT_DIR}/slurm/cifar100/common.sh"
# step1 (Sinkhorn-DRO robust alignment): 80 epochs, 2 views, robust_weight 0.10 ramped over epochs 10-25, radius 0.03.
python -u sbog/pretrain_encoder_cifar100_robust.py \
  --data_root "$DATA/cifarpy" --token_embed_path "$TOKEN_EMBED" \
  --id_feat_output "$OUT/metadata/id_feat_cifar100_ra2rw010.npy" \
  --save "$OUT/runs/pretrain_robust/" --my_info ra2rw010 \
  --epochs 80 --batch_size 128 --learning_rate 0.08 --momentum 0.9 --decay 0.0005 \
  --ngpu 1 --prefetch 4 --max_features_per_class 500 \
  --num_views 2 --robust_weight 0.10 --robust_start_epoch 10 --robust_ramp_epochs 15 \
  --robust_temperature 1.0 --robust_radius 0.03 --logit_temperature 0.1
# step2
run_step2 "$OUT/metadata/id_feat_cifar100_ra2rw010.npy" "$OUT/metadata/soe_robust"
ls -la "$OUT/metadata/soe_robust"
