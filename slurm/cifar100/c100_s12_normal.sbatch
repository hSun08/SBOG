#!/bin/bash
#SBATCH -J c100_s12_normal
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
# step1 (normal alignment): CE on cosine logits to the class text tokens, 200 epochs; dumps 500 ID features / class.
python -u sbog/pretrain_encoder_cifar100.py \
  --data_root "$DATA/cifarpy" --token_embed_path "$TOKEN_EMBED" \
  --id_feat_output "$OUT/metadata/id_feat_cifar100_199epoch.npy" \
  --save "$OUT/runs/pretrain_normal/" --my_info c100_full \
  --epochs 200 --batch_size 256 --learning_rate 0.1 --momentum 0.9 --decay 0.0005 \
  --ngpu 1 --prefetch 4 --max_features_per_class 500
# step2
run_step2 "$OUT/metadata/id_feat_cifar100_199epoch.npy" "$OUT/metadata/soe_normal"
ls -la "$OUT/metadata/soe_normal"
