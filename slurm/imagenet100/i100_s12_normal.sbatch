#!/bin/bash
#SBATCH -J i100_s12_normal
#SBATCH -p a100-40gb       # <- edit: a partition with 40 GB+ GPUs
#SBATCH -A liu334          # <- edit: your SLURM account
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 1-16:00:00
#SBATCH -o outputs/imagenet100/logs/%x_%j.out
#SBATCH -e outputs/imagenet100/logs/%x_%j.err
# Submit from the repo root (submit_*.sh does this): common.sh is located via SLURM_SUBMIT_DIR, or set SBOG_REPO.
set -euo pipefail
source "${SBOG_REPO:-$SLURM_SUBMIT_DIR}/slurm/imagenet100/common.sh"
# step1 (normal alignment): CE on cosine logits to the CLIP class tokens, 80 epochs, bs 64; dumps 1000 ID features / class (~20 h).
mkdir -p "$OUT/runs/pretrain_normal"
python -u sbog/pretrain_encoder_in100.py \
  --data_root "$IN100" --token_embed_path "$TOKEN_EMBED" \
  --id_feat_output "$OUT/metadata/id_feat_custom100_full.npy" \
  --save "$OUT/runs/pretrain_normal/" --my_info custom100_full \
  --epochs 80 --batch_size 64 --learning_rate 0.1 --momentum 0.9 --decay 0.0005 \
  --ngpu 1 --prefetch 4 --max_features_per_class 1000
# optional: our own vanilla CE classifier (10 epochs, every epoch kept). NOT used as the step-4 init by default any more
# (common.sh points VANILLA_INIT at the DreamOOD-released CE model); kept for the ablation in CHECKLIST.
python -u sbog/pretrain_vanilla_ce_in100.py --data_root "$IN100" --save "$OUT/runs/vanilla_ce" --my_info custom100 \
  --epochs 10 --batch_size 64 --learning_rate 0.1 --prefetch 8
# step2
run_step2 "$OUT/metadata/id_feat_custom100_full.npy" "$OUT/metadata/soe_normal"
ls -la "$OUT/metadata/soe_normal" "$OUT/runs/vanilla_ce"
