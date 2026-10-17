#!/bin/bash
#SBATCH -J c100_smoke
#SBATCH -p a100-40gb       # <- edit: a partition with 40 GB+ GPUs
#SBATCH -A liu334          # <- edit: your SLURM account
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-03:00:00
#SBATCH -o outputs/cifar100/logs/%x_%j.out
#SBATCH -e outputs/cifar100/logs/%x_%j.err
# End-to-end smoke test of the whole CIFAR-100 pipeline with tiny settings (~30-60 min).
# Writes to outputs/cifar100_smoke so it never touches the real outputs. Numbers are meaningless; only checks env/code/data.
export SBOG_OUT="${SBOG_REPO:-$SLURM_SUBMIT_DIR}/outputs/cifar100_smoke"
# Submit from the repo root (submit_*.sh does this): common.sh is located via SLURM_SUBMIT_DIR, or set SBOG_REPO.
set -euo pipefail
source "${SBOG_REPO:-$SLURM_SUBMIT_DIR}/slurm/cifar100/common.sh"
echo "### step1 normal (1 epoch)"
python -u sbog/pretrain_encoder_cifar100.py --data_root "$DATA/cifarpy" --token_embed_path "$TOKEN_EMBED" \
  --id_feat_output "$OUT/metadata/id_feat_cifar100_199epoch.npy" --save "$OUT/runs/pretrain_normal/" --my_info smoke \
  --epochs 1 --batch_size 256 --learning_rate 0.1 --ngpu 1 --prefetch 4 --max_features_per_class 500
echo "### step1 robust (1 epoch)"
python -u sbog/pretrain_encoder_cifar100_robust.py --data_root "$DATA/cifarpy" --token_embed_path "$TOKEN_EMBED" \
  --id_feat_output "$OUT/metadata/id_feat_cifar100_ra2rw010.npy" --save "$OUT/runs/pretrain_robust/" --my_info smoke \
  --epochs 1 --batch_size 128 --learning_rate 0.08 --ngpu 1 --prefetch 4 --max_features_per_class 500 \
  --num_views 2 --robust_weight 0.10 --robust_start_epoch 0 --robust_ramp_epochs 1 --max_train_steps 50
echo "### step2 (small)"
run_step2 "$OUT/metadata/id_feat_cifar100_199epoch.npy" "$OUT/metadata/soe_normal" \
  --num_samples_per_class 100 --anchor_topk 20 --anchor_pool_size 40 --proposals_per_anchor 64 --max_rounds 1
EMB=$(find "$OUT/metadata/soe_normal" -maxdepth 1 -name 'outlier_energy_semantic_embed*.npy' | head -n 1); echo "embedding: $EMB"
echo "### step3 (25 x 4 = 100 images)"
run_step3 "$EMB" "$OUT/generated/cifar100_soe_normal" --n_iter 25
echo "generated: $(find "$OUT/generated/cifar100_soe_normal/samples" -type f | wc -l)"
echo "### step4 (1 epoch, 20 steps)"
run_step4 cifar100_soe_normal "$OUT/runs/detector_normal" --epochs 1 --max_steps_per_epoch 20 --batch_size 64 --oe_batch_size 16
CKPT=$(latest_ckpt "$OUT/runs/detector_normal/energy_ft_sd" 'cifar100_wrn_s1_energy_ft_sd_*cifar100_soe_normal*_epoch_*.pt'); echo "ckpt: $CKPT"
echo "### step5 (+ 5-trial bootstrap)"
run_step5 "$CKPT" cifar100_soe_normal_smoke "$OUT/bootstrap_eval/normal_bootstrap.json" --bootstrap_trials 5
python -u tools/collect_cifar100_table.py --json "$OUT/bootstrap_eval/normal_bootstrap.json" --branch normal || true
echo "### SMOKE TEST PASSED"
