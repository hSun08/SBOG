#!/bin/bash
#SBATCH -J i100_smoke
#SBATCH -p a100-40gb       # <- edit: a partition with 40 GB+ GPUs
#SBATCH -A liu334          # <- edit: your SLURM account
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-04:00:00
#SBATCH -o outputs/imagenet100/logs/%x_%j.out
#SBATCH -e outputs/imagenet100/logs/%x_%j.err
# End-to-end smoke test of the ImageNet-100 pipeline with tiny settings (~1-1.5 h). Writes to outputs/imagenet100_smoke.
export SBOG_OUT="${SBOG_REPO:-$SLURM_SUBMIT_DIR}/outputs/imagenet100_smoke"
# Submit from the repo root (submit_*.sh does this): common.sh is located via SLURM_SUBMIT_DIR, or set SBOG_REPO.
set -euo pipefail
source "${SBOG_REPO:-$SLURM_SUBMIT_DIR}/slurm/imagenet100/common.sh"
echo "### step1 normal (1 epoch)"; mkdir -p "$OUT/runs/pretrain_normal"
python -u sbog/pretrain_encoder_in100.py --data_root "$IN100" --token_embed_path "$TOKEN_EMBED" \
  --id_feat_output "$OUT/metadata/id_feat_custom100_full.npy" --save "$OUT/runs/pretrain_normal/" --my_info smoke \
  --epochs 1 --batch_size 64 --learning_rate 0.1 --ngpu 1 --prefetch 4 --max_features_per_class 1000
echo "### step1 robust (50 steps)"; mkdir -p "$OUT/runs/pretrain_robust"
python -u sbog/pretrain_encoder_in100_robust.py --data_root "$IN100" --token_embed_path "$TOKEN_EMBED" \
  --id_feat_output "$OUT/metadata/id_feat_custom100_full_ra2rw020.npy" --save "$OUT/runs/pretrain_robust/" --my_info smoke \
  --epochs 1 --batch_size 48 --learning_rate 0.08 --ngpu 1 --prefetch 4 --max_features_per_class 1000 \
  --num_views 2 --robust_weight 0.20 --robust_start_epoch 0 --robust_ramp_epochs 1 --max_train_steps 50
echo "### vanilla CE (1 epoch, 200 steps)"
python -u sbog/pretrain_vanilla_ce_in100.py --data_root "$IN100" --save "$OUT/runs/vanilla_ce" --my_info custom100 --epochs 1 --max_steps_per_epoch 200
INIT="$OUT/runs/vanilla_ce/in100_wrn_s1_vanilla_in100_custom100_epoch_0.pt"
echo "### step2 (small)"
run_step2 "$OUT/metadata/id_feat_custom100_full.npy" "$OUT/metadata/soe_normal" \
  --num_samples_per_class 100 --anchor_topk 20 --anchor_pool_size 40 --proposals_per_anchor 64 --max_rounds 1
EMB=$(find "$OUT/metadata/soe_normal" -maxdepth 1 -name 'outlier_energy_semantic_embed*.npy' | head -n 1); echo "embedding: $EMB"
echo "### step3 (25 x 4 = 100 images)"
run_step3 "$EMB" "$OUT/generated/custom100_soe_normal" --n_iter 25
echo "generated: $(find "$OUT/generated/custom100_soe_normal/samples" -type f | wc -l)"
echo "### step4 (1 epoch)"
run_step4 custom100_soe_normal "$OUT/runs/detector_normal" "$INIT" --epochs 1
CKPT=$(latest_ckpt "$OUT/runs/detector_normal" 'in100_wrn_s1_energy_ft_sd_*custom100_soe_normal*_epoch_*.pt'); echo "ckpt: $CKPT"
echo "### step5 (+ 5-trial bootstrap)"
run_step5 "$CKPT" custom100_soe_normal_smoke "$OUT/bootstrap_eval/normal_bootstrap.json" --bootstrap_trials 5
python -u tools/collect_in100_table.py --json "$OUT/bootstrap_eval/normal_bootstrap.json" --branch normal || true
echo "### SMOKE TEST PASSED"
