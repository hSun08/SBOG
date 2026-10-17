# Sourced by every CIFAR-100 SLURM script. Everything lives under the repo root.
set -euo pipefail
REPO=${SBOG_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}   # repo root: SLURM submit dir by default
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${SBOG_ENV:-$REPO/env}"   # conda env created as in README (./env) or any env name via SBOG_ENV
export PYTHONPATH="$REPO:$REPO/sbog:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="$REPO/cache/hf"
export TORCH_HOME="$REPO/cache/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA="$REPO/data"
OUT="${SBOG_OUT:-$REPO/outputs/cifar100}"
SD_CKPT="$REPO/models/sd-v1-4.ckpt"
TOKEN_EMBED="$REPO/assets/token_embed_c100.npy"
mkdir -p "$OUT"/{metadata,generated,bootstrap_eval,logs} "$OUT"/runs/{pretrain_normal,pretrain_robust,vanilla_ce}   # pretrain scripts use os.mkdir: parents must exist
cd "$REPO"
bash tools/preflight_cifar100.sh
echo "== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} OUT=$OUT"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
git -C "$REPO" rev-parse HEAD 2>/dev/null || true

# SOE latent-outlier synthesis (step2). Identical settings for the normal and robust branches.
run_step2 () {  # $1 = id_feat npy, $2 = output dir
  mkdir -p "$2"
  python -u sbog/soe_generate_latents.py \
    --token_embed_path "$TOKEN_EMBED" --id_feat_path "$1" --output_dir "$2" \
    --energy_eps 0.05 --gaussian_mag_ood_det 0.012 \
    --anchor_topk 200 --anchor_pool_size 400 --anchor_semantic_quantile 0.25 \
    --token_cos_floor_min 0.25 --token_cos_floor_max 0.90 --seed_cos_floor 0.70 \
    --local_k 128 --num_samples_per_class 10000 --proposals_per_anchor 256 --max_rounds 4 \
    --anchor_batch_size 32 --candidate_chunk_size 4096 --reference_chunk_size 256 \
    --threshold_margin 0.0 --device auto --log_every 5 "${@:3}"
}
# Stable Diffusion decoding of the accepted latents (step3): 25000 iterations x 4 = 100k images, round-robin over classes.
run_step3 () {  # $1 = embedding npy, $2 = output dir, extra args appended
  mkdir -p "$2"
  python -u sbog/sample_outliers_sd.py --plms --n_iter 25000 --n_samples 4 \
    --outdir "$2" --loaded_embedding "$1" --ckpt "$SD_CKPT" --id_data cifar100 \
    --class_schedule round_robin --skip_grid "${@:3}"
}
# Outlier-exposure detector training (step4): original CIFAR recipe, 200 epochs, energy_weight 2.5.
run_step4 () {  # $1 = my_info (generated subdir name), $2 = save dir, extra args appended
  mkdir -p "$2"
  python -u sbog/train_detector_cifar100.py \
    --data_root "$DATA/cifarpy" --generated_ood_root "$OUT/generated" --my_info "$1" --save "$2/" \
    --epochs 200 --batch_size 256 --oe_batch_size 256 --test_bs 200 --ngpu 1 --prefetch 4 \
    --score energy --energy_weight 2.5 "${@:3}"
}
# Evaluation (step5): point estimates + 500-trial bootstrap (seed 17) = the numbers in Table 4.
run_step5 () {  # $1 = checkpoint (.pt, the _bundle.pt is picked up automatically), $2 = method name, $3 = bootstrap json, extra args appended
  python -u sbog/test_detector_cifar100.py --load "$1" --method_name "$2" \
    --data_root "$DATA/cifarpy" --textures_root "$DATA/dtd/images" --svhn_root "$DATA/svhn" \
    --places_root "$DATA/places365_test" --lsun_root "$DATA/LSUN_C" --isun_root "$DATA/iSUN" \
    --score energy --ngpu 1 --prefetch 4 --test_bs 200
  python -u sbog/test_detector_cifar100_bootstrap.py --load "$1" --method_name "${2}_bootstrap" \
    --data_root "$DATA/cifarpy" --textures_root "$DATA/dtd/images" --svhn_root "$DATA/svhn" \
    --places_root "$DATA/places365_test" --lsun_root "$DATA/LSUN_C" --isun_root "$DATA/iSUN" \
    --score energy --bootstrap_trials 500 --bootstrap_seed 17 --subset_seed 17 \
    --test_bs 200 --prefetch 4 --ngpu 1 --output_json "$3" "${@:4}"
}
latest_ckpt () {  # $1 = dir, $2 = glob
  find "$1" -maxdepth 1 -type f -name "$2" ! -name '*_bundle.pt' | sort -V | tail -n 1
}
