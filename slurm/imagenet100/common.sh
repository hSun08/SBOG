# Sourced by every ImageNet-100 SLURM script. Everything lives under the repo root.
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
IN100="$DATA/imagenet100"                 # ImageFolder: train/ val/ test/ metadata/
OOD="$DATA/imagenet_ood"                  # iNaturalist_imagefolder/ SUN_imagefolder/ Places_imagefolder/
OUT="${SBOG_OUT:-$REPO/outputs/imagenet100}"
SD_CKPT="$REPO/models/sd-v1-4.ckpt"
TOKEN_EMBED="$IN100/metadata/token_embed_custom100.npy"
# step-4 init: the CE-pretrained ResNet-34 released by DreamOOD (setup/download_in100_ce_init.sh). Our own 10-epoch vanilla CE
# (runs/vanilla_ce/..._epoch_9.pt) gave a clearly worse detector (ID error 15.5 % vs 11.8 %), see CHECKLIST.
VANILLA_INIT="${SBOG_IN100_INIT:-$REPO/models/dreamood_release/in100_initialized_model_for_ood_det.pt}"
mkdir -p "$OUT"/{metadata,generated,bootstrap_eval,logs} "$OUT"/runs/{pretrain_normal,pretrain_robust,vanilla_ce}   # pretrain scripts use os.mkdir: parents must exist
cd "$REPO"
bash tools/preflight_in100.sh
echo "== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} OUT=$OUT"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
git -C "$REPO" rev-parse HEAD 2>/dev/null || true

# SOE latent-outlier synthesis (step 2). Same settings as the paper runs (no token_cos_floor_max on ImageNet-100).
run_step2 () {  # $1 = id_feat npy, $2 = output dir, extra args appended
  mkdir -p "$2"
  python -u sbog/soe_generate_latents.py \
    --token_embed_path "$TOKEN_EMBED" --id_feat_path "$1" --output_dir "$2" \
    --energy_eps 0.05 --gaussian_mag_ood_det 0.012 \
    --anchor_topk 200 --anchor_pool_size 400 --anchor_semantic_quantile 0.25 \
    --token_cos_floor_min 0.25 --seed_cos_floor 0.70 \
    --local_k 128 --num_samples_per_class 10000 --proposals_per_anchor 256 --max_rounds 4 \
    --anchor_batch_size 32 --candidate_chunk_size 4096 --reference_chunk_size 256 \
    --threshold_margin 0.0 --device auto --log_every 5 "${@:3}"
}
# Stable Diffusion decoding (step 3): 25000 x 4 = 100k images, round-robin over the 100 classes, class names from metadata/.
run_step3 () {  # $1 = embedding npy, $2 = output dir, extra args appended
  mkdir -p "$2"
  python -u sbog/sample_outliers_sd.py --plms --n_iter 25000 --n_samples 4 \
    --outdir "$2" --loaded_embedding "$1" --ckpt "$SD_CKPT" --id_data in100 \
    --custom_metadata_dir "$IN100/metadata" --class_schedule round_robin --skip_grid "${@:3}"
}
# Outlier-exposure detector (step 4): 30 epochs, bs 32, energy_weight 0.75, initialised from the vanilla CE checkpoint.
run_step4 () {  # $1 = my_info (generated subdir), $2 = save dir (script appends energy_ft_sd/), $3 = init ckpt, extra args appended
  mkdir -p "$2"
  python -u sbog/train_detector_in100.py \
    --data_root "$IN100" --generated_ood_root "$OUT/generated" --my_info "$1" --load "$3" --save "$2/" \
    --epochs 30 --batch_size 32 --oe_batch_size 32 --test_bs 200 --ngpu 1 --prefetch 4 \
    --score energy --energy_weight 0.75 --additional_info w075_e30 --log_interval 20 "${@:4}"
}
# Evaluation (step 5): point estimates + 500-trial bootstrap (seed 17) = Table 3 numbers.
run_step5 () {  # $1 = checkpoint, $2 = method name, $3 = bootstrap json, extra args appended
  python -u sbog/test_detector_in100.py --load "$1" --method_name "$2" --data_root "$IN100" \
    --inat_root "$OOD/iNaturalist_imagefolder" --places_root "$OOD/Places_imagefolder" \
    --sun_root "$OOD/SUN_imagefolder" --textures_root "$DATA/dtd/images" \
    --score energy --ngpu 1 --prefetch 4 --test_bs 64
  python -u sbog/test_detector_in100_bootstrap.py --load "$1" --method_name "${2}_bootstrap" --data_root "$IN100" \
    --inat_root "$OOD/iNaturalist_imagefolder" --places_root "$OOD/Places_imagefolder" \
    --sun_root "$OOD/SUN_imagefolder" --textures_root "$DATA/dtd/images" \
    --score energy --bootstrap_trials 500 --bootstrap_seed 17 --subset_seed 17 \
    --ngpu 1 --prefetch 4 --test_bs 64 --output_json "$3" "${@:4}"
}
latest_ckpt () { find "$1" -maxdepth 2 -type f -name "$2" ! -name '*_bundle.pt' | sort -V | tail -n 1; }
