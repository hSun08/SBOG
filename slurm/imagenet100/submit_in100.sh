#!/bin/bash
# Submits the full ImageNet-100 reproduction (both branches) with SLURM dependencies.
# Usage: submit_in100.sh [--after <jobid[:jobid...]>]
set -euo pipefail
cd "$(dirname "$0")/../.."           # repo root; SLURM resolves the relative log paths from here
mkdir -p outputs/imagenet100/logs
DEP=""; [[ "${1:-}" == "--after" ]] && DEP="--dependency=afterok:$2"
jn=$(sbatch --parsable $DEP slurm/imagenet100/i100_s12_normal.sbatch)          # step1 normal + vanilla CE init + step2
jr=$(sbatch --parsable $DEP slurm/imagenet100/i100_s12_robust.sbatch)
j3n=$(sbatch --parsable --dependency=afterok:$jn slurm/imagenet100/i100_s3_normal.sbatch)
j3r=$(sbatch --parsable --dependency=afterok:$jr slurm/imagenet100/i100_s3_robust.sbatch)
j45n=$(sbatch --parsable --dependency=afterok:$j3n slurm/imagenet100/i100_s45_normal.sbatch)
j45r=$(sbatch --parsable --dependency=afterok:$j3r:$jn slurm/imagenet100/i100_s45_robust.sbatch)   # robust step4 also needs the vanilla init from the normal step1 job
echo "normal: step1+init+2=$jn${DEP:+ ($DEP)}  step3=$j3n  step4+5=$j45n"
echo "robust: step1+2=$jr${DEP:+ ($DEP)}  step3=$j3r  step4+5=$j45r (afterok:$j3r:$jn)"
