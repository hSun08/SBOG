#!/bin/bash
# Submits the full CIFAR-100 reproduction (both branches) with SLURM dependencies.
# Usage: submit_cifar100.sh [--after <jobid[:jobid...]>]   (e.g. chain behind the smoke test)
set -euo pipefail
cd "$(dirname "$0")/../.."           # repo root; SLURM resolves the relative log paths from here
mkdir -p outputs/cifar100/logs
DEP=""; [[ "${1:-}" == "--after" ]] && DEP="--dependency=afterok:$2"
for b in normal robust; do
  j12=$(sbatch --parsable $DEP slurm/cifar100/c100_s12_$b.sbatch)
  j3=$(sbatch --parsable --dependency=afterok:$j12 slurm/cifar100/c100_s3_$b.sbatch)
  j45=$(sbatch --parsable --dependency=afterok:$j3 slurm/cifar100/c100_s45_$b.sbatch)
  echo "$b: step1+2=$j12${DEP:+ ($DEP)}  step3=$j3 (afterok:$j12)  step4+5=$j45 (afterok:$j3)"
done
