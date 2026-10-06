#!/usr/bin/env bash
#SBATCH --job-name=uf3
#SBATCH --output=logs/uf3_%A_%a.log
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # UF3 is CPU-bound: give it cores
#SBATCH --mem=64G                   # batched fit keeps RAM modest even for big n
# ---------------------------------------------------------------------------
# PARALLEL RUNS via a SLURM job array. Each array task runs ONE line of
# scripts/jobs.txt (comment lines starting with # are skipped) through the
# BATCHED (split-RAM) fit, so large frame counts fit in memory on a normal node.
# UF3 is CPU-bound (no GPU code path): do NOT request --gres=gpu.
#
# EDIT for your cluster: partition/account lines, and the module/env setup
# (the two commented lines below) if `uv` is not already on PATH.
#
# One-time setup on the cluster:
#     uv sync                              # locked env, Python 3.12.14
#     unzip umlff_data_archive.zip         # creates ./data/
#     # fixed 2000-frame test slice of Validation (Track A scores the SAME test every job):
#     uv run python -c "from ase.io import read,write; import numpy as np; \
#       fr=read('data/Validation.extxyz',':'); i=sorted(np.random.default_rng(0).permutation(len(fr))[:2000]); \
#       write('data/Validation_test2k.extxyz',[fr[k] for k in i])"
#
# Submit EVERYTHING at once (all tasks run concurrently, scheduler permitting):
#     sbatch --array=1-$(grep -vc '^#' scripts/jobs.txt) scripts/run_slurm.sh
# Or a subset:
#     sbatch --array=1-12  scripts/run_slurm.sh     # Track A only (2-body large sets)
#     sbatch --array=13-21 scripts/run_slurm.sh     # Track B only (3-body)
# NOTE: --array indices are 1-based and skip the comment lines automatically
#       because we filter them below; keep jobs.txt's non-comment order stable.
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# module load python/3.12      # <- cluster-specific, if needed
# source ~/.local/bin/env      # <- if `uv` is not already on PATH

# pick the Nth non-comment line of jobs.txt (N = this array task index)
FLAGS=$(grep -v '^#' scripts/jobs.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "${FLAGS}" ]; then echo "no job for array index ${SLURM_ARRAY_TASK_ID}"; exit 0; fi

echo "[array ${SLURM_ARRAY_TASK_ID}] scripts/fit_uf3_batched.py ${FLAGS}"
uv run python scripts/fit_uf3_batched.py ${FLAGS}
