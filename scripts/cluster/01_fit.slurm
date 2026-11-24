#!/usr/bin/env bash
#SBATCH --job-name=uf3-fit
#SBATCH --output=logs/fit_%A_%a.log
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # UF3 is CPU-bound: give it cores, NOT a GPU
#SBATCH --mem=96G                   # batched 2-body keeps RAM modest; 3-body subsystems are small
## --partition=<your_cpu_partition>  # <- EDIT for your cluster
## --account=<your_account>          # <- EDIT for your cluster
# ---------------------------------------------------------------------------
# Fit the MD-campaign prerequisite models, ONE per array task, from
# scripts/cluster/fit_manifest.txt. UF3 has no GPU code path -> do NOT request --gres=gpu.
#
#   sbatch --array=1-$(grep -vc '^[[:space:]]*#' scripts/cluster/fit_manifest.txt) scripts/cluster/01_fit.slurm
#
# A subset (e.g. only the 5 blend-prerequisite models):
#   sbatch --array=1-5 scripts/cluster/01_fit.slurm
# ---------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# module load python/3.12      # <- cluster-specific, if needed
# source ~/.local/bin/env      # <- if `uv` is not already on PATH

LINE=$(grep -v '^[[:space:]]*#' scripts/cluster/fit_manifest.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")
[ -z "${LINE}" ] && { echo "no fit for array index ${SLURM_ARRAY_TASK_ID}"; exit 0; }
echo "[fit ${SLURM_ARRAY_TASK_ID}] ${LINE}"
bash -c "${LINE}"
