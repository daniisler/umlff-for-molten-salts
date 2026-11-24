#!/usr/bin/env bash
#SBATCH --job-name=ss-md
#SBATCH --output=logs/ss_%A_%a.log
#SBATCH --time=48:00:00             # MACE on CPU is slow -> generous walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
## --partition=<your_cpu_partition>  # <- EDIT for your cluster
## --account=<your_account>          # <- EDIT for your cluster
# ---------------------------------------------------------------------------
# SuperSalt REFERENCE MD -- one salt/blend (3-T sweep + 5-property analysis) per array
# task, from scripts/md_jobs_supersalt.txt. Needs NO fitting (data/SuperSalt-swa.model is
# committed), so this array runs in parallel with the fits.
#
#   sbatch --array=1-$(grep -vc '^[[:space:]]*#' scripts/md_jobs_supersalt.txt) scripts/cluster/03_md_supersalt.slurm
#
# MACE is ~1000x slower per step than the UF3 fast backend -> systems are kept small.
# Tune ATOMS/NPT/NVT down if queue time is tight (density gets noisier, not wrong).
# ---------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# module load python/3.12      # <- cluster-specific, if needed
# source ~/.local/bin/env      # <- if `uv` is not already on PATH

export ATOMS="${ATOMS:-150}" NPT="${NPT:-15000}" NVT="${NVT:-8000}"
bash scripts/run_supersalt_jobs.sh "${SLURM_ARRAY_TASK_ID}" "${SLURM_ARRAY_TASK_ID}"
