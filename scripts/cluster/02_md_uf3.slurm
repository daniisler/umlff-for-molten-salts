#!/usr/bin/env bash
#SBATCH --job-name=uf3-md
#SBATCH --output=logs/md_%A_%a.log
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # the fast UF3 backends are single-node, CPU-bound
#SBATCH --mem=16G
## --partition=<your_cpu_partition>  # <- EDIT for your cluster
## --account=<your_account>          # <- EDIT for your cluster
# ---------------------------------------------------------------------------
# One UF3 MD job (one salt/blend, 3-T sweep + 5-property analysis) per array task, from
# scripts/cluster/md_manifest.txt. Needs the prerequisite models -> run 01_fit.slurm first
# (submit_all.sh chains them with --dependency=afterok).
#
#   sbatch --array=1-$(grep -vc '^[[:space:]]*#' scripts/cluster/md_manifest.txt) scripts/cluster/02_md_uf3.slurm
# ---------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# module load python/3.12      # <- cluster-specific, if needed
# source ~/.local/bin/env      # <- if `uv` is not already on PATH

read -r JF IDX DT <<< "$(grep -v '^[[:space:]]*#' scripts/cluster/md_manifest.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")"
[ -z "${JF:-}" ] && { echo "no md job for array index ${SLURM_ARRAY_TASK_ID}"; exit 0; }
echo "[md ${SLURM_ARRAY_TASK_ID}] ${JF}  job ${IDX}  at dt=${DT} fs"
DT_FS="${DT}" JOBS_FILE="${JF}" bash scripts/run_md_jobs.sh "${IDX}" "${IDX}"
