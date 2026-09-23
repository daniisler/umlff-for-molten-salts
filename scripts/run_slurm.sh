#!/usr/bin/env bash
#SBATCH --job-name=uf3
#SBATCH --output=logs/uf3_%A_%a.log
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # UF3 is CPU-bound: give it cores
#SBATCH --mem=64G                   # raise a LOT for the full 3-body run (Track B)
# ---------------------------------------------------------------------------
# This is a CPU job. UF3 has NO GPU code path, so do NOT request --gres=gpu.
# The cost is featurization + one linear solve, both CPU/RAM bound. See
# docs/uf3_cluster_run_plan.md for the full rationale and the job matrix.
#
# Reproduce the environment once on the cluster:
#     uv sync            # builds the locked env (Python 3.12.14)
# Unpack the data next to the repo so ./data/ exists:
#     unzip umlff_data_archive.zip
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# --- pick ONE run, or drive many via a SLURM array (see bottom) ---

# (A2) 2-body, 4000 frames:
uv run python scripts/fit_uf3.py --mode 2body --n-train 4000 --n-test 1000 --seed 42

# (B0) full 3-body feasibility probe (instant, prints feature count + RAM):
# uv run python scripts/fit_uf3.py --mode 3body --probe-only --n-train 50

# (B1) full 3-body attempt — needs a high-memory node (e.g. --mem=512G):
# uv run python scripts/fit_uf3.py --mode 3body --n-train 2000 --n-test 500 \
#     --cutoff3 3.5 --res3 4 --force

# (C1b) 3-body on the Mg subsystem at higher fidelity:
# uv run python scripts/fit_uf3.py --mode 3body --n-train 800 --n-test 200 \
#     --train-file sub_CaMgCl.extxyz --test-file sub_CaMgCl.extxyz \
#     --elements Ca Cl Mg --cutoff3 5.0 --res3 6

# ---------------------------------------------------------------------------
# To fan out the whole matrix as an array, put one flag-string per line in
# jobs.txt and submit with:  sbatch --array=1-$(wc -l < jobs.txt) run_slurm.sh
# then replace the block above with:
#     FLAGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs.txt)
#     uv run python scripts/fit_uf3.py ${FLAGS}
# ---------------------------------------------------------------------------
