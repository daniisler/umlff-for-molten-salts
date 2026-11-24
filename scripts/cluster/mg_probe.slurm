#!/usr/bin/env bash
#SBATCH --job-name=mg-probe
#SBATCH --output=logs/mg_probe_%j.log
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
## --partition=<your_cpu_partition>  # <- EDIT for your cluster
## --account=<your_account>          # <- EDIT for your cluster
# ---------------------------------------------------------------------------
# MgCl2 3-body boiling-point probe -- a finer, numerically STABLE temperature sweep
# (dt = 0.25 fs, FIRE -> NVT-eq -> NPT at 850/950/1000/1050 K + an NVT density check)
# asking the decisive question: does the 3-body angular term lift MgCl2's boiling point
# back above its melting point (987 K)? This is SEPARATE from the property campaign
# (which runs MgCl2 at 900/1000/1100 K); it's a diagnostic, run it on its own.
#
#   sbatch scripts/cluster/mg_probe.slurm
#
# Reliable here because the node never sleeps (the laptop kept napping mid-run). CPU-only.
# Output: results/mg_phase_probe_3body/summary.json (+ per-run dirs); the verdict prints at the end.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# module load python/3.12      # <- cluster-specific, if needed
# source ~/.local/bin/env      # <- if `uv` is not already on PATH

# the probe needs the 3-body MgCl model; 3-body models are not committed, so fit it if absent
[ -f results/3body_MgCl/model.json ] || CATIONS="Mg" bash scripts/fit_3body_models.sh

uv run python scripts/mg_phase_probe_3body.py
