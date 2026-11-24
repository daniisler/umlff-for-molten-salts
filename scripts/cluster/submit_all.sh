#!/usr/bin/env bash
# Submit the whole UF3-vs-SuperSalt MD campaign to SLURM in dependency order:
#   1. fit the prerequisite models        (01_fit.slurm, array)
#   2. UF3 MD -- needs those models        (02_md_uf3.slurm, afterok the fits)
#   3. SuperSalt MD -- needs no fitting     (03_md_supersalt.slurm, runs in parallel)
# Run from the repo root on the cluster login node. FIRST edit the #SBATCH partition/
# account/module lines in the three .slurm files for your cluster.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../.."

nfit=$(grep -vc '^[[:space:]]*#' scripts/cluster/fit_manifest.txt)
nmd=$(grep -vc '^[[:space:]]*#' scripts/cluster/md_manifest.txt)
nss=$(grep -vc '^[[:space:]]*#' scripts/md_jobs_supersalt.txt)

fit=$(sbatch --parsable --array=1-"${nfit}" scripts/cluster/01_fit.slurm)
echo "fits       -> job ${fit}  (${nfit} tasks)"

md=$(sbatch --parsable --dependency=afterok:"${fit}" --array=1-"${nmd}" scripts/cluster/02_md_uf3.slurm)
echo "UF3 MD     -> job ${md}  (${nmd} tasks, starts after ALL fits succeed)"

ss=$(sbatch --parsable --array=1-"${nss}" scripts/cluster/03_md_supersalt.slurm)
echo "SuperSalt  -> job ${ss}  (${nss} tasks, independent of the fits)"

echo
echo "watch:    squeue -u \$USER     |  tail -f logs/*.log"
echo "outputs:  results/<salt>_properties.json  and  results/supersalt_<name>_properties.json"
echo
echo "NOTE: the UF3 MD array uses afterok, so if ONE fit fails the whole MD array is held."
echo "      Re-run just the failed fit (e.g. sbatch --array=<k> scripts/cluster/01_fit.slurm),"
echo "      then release:  scontrol release ${md}   (or resubmit 02_md_uf3.slurm)."
