#!/usr/bin/env bash
# Submit the whole UF3-vs-SuperSalt MD campaign to SLURM in dependency order:
#   1. fit the prerequisite models        (01_fit.slurm, array)
#   2. UF3 MD -- needs those models        (02_md_uf3.slurm, afterok the fits)
#   3. SuperSalt MD -- needs no fitting     (03_md_supersalt.slurm, runs in parallel)
#   4. MgCl2 3-body boiling-point probe    (mg_probe.slurm, afterok the fits)
# Run from the repo root on the cluster login node. FIRST edit the #SBATCH partition/
# account/module lines in the .slurm files for your cluster.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../.."

nfit=$(grep -vc '^[[:space:]]*#' scripts/cluster/fit_manifest.txt)
nmd=$(grep -vc '^[[:space:]]*#' scripts/cluster/md_manifest.txt)
nss=$(grep -vc '^[[:space:]]*#' scripts/md_jobs_supersalt.txt)

fit=$(sbatch --parsable --array=1-"${nfit}" scripts/cluster/01_fit.slurm)
echo "fits        -> job ${fit}  (${nfit} tasks)"

md=$(sbatch --parsable --dependency=afterok:"${fit}" --array=1-"${nmd}" scripts/cluster/02_md_uf3.slurm)
echo "UF3 MD      -> job ${md}  (${nmd} tasks, starts after ALL fits succeed)"

ss=$(sbatch --parsable --array=1-"${nss}" scripts/cluster/03_md_supersalt.slurm)
echo "SuperSalt   -> job ${ss}  (${nss} tasks, independent of the fits)"

probe=$(sbatch --parsable --dependency=afterok:"${fit}" scripts/cluster/mg_probe.slurm)
echo "MgCl2 probe -> job ${probe}  (1 task, starts after fits; uses the 3body_MgCl model)"

echo
echo "watch:    squeue -u \$USER     |  tail -f logs/*.log"
echo "outputs:  results/<salt>_properties.json , results/supersalt_<name>_properties.json ,"
echo "          results/mg_phase_probe_3body/summary.json"
echo
echo "NOTE: the UF3 MD array and the probe use afterok, so if ONE fit fails they are held."
echo "      Re-run just the failed fit (e.g. sbatch --array=<k> scripts/cluster/01_fit.slurm),"
echo "      then release:  scontrol release ${md} ${probe}  (or resubmit those scripts)."
