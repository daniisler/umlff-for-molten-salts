#!/usr/bin/env bash
# Run every job in scripts/jobs.txt through the BATCHED (split-RAM) UF3 fit,
# in parallel, on a plain machine. No SLURM / no scheduler needed -- just run it:
#
#     bash scripts/run_jobs.sh                 # run all jobs, N_PARALLEL at a time
#     N_PARALLEL=8 bash scripts/run_jobs.sh    # 8 jobs at once
#     bash scripts/run_jobs.sh 13 21           # only jobs 13..21 (1-based; comment lines skipped)
#
# Knobs (environment variables):
#     N_PARALLEL    jobs to run at the same time        (default 4)
#     CORES_PER_JOB threads each job may use             (default 4)
#     RUN           how to launch python                 (default "uv run python")
#                   e.g.  RUN="python"  or  RUN="/path/to/venv/bin/python"
#
# Each job's output -> logs/job_NN.log ; metrics land in results/<run_name>/metrics.json
# as usual. Roll them up afterwards with the collector one-liner in HANDOFF.md.
#
# One-time data setup (same as before):
#     unzip umlff_data_archive.zip        # creates ./data/
#     # fixed 2000-frame test slice of Validation (Track A scores the SAME test every job):
#     ${RUN:-uv run python} -c "from ase.io import read,write; import numpy as np; \
#       fr=read('data/Validation.extxyz',':'); i=sorted(np.random.default_rng(0).permutation(len(fr))[:2000]); \
#       write('data/Validation_test2k.extxyz',[fr[k] for k in i])"
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root, no matter where this is called from
mkdir -p logs

N_PARALLEL="${N_PARALLEL:-4}"
CORES_PER_JOB="${CORES_PER_JOB:-4}"
RUN="${RUN:-uv run python}"

# keep parallel jobs from oversubscribing the cores
export OMP_NUM_THREADS="$CORES_PER_JOB"
export OPENBLAS_NUM_THREADS="$CORES_PER_JOB"
export MKL_NUM_THREADS="$CORES_PER_JOB"

# collect job lines (skip comments and blank lines), portably into an array
JOBS=()
while IFS= read -r line; do
  JOBS+=("$line")
done < <(grep -v '^[[:space:]]*#' scripts/jobs.txt | grep -v '^[[:space:]]*$')
total=${#JOBS[@]}

start="${1:-1}"
end="${2:-$total}"

echo "jobs ${start}..${end} of ${total}  |  ${N_PARALLEL} at a time  |  ${CORES_PER_JOB} cores each  |  RUN='${RUN}'"

running=0
for (( i=start; i<=end; i++ )); do
  flags="${JOBS[$((i-1))]}"
  [ -z "$flags" ] && continue
  log="logs/job_$(printf '%02d' "$i").log"
  echo "[job ${i}/${total}] -> ${log}  ::  ${flags}"
  ( ${RUN} scripts/fit_uf3_batched.py ${flags} ) > "$log" 2>&1 &
  running=$((running + 1))
  if [ "$running" -ge "$N_PARALLEL" ]; then
    wait          # let the current batch finish before launching the next
    running=0
  fi
done
wait
echo "all done. metrics in results/  (roll up with the collector one-liner in HANDOFF.md)"
