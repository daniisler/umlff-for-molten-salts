#!/usr/bin/env bash
# Run the MD property sweeps listed in scripts/md_jobs.txt. Each line is one salt:
#     <model_dir> <prefix> <cation> <T1> <T2> <T3> ...
# For each line: run NPT+NVT at every temperature, then analyze_properties.py computes
# the five properties (density, RDF, self-diffusion, Cp, thermal expansion alpha_V).
#
#     bash scripts/run_md_jobs.sh                 # all jobs
#     bash scripts/run_md_jobs.sh 1 9             # only jobs 1..9 (the core salts)
#     RUN="python" bash scripts/run_md_jobs.sh    # how to launch python (default: uv run python)
#     ATOMS=400 NPT=60000 NVT=60000 bash scripts/run_md_jobs.sh   # these are the defaults
#
# Jobs are independent -> run different index ranges on different nodes to parallelise.
# Comment and blank lines in md_jobs.txt are skipped when counting jobs.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

RUN="${RUN:-uv run python}"
ATOMS="${ATOMS:-400}"
NPT="${NPT:-60000}"
NVT="${NVT:-60000}"

# collect job lines (skip comments and blanks)
JOBS=()
while IFS= read -r line; do
  JOBS+=("$line")
done < <(grep -v '^[[:space:]]*#' scripts/md_jobs.txt | grep -v '^[[:space:]]*$')
total=${#JOBS[@]}

start="${1:-1}"
end="${2:-$total}"
echo "md jobs ${start}..${end} of ${total}  |  ${ATOMS} atoms, NPT ${NPT} / NVT ${NVT}  |  RUN='${RUN}'"

for (( i=start; i<=end; i++ )); do
  read -r model prefix cat temps <<< "${JOBS[$((i-1))]}"
  [ -z "${model:-}" ] && continue
  echo "[job ${i}/${total}] ${prefix}  (cation ${cat}; T = ${temps})"
  for temp in ${temps}; do
    echo "=========== ${prefix} @ ${temp} K  (${ATOMS} atoms) ==========="
    ${RUN} scripts/md_moltensaltcalc.py "results/${model}/model.json" "$temp" "$NPT" "$NVT" \
      "${prefix}_${temp}K" mtknpt "$ATOMS" "$cat" || echo "WARN: ${prefix} @ ${temp} K did not finish"
  done
  echo "=== analyse ${prefix}: density, RDF, diffusion, Cp, alpha_V ==="
  ${RUN} scripts/analyze_properties.py "$prefix" ${temps} || echo "WARN: analysis of ${prefix} failed"
done
echo "done jobs ${start}..${end}. Five properties per salt in results/<salt>_properties.json."
