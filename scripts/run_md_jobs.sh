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
BAROSTAT="${BAROSTAT:-mtknpt}"
JOBS_FILE="${JOBS_FILE:-scripts/md_jobs.txt}"   # override for the 3-body campaign
# Self-describing run tag so new runs never overwrite others in the SHARED results/ folder.
# It encodes the knobs that vary between runs (timestep, atoms, step counts, barostat); the
# model backend (2b/3b) is already in <prefix>. Override RUN_TAG for a custom campaign label.
RUN_TAG="${RUN_TAG:-dt${DT_FS:-1.0}_n${ATOMS}_npt${NPT}_nvt${NVT}_${BAROSTAT}}"

# collect job lines (skip comments and blanks)
JOBS=()
while IFS= read -r line; do
  JOBS+=("$line")
done < <(grep -v '^[[:space:]]*#' "$JOBS_FILE" | grep -v '^[[:space:]]*$')
total=${#JOBS[@]}

start="${1:-1}"
end="${2:-$total}"
echo "md jobs ${start}..${end} of ${total} from ${JOBS_FILE}  |  ${ATOMS} atoms, NPT ${NPT} / NVT ${NVT}  |  dt=${DT_FS:-1.0} fs  |  tag=${RUN_TAG}  |  RUN='${RUN}'"

for (( i=start; i<=end; i++ )); do
  read -r model prefix cat temps <<< "${JOBS[$((i-1))]}"
  [ -z "${model:-}" ] && continue
  pfx="${prefix}_${RUN_TAG}"            # unique, self-describing output name
  echo "[job ${i}/${total}] ${pfx}  (cation ${cat}; T = ${temps})"
  for temp in ${temps}; do
    echo "=========== ${pfx} @ ${temp} K  (${ATOMS} atoms) ==========="
    ${RUN} scripts/md_moltensaltcalc.py "results/${model}/model.json" "$temp" "$NPT" "$NVT" \
      "${pfx}_${temp}K" "$BAROSTAT" "$ATOMS" "$cat" || echo "WARN: ${pfx} @ ${temp} K did not finish"
  done
  echo "=== analyse ${pfx}: density, RDF, diffusion, Cp, alpha_V ==="
  ${RUN} scripts/analyze_properties.py "$pfx" ${temps} || echo "WARN: analysis of ${pfx} failed"
done
echo "done jobs ${start}..${end}. Properties -> results/<prefix>_${RUN_TAG}_properties.json ; runs -> results/md_<prefix>_${RUN_TAG}_<T>K/"
