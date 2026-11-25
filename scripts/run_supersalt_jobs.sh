#!/usr/bin/env bash
# Run the SuperSalt REFERENCE property sweeps listed in scripts/md_jobs_supersalt.txt.
# Each line is one salt/blend:   <name> <composition> <dt_fs> <T1> <T2> <T3>
# For each line: run SuperSalt (MACE + ZBL safety wall) NPT+NVT at every temperature via
# md_supersalt.py, writing results/md_supersalt_<name>_<T>K/, then analyze_properties.py
# computes the five properties -> results/supersalt_<name>_properties.json.
#
#     bash scripts/run_supersalt_jobs.sh              # all jobs
#     bash scripts/run_supersalt_jobs.sh 1 11         # only the 11 pure salts
#     bash scripts/run_supersalt_jobs.sh 12 16        # only the 5 blends
#     ATOMS=150 NPT=15000 NVT=8000 EQ=1500 bash scripts/run_supersalt_jobs.sh   # defaults
#     RUN="python" bash scripts/run_supersalt_jobs.sh # how to launch python (default: uv run python)
#
# MACE on CPU is ~1000x slower per step than the UF3 fast backend, so systems are kept
# small (ATOMS) and sweeps short (NPT/NVT). Jobs are independent -> run index ranges on
# different nodes to parallelise. Comment/blank lines in the jobs file are skipped.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

RUN="${RUN:-uv run python}"
ATOMS="${ATOMS:-150}"
NPT="${NPT:-15000}"
NVT="${NVT:-8000}"
EQ="${EQ:-1500}"
MODEL="${MODEL:-data/SuperSalt-swa.model}"
JOBS_FILE="${JOBS_FILE:-scripts/md_jobs_supersalt.txt}"

JOBS=()
while IFS= read -r line; do
  JOBS+=("$line")
done < <(grep -v '^[[:space:]]*#' "$JOBS_FILE" | grep -v '^[[:space:]]*$')
total=${#JOBS[@]}

start="${1:-1}"
end="${2:-$total}"
echo "supersalt jobs ${start}..${end} of ${total} from ${JOBS_FILE}  |  ${ATOMS} atoms, NPT ${NPT} / NVT ${NVT}, eq ${EQ}  |  RUN='${RUN}'"

for (( i=start; i<=end; i++ )); do
  read -r name comp dt temps <<< "${JOBS[$((i-1))]}"
  [ -z "${name:-}" ] && continue
  # self-describing tag so new runs never overwrite others in the SHARED results/ folder
  tag="${RUN_TAG:-dt${dt}_n${ATOMS}_npt${NPT}_nvt${NVT}}"
  pfx="${name}_${tag}"
  echo "[job ${i}/${total}] supersalt ${pfx}  (comp ${comp}; dt ${dt} fs; T = ${temps})"
  for temp in ${temps}; do
    tK="${temp%.*}"
    echo "=========== supersalt ${pfx} @ ${tK} K  (${ATOMS} atoms, dt ${dt} fs) ==========="
    ${RUN} scripts/md_supersalt.py "${comp}" --temp "${temp}" --dt "${dt}" \
      --atoms "${ATOMS}" --npt "${NPT}" --nvt "${NVT}" --eq "${EQ}" \
      --model "${MODEL}" --label "${pfx}_${tK}K" \
      || echo "WARN: supersalt ${pfx} @ ${tK} K did not finish"
  done
  echo "=== analyse supersalt_${pfx}: density, RDF, diffusion, Cp, alpha_V ==="
  ${RUN} scripts/analyze_properties.py "supersalt_${pfx}" ${temps} \
    || echo "WARN: analysis of supersalt_${pfx} failed"
done
echo "done jobs ${start}..${end}. Properties in results/supersalt_<name>_<tag>_properties.json."
