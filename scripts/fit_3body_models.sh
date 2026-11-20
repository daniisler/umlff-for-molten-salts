#!/usr/bin/env bash
# Fit the 11 pure single-chloride 3-body + ZBL models for the 3-body property campaign.
# For each cation X: carve the pure X-Cl frames from data/Training.extxyz, split, and fit
# with the LOCKED 3-body operating point (cutoff3=5, res3=10, ridge3=1e-5, kappa=0.3, ZBL).
# Each model -> results/3body_<X>Cl/model.json.
#
# 3-body featurization is slow (~tens of min per salt at NTRAIN frames) -> run on the cluster.
# Jobs are independent; split the CATIONS list across nodes to parallelise.
#
#   bash scripts/fit_3body_models.sh                 # all 11
#   NTRAIN=300 bash scripts/fit_3body_models.sh      # frames per fit (default 300)
#   CATIONS="Ca Zn Zr" bash scripts/fit_3body_models.sh   # just a subset
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:-uv run python}"
NTRAIN="${NTRAIN:-500}"
NTEST="${NTEST:-40}"
CATIONS="${CATIONS:-Li Na K Rb Cs Mg Ca Sr Ba Zn Zr}"

for c in ${CATIONS}; do
  stem="sub_${c}Cl_3body"
  run="3body_${c}Cl"
  if [ -f "results/${run}/model.json" ]; then
    echo "[skip] ${run} already fitted"; continue
  fi
  if [ ! -f "data/${stem}_train.extxyz" ]; then
    echo "[carve] ${c}Cl  (pure ${c}-Cl frames)"
    ${RUN} scripts/make_subsystem.py --elements Cl "${c}" --out "${stem}.extxyz"
    ${RUN} scripts/split_frames.py --input "${stem}.extxyz"
  fi
  echo "[fit ] ${run}  (n_train=${NTRAIN}, res3=10, cutoff3=5, ridge3=1e-5, ZBL)"
  ${RUN} scripts/fit_uf3.py --mode 3body \
    --train-file "${stem}_train.extxyz" --test-file "${stem}_test.extxyz" \
    --elements Cl "${c}" --n-train "${NTRAIN}" --n-test "${NTEST}" \
    --cutoff3 5 --res3 10 --ridge3 1e-5 --zbl --run-name "${run}" --force \
    || echo "WARN: ${run} fit did not finish"
done
echo "done. 3-body models in results/3body_*Cl/model.json"
