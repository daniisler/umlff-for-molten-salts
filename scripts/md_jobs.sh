#!/usr/bin/env bash
# MD property pipeline -- the CLEAN 5 liquid properties across the single-salt series.
#
# Per salt: a 3-temperature sweep. Each temperature is an NPT + NVT run
# (md_moltensaltcalc.py); then analyze_properties.py turns the sweep into the five
# properties:
#     density  .  RDF/structure  .  self-diffusion  .  heat capacity Cp  .  thermal expansion alpha_V
# alpha_V and Cp NEED several temperatures (they are slopes/differences in T), which
# is why each salt is a sweep, not a single run.
#
# Bulk modulus is NOT computed here (it is a 0 K solid property).
# Viscosity is Track 2 below (separate + much longer; off by default).
#
#     bash scripts/md_jobs.sh                 # run Track 1 (all 11 salts)
#     RUN="python" bash scripts/md_jobs.sh    # choose how python launches (default: uv run python)
#     ATOMS=400 NPT=60000 NVT=60000 bash scripts/md_jobs.sh   # these are already the defaults
#
# Copy any single `sweep ...` line to run one salt. Each salt = 3 runs at 400 atoms /
# 60000+60000 steps -> several cluster-hours per salt.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

RUN="${RUN:-uv run python}"
ATOMS="${ATOMS:-400}"
NPT="${NPT:-60000}"
NVT="${NVT:-60000}"

# one temperature of a salt: run_T <model_dir> <T> <prefix> <cation>
run_T () {
  local model="results/$1/model.json" temp="$2" prefix="$3" cat="$4"
  echo "=========== ${prefix} @ ${temp} K  (${ATOMS} atoms) ==========="
  ${RUN} scripts/md_moltensaltcalc.py "$model" "$temp" "$NPT" "$NVT" "${prefix}_${temp}K" mtknpt "$ATOMS" "$cat" \
    || echo "WARN: ${prefix} @ ${temp} K did not finish cleanly"
}

# sweep a salt over its temperatures, then compute the 5 properties:
#   sweep <model_dir> <prefix> <cation> <T1> <T2> <T3> ...
sweep () {
  local model="$1" prefix="$2" cat="$3"; shift 3
  for temp in "$@"; do run_T "$model" "$temp" "$prefix" "$cat"; done
  echo "=== analyse ${prefix}: density, RDF, diffusion, Cp, alpha_V ==="
  ${RUN} scripts/analyze_properties.py "$prefix" "$@" || echo "WARN: analysis of ${prefix} failed"
}

# ===== Track 1 -- clean-5 property sweep (all 11 salts) =====
#        model_dir           prefix  cation   T1   T2   T3   (temperatures span each salt's liquid range)
sweep 2body_LiCl_zbl    LiCl  Li   920  1070 1220
sweep 2body_NaCl_zbl    NaCl  Na   1100 1250 1400
sweep 2body_KRbCl_zbl   KCl   K    1075 1225 1375
sweep 2body_KRbCl_zbl   RbCl  Rb   1020 1170 1320
sweep 2body_CsCl_zbl    CsCl  Cs   950  1100 1250
sweep 2body_CaMgCl_zbl  MgCl2 Mg   1000 1150 1300
sweep 2body_CaMgCl_zbl  CaCl2 Ca   1075 1225 1375
sweep 2body_SrCl_zbl    SrCl2 Sr   1180 1330 1480
sweep 2body_BaCl_zbl    BaCl2 Ba   1270 1380 1490
sweep 2body_ZnCl_zbl    ZnCl2 Zn   600  780  960
sweep 2body_CsZrCl_zbl  ZrCl4 Zr   710  800  890   # > 765 K is model-only (Janz valid 710-765)

echo "Track 1 done. Five properties per salt in results/<salt>_properties.json."

# ===== Track 2 -- viscosity (SEPARATE, much longer; off by default) =====
# Green-Kubo viscosity needs a long, well-sampled NVT run and is noisy. Enable
# per salt as a follow-up. Short NPT for the box, then a long NVT, then analyse:
# visc () {
#   local model="results/$1/model.json" temp="$2" prefix="$3" cat="$4"
#   ${RUN} scripts/md_moltensaltcalc.py "$model" "$temp" 20000 200000 "${prefix}_visc_${temp}K" mtknpt "$ATOMS" "$cat"
#   ${RUN} scripts/viscosity.py "results/md_${prefix}_visc_${temp}K" "$temp" --tmax-fs 20000
# }
# visc 2body_NaCl_zbl 1100 NaCl Na
