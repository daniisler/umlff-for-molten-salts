#!/usr/bin/env bash
# Generate the 6 subsystem split files that Track G / Track H need, from
# data/Training.extxyz. data/ is gitignored, so these are NOT in the repo and
# must be (re)generated once on each fresh machine / node.
#
#     bash scripts/setup_subsystems.sh            # generate any that are missing
#     RUN="python" bash scripts/setup_subsystems.sh   # override how python is launched
#
# Deterministic: make_subsystem filters Training.extxyz; split_frames uses
# seed 42 / 20% test -- so the splits reproduce exactly what the earlier tracks used.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:-uv run python}"

gen () {                 # $1 = output stem (e.g. sub_KRbCl);  $2.. = elements
  local out="$1"; shift
  if [ -f "data/${out}_train.extxyz" ] && [ -f "data/${out}_test.extxyz" ]; then
    echo "[skip] ${out} splits already present"; return
  fi
  echo "[gen ] ${out}  (elements: $*)"
  ${RUN} scripts/make_subsystem.py --elements "$@" --out "${out}.extxyz"
  ${RUN} scripts/split_frames.py --input "${out}.extxyz"
}

gen sub_KRbCl  Cl K Rb       # spherical +1 (control)
gen sub_ClCsLi Cl Li Cs      # monovalent, Li small
gen sub_CaMgCl Cl Ca Mg      # small divalents
gen sub_BaClSr Cl Ba Sr      # large soft divalents
gen sub_ClKZn  Cl K Zn       # TM divalent Zn2+
gen sub_CsZrCl Cl Cs Zr      # directional +4

echo "all 6 subsystem split files ready in data/"
