"""
Quick live test: does UFCalculator produce energy / forces / STRESS end-to-end?

Loads an already-fitted model.json, wraps it in UFCalculator, puts it on ONE
real frame from the Cs-Zr-Cl test set, and calls get_potential_energy /
get_forces / get_stress. Times the stress call (it's numerical => ~12 energy
evals, so this tells us the real cost).

Run from the repo root:
    uv run python scripts/stress_test.py
Optionally pass a different model.json as the first argument.
"""
import sys
import time

import numpy as np
from ase.io import read

from uf3.regression.least_squares import WeightedLinearModel
from uf3.forcefield.calculator import UFCalculator

# --- pick a fitted model (default: the n=500 2-body-equivalent 3-body fit is slow;
#     use a 2-body model for a fast first check). Override via argv[1]. ---
DEFAULT_MODEL = "results_cluster/results 2/2body_n10000_c8_w0.3_seed1_fb1000_86a58c/model.json"
TEST_FILE = "data/sub_CsZrCl_test.extxyz"

model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

print(f"Loading model:  {model_path}")
model = WeightedLinearModel.from_json(model_path)

print(f"Reading 1 frame: {TEST_FILE}")
atoms = read(TEST_FILE, index=0)
print(f"  n_atoms = {len(atoms)}, pbc = {atoms.pbc}, "
      f"cell volume = {atoms.get_volume():.1f} A^3")

calc = UFCalculator(model)
atoms.calc = calc
# UFCalculator reads energy/forces from info/arrays; clear any stale DFT values
atoms.info.pop("energy", None)
for k in ("fx", "fy", "fz", "force"):
    atoms.arrays.pop(k, None)

print("\n--- energy ---")
t0 = time.time()
e = atoms.get_potential_energy()
print(f"  E = {e:.4f} eV   ({time.time()-t0:.2f} s)   "
      f"= {e/len(atoms)*1000:.2f} meV/atom")

print("\n--- forces ---")
t0 = time.time()
f = atoms.get_forces()
print(f"  forces shape {f.shape}, max |F| = {np.abs(f).max():.4f} eV/A   "
      f"({time.time()-t0:.2f} s)")

print("\n--- STRESS (numerical) ---")
t0 = time.time()
try:
    s = atoms.get_stress()          # Voigt 6-vector, eV/A^3
    dt = time.time() - t0
    GPa = s / 0.006241509074        # eV/A^3 -> GPa
    print(f"  stress (Voigt, eV/A^3): {np.array2string(s, precision=5)}")
    print(f"  stress (Voigt, GPa):    {np.array2string(GPa, precision=4)}")
    print(f"  hydrostatic pressure  = {-GPa[:3].mean():.4f} GPa")
    print(f"  STRESS CALL TOOK {dt:.1f} s  "
          f"(numerical => ~12 energy evals; scales with model cost)")
    print("\nRESULT: get_stress() WORKS end-to-end. NPT / EOS are unblocked.")
except Exception as exc:        # noqa: BLE001  (we want to see any failure)
    print(f"  get_stress() FAILED: {type(exc).__name__}: {exc}")
    print("\nRESULT: stress does NOT work out of the box - needs attention "
          "before NPT/EOS.")
