"""Speed + equivalence test: fit_uf3 (old) vs fit_uf3_fast (Daniel's) backend.

The fast calculator isn't registered in the moltensaltcalc package yet (the simulator
discovers models by importing moltensaltcalc.models.<name>), so we build it DIRECTLY
and swap it into a working fit_uf3 simulator -- run_npt_simulation attaches sim.calc to
the atoms at call time, so the timing is otherwise identical to md_moltensaltcalc.py
(same melt, same mtknpt NPT, stress every step).

Run from the repo root:

    uv run python scripts/md_speed_test.py <model.json> <T_K> <npt_steps> <atoms> <which>

  <which> = both (default) | fast | old
  defaults: CaMgCl 2-body zbl model, 1100 K, 150 NPT steps, ~200 atoms, both.

'old' is deliberately slow (numerical stress ~12 energy evals/step); keep npt_steps
modest when running both. The fast/old RATIO is the headline and is ~size-independent.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from ase.optimize import FIRE
from moltensaltcalc import MoltenSaltSimulator
from moltensaltcalc.models.fit_uf3 import ZBLRepulsion
from uf3.forcefield.calculator import UFCalculator
from uf3.regression.least_squares import WeightedLinearModel

from fit_uf3_fast import UF3FastCalculator  # noqa: F401

try:
    import matscipy  # noqa: F401
    _HAS_MATSCIPY = True
except ImportError:
    _HAS_MATSCIPY = False

CHARGE = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Zn": 2, "Zr": 4}
ANION = "Cl"


def _arg(i, default):
    return sys.argv[i] if len(sys.argv) > i and sys.argv[i] != "" else default


def build_composition(cations, target):
    charge_per_unit = sum(CHARGE[c] for c in cations)
    atoms_per_unit = len(cations) + charge_per_unit
    m = max(1, round(target / atoms_per_unit))
    return [m] * len(cations), m * charge_per_unit


MODEL = _arg(1, "results/2body_CaMgCl_zbl/model.json")
T_K = float(_arg(2, 1100.0))
NPT_STEPS = int(_arg(3, 150))
TARGET = int(_arg(4, 200))
WHICH = _arg(5, "both")

NPT_KW = dict(
    T=T_K, timestep_fs=1.0, npt_dyn="mtknpt",
    taut_fs=100.0, taup_fs=1000.0, compressibility_per_bar=5e-6, pressure_bar=1.01325,
    tchain=3, pchain=3, tloop=1, ploop=1,
    print_interval=10_000, write_interval=10_000, print_status=False, logfile=None,
)

print(f"model      : {MODEL}")
print(f"matscipy   : {'yes (C neighbour list)' if _HAS_MATSCIPY else 'NO -> ASE fallback (slower)'}")

# one working simulator on the OLD backend; its calc is the reference (SumCalculator)
sim = MoltenSaltSimulator(
    model_name="fit_uf3", model_parameters={"model_path": MODEL}, device="cpu", dispersion=None,
)
old_calc = sim.calc

# fast backend built directly (same model.json, same ZBL wall r_in=4/r_out=5 as the fit)
fast_calc = UF3FastCalculator(
    UFCalculator(WeightedLinearModel.from_json(MODEL)),
    wall=ZBLRepulsion(r_in=4.0, r_out=5.0),
)

# melt from the model's own elements (same as the driver)
elements = list(old_calc.elements)
cations = [e for e in elements if e != ANION]
n_cations, n_cl = build_composition(cations, TARGET)
atoms = sim.build_system(
    salt_anion=[ANION], salt_cation=cations, n_anions=[n_cl], n_cations=n_cations,
    density_guess=2.5, lattice="random",
)
print(f"system     : {dict(zip(cations, n_cations, strict=False))} + {{'{ANION}': {n_cl}}} = {len(atoms)} atoms")
print(f"T = {T_K:.0f} K, NPT steps = {NPT_STEPS}\n")

# relax overlaps once so both backends start from the identical config
atoms.calc = fast_calc
FIRE(atoms, logfile=None).run(fmax=1.0, steps=200)

# ---- single-point agreement (fast vs old) ----
a1 = atoms.copy(); a1.calc = fast_calc
a2 = atoms.copy(); a2.calc = old_calc
e1, f1, s1 = a1.get_potential_energy(), a1.get_forces(), a1.get_stress()
e2, f2, s2 = a2.get_potential_energy(), a2.get_forces(), a2.get_stress()
print("--- agreement (fast vs old), single point ---")
print(f"  d(energy)/atom = {abs(e1 - e2) / len(atoms):.2e} eV")
print(f"  max|dF|        = {np.abs(f1 - f2).max():.2e} eV/A")
print(f"  max|dstress|   = {np.abs(s1 - s2).max():.2e} eV/A^3\n")


def time_npt(calc, label):
    sim.calc = calc
    a = atoms.copy()
    a.calc = calc  # run_npt_simulation expects the calc already attached (as the driver does)
    t0 = time.perf_counter()
    sim.run_npt_simulation(a, steps=NPT_STEPS, traj_file=f"results/_speedtest_{label}.traj", **NPT_KW)
    wall = time.perf_counter() - t0
    print(f"  {label:4s}: {wall:6.1f} s  ->  {wall / NPT_STEPS * 1000:7.1f} ms/step   {NPT_STEPS * 0.06 / wall:7.2f} ps/min")
    return wall / NPT_STEPS * 1000


print(f"--- NPT timing ({NPT_STEPS} steps, mtknpt, stress every step) ---")
ms_fast = time_npt(fast_calc, "fast") if WHICH in ("both", "fast") else None
ms_old = time_npt(old_calc, "old") if WHICH in ("both", "old") else None
if ms_fast and ms_old:
    print(f"\n  speedup: {ms_old / ms_fast:.1f}x faster")
