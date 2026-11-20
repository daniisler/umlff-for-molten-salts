"""Quick results check for the FAST 3-body backend on a real (non-Mg) salt.

Mirrors md_moltensaltcalc.py (FIRE -> NPT -> NVT -> density + RDF), but drives the
fast 3-body backend (fit_uf3_fast_3body) by building it directly and setting sim.calc,
the same way md_speed_test_3body.py does. Composition is derived from the model's own
elements, so it always matches the model.

Run from the repo root:
    uv run python scripts/run_check_3body.py results/3body_NaCl/model.json
    uv run python scripts/run_check_3body.py <model.json> <T_K> <npt_steps> <nvt_steps> <target_atoms>

Defaults: 1100 K, 4000 NPT + 2000 NVT (= 6k steps), 50 atoms. A small/short sanity
run (finite-size + short-time caveats apply) -- not a publishable number.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.optimize import FIRE
from fit_uf3_fast_3body import UF3Fast3BodyCalculator  # sibling scripts/ module (on path as a script)
from moltensaltcalc import MoltenSaltAnalyzer, MoltenSaltSimulator
from moltensaltcalc.models.fit_uf3 import ZBLRepulsion
from uf3.forcefield.calculator import UFCalculator
from uf3.regression.least_squares import WeightedLinearModel

np.random.seed(42)

CHARGE = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Zn": 2, "Zr": 4}
ANION = "Cl"


def _arg(i, default):
    return sys.argv[i] if len(sys.argv) > i and sys.argv[i] != "" else default


MODEL = _arg(1, "results/3body_NaCl/model.json")
T_K = float(_arg(2, 1100.0))
NPT_STEPS = int(_arg(3, 4000))
NVT_STEPS = int(_arg(4, 2000))
TARGET = int(_arg(5, 50))
DT_FS = 1.0
DENSITY_GUESS = 2.5

# --- build the fast 3-body calc directly, attach to a (model-less) simulator ---
fast = UF3Fast3BodyCalculator(
    UFCalculator(WeightedLinearModel.from_json(MODEL)),
    wall=ZBLRepulsion(r_in=4.0, r_out=5.0),
)
sim = MoltenSaltSimulator(device="cpu", dispersion=None)
sim.calc = fast

cations = [e for e in fast.elements if e != ANION]
charge_per_unit = sum(CHARGE[c] for c in cations)
atoms_per_unit = len(cations) + charge_per_unit
m = max(1, round(TARGET / atoms_per_unit))
n_cations = [m] * len(cations)
n_cl = m * charge_per_unit

LABEL = "".join(cations) + ANION + f"_{int(T_K)}K_fast3b"
OUT = Path("results") / f"check_{LABEL}"
OUT.mkdir(parents=True, exist_ok=True)
NPT_TRAJ, NVT_TRAJ = str(OUT / "npt.traj"), str(OUT / "nvt.traj")

print(f"model : {MODEL}  (fast 3-body backend)")
print(f"system: {dict(zip(cations, n_cations, strict=False))} + {{'{ANION}': {n_cl}}} = {sum(n_cations) + n_cl} atoms")
print(f"T = {T_K:.0f} K, dt = {DT_FS} fs, NPT = {NPT_STEPS}, NVT = {NVT_STEPS}\nout   : {OUT}/\n")

atoms = sim.build_system(
    salt_anion=[ANION],
    salt_cation=cations,
    n_anions=[n_cl],
    n_cations=n_cations,
    density_guess=DENSITY_GUESS,
    lattice="random",
)
atoms.calc = sim.calc
e0 = atoms.get_potential_energy()
FIRE(atoms, logfile=str(OUT / "min.log")).run(fmax=1.0, steps=200)
print(f"minimise: {e0:.1f} -> {atoms.get_potential_energy():.1f} eV\n")

t0 = time.time()
print("--- NPT (density) ---")
atoms_eq = sim.run_npt_simulation(
    atoms,
    T=T_K,
    steps=NPT_STEPS,
    timestep_fs=DT_FS,
    npt_dyn="mtknpt",
    taut_fs=100.0,
    taup_fs=1000.0,
    compressibility_per_bar=5e-6,
    pressure_bar=1.01325,
    tchain=3,
    pchain=3,
    tloop=1,
    ploop=1,
    write_interval=10,
    print_interval=500,
    traj_file=NPT_TRAJ,
    logfile=str(OUT / "npt_run.log"),
)
print("\n--- NVT (RDF) ---")
sim.run_nvt_simulation(
    atoms_eq,
    T=T_K,
    steps=NVT_STEPS,
    timestep_fs=DT_FS,
    nvt_dyn="nosehoover",
    tdamp_fs=100.0,
    write_interval=10,
    print_interval=500,
    traj_file=NVT_TRAJ,
    logfile=str(OUT / "nvt_run.log"),
)
wall = time.time() - t0
print(f"\nMD done in {wall:.0f} s ({wall / (NPT_STEPS + NVT_STEPS) * 1000:.0f} ms/step)\n")

ana = MoltenSaltAnalyzer(
    traj_files_npt=NPT_TRAJ, temperatures_npt=[T_K], traj_files_nvt=NVT_TRAJ, temperatures_nvt=[T_K], timestep_fs=DT_FS
)
rho = ana.compute_eq_density(T=T_K, eq_fraction=0.5)
rho_last25 = ana.compute_eq_density(T=T_K, eq_fraction=0.25)
dens_t, _ = ana.compute_density_vs_time(T=T_K)
_n = len(dens_t)
_q = max(1, _n // 4)
rho_prev25 = float(np.mean(dens_t[_n - 2 * _q : _n - _q]))
print("=== DENSITY ===")
print(f"  eq density (last 50%) = {rho:.3f} g/cm^3   | drift(last25 vs prev25) = {rho_last25 - rho_prev25:+.3f}")
print(
    f"  atoms = {len(atoms)}, NPT steps = {NPT_STEPS}  (|drift|<~0.02 = equilibrated; small box = finite-size caveat)"
)

print("\n=== RDF ===")
_cell = np.array(atoms_eq.cell)
_vol = abs(atoms_eq.get_volume())
_widths = [_vol / np.linalg.norm(np.cross(_cell[(i + 1) % 3], _cell[(i + 2) % 3])) for i in range(3)]
_rmax = min(6.0, 0.49 * min(_widths))
rdf = ana.compute_rdf(T=T_K, rmax=_rmax, nbins=100)
summary = {"model": MODEL, "T_K": T_K, "n_atoms": len(atoms), "eq_density_g_cm3": float(rho), "rdf_first_peak": {}}
RMIN = 1.8
for pair, (dists, g, _std) in rdf.items():
    d = np.asarray(dists)
    gg = np.asarray(g)
    msk = d >= RMIN
    idx = int(np.where(msk)[0][np.argmax(gg[msk])])
    label = f"{pair[0]}-{pair[1]}" if isinstance(pair, tuple) else str(pair)
    print(f"  {label:10s} first peak at {dists[idx]:.2f} A  (g = {g[idx]:.2f})")
    summary["rdf_first_peak"][label] = {"r_ang": float(dists[idx]), "g": float(g[idx])}
with open(OUT / "summary.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)
print(f"\nwrote {OUT}/summary.json + trajectories")
