"""
Local MD smoke test for the UF3 2-body model (no MoltenSaltCalc needed).

Purpose: BEFORE the MD phase, confirm that UFCalculator can actually drive an
ASE molecular-dynamics run -- i.e. the forces update every step (the ASE-protocol
quirk is harmless), the dynamics stay STABLE (no blow-up / NaN), and the model
produces a physically sane radial distribution function (RDF) for a molten salt.

It runs NVT (fixed volume -> no stress needed) with a Langevin thermostat, starting
from an already-equilibrated liquid frame from the data (so the density is realistic).

Run from the repo root:
    uv run python scripts/md_smoke_test.py
    uv run python scripts/md_smoke_test.py <model.json> <frame_file.extxyz> <T_kelvin> <n_prod>

Defaults: the local Cs-Zr-Cl 2-body model + a Cs-Zr-Cl liquid frame, 1100 K.
2-body forces are ~0.2 s/step at 80 atoms, so ~2000 steps is a few minutes.
"""
from __future__ import annotations

import sys
import time

import numpy as np
from ase import units
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

from uf3.regression.least_squares import WeightedLinearModel
from uf3.forcefield.calculator import UFCalculator

# ---- config (argv overridable) ----
def _arg(i, default):
    """argv[i] if given AND non-empty, else the default."""
    return sys.argv[i] if len(sys.argv) > i and sys.argv[i] != "" else default


MODEL = _arg(1, "results/2body_ClCsZr_n200_c8_w0.3_seed42_927efc/model.json")
FRAME = _arg(2, "data/sub_CsZrCl_test.extxyz")
T_K = float(_arg(3, 1100.0))
N_PROD = int(_arg(4, 1500))
N_EQUIL = 500
DT_FS = 1.0
FRICTION = 0.01  # 1/fs
SAMPLE_EVERY = 10   # collect a frame for the RDF every N steps
RDF_RMAX = 6.0   # capped below to < half the smallest cell length (MIC limit)
RDF_NBINS = 100

print(f"model: {MODEL}")
print(f"frame: {FRAME}")
print(f"T = {T_K} K,  dt = {DT_FS} fs,  equil = {N_EQUIL}, prod = {N_PROD} steps")

model = WeightedLinearModel.from_json(MODEL)
calc = UFCalculator(model)

atoms = read(FRAME, index=0)
# clean any stored DFT energy/forces so the calc computes fresh
atoms.info.pop("energy", None)
for k in ("fx", "fy", "fz", "force"):
    atoms.arrays.pop(k, None)
atoms.calc = calc
print(f"n_atoms = {len(atoms)}, cell volume = {atoms.get_volume():.1f} A^3, pbc = {atoms.pbc}")
syms = sorted(set(atoms.get_chemical_symbols()))
print(f"elements present: {syms}")

# initial velocities at T
MaxwellBoltzmannDistribution(atoms, temperature_K=T_K)
Stationary(atoms)
ZeroRotation(atoms)

dyn = Langevin(atoms, timestep=DT_FS * units.fs, temperature_K=T_K, friction=FRICTION / units.fs)

# ---- stability monitors ----
pos0 = atoms.get_positions().copy()
history = {"step": [], "T": [], "Epot": []}
blew_up = [False]
t0 = time.time()


def monitor():
    """Record T / Epot and watch for NaN or explosion."""
    e = atoms.get_potential_energy()
    T = atoms.get_temperature()
    f = atoms.get_forces()
    step = len(history["step"]) * 10
    history["step"].append(step)
    history["T"].append(T)
    history["Epot"].append(e / len(atoms) * 1000.0)  # meV/atom
    maxf = np.abs(f).max()
    bad = (not np.isfinite(e)) or (not np.isfinite(f).all()) or (T > 20 * T_K) or (maxf > 50.0)
    if bad:
        blew_up[0] = True
    print(f"  step {step:5d}  T = {T:7.1f} K   Epot = {e/len(atoms)*1000:8.2f} meV/atom   "
          f"max|F| = {maxf:6.2f} eV/A{'   <-- UNSTABLE' if bad else ''}")


dyn.attach(monitor, interval=10)

print("\n--- equilibration ---")
try:
    dyn.run(N_EQUIL)
except Exception as exc:  # noqa: BLE001
    print(f"  MD crashed during equilibration: {type(exc).__name__}: {exc}")
    sys.exit(1)

print("\n--- production (collecting frames for RDF) ---")
traj = []


def collect():
    traj.append(atoms.copy())


dyn.attach(collect, interval=SAMPLE_EVERY)
try:
    dyn.run(N_PROD)
except Exception as exc:  # noqa: BLE001
    print(f"  MD crashed during production: {type(exc).__name__}: {exc}")
    sys.exit(1)

wall = time.time() - t0
nsteps = N_EQUIL + N_PROD
print(f"\nMD finished: {nsteps} steps in {wall:.0f} s  ({wall/nsteps*1000:.0f} ms/step)")

# ---- verdicts ----
Tmean = np.mean(history["T"][len(history["T"]) // 2:])  # 2nd-half average
drift = np.abs(np.array(atoms.get_positions()) - pos0).max()
print("\n=== STABILITY ===")
print(f"  mean T (2nd half) = {Tmean:.1f} K   (target {T_K:.0f} K)")
print(f"  Epot finite: {np.isfinite(history['Epot']).all()}")
print(f"  max atomic displacement from start = {drift:.2f} A  (liquid diffusion is expected)")
if blew_up[0]:
    print("  VERDICT: UNSTABLE -- the 2-body model blew up / NaN. Investigate before MD phase.")
else:
    print("  VERDICT: STABLE -- forces update, thermostat holds, no blow-up. ASE MD works.")

# ---- RDF ----
print("\n=== RDF ===")
try:
    from ase.geometry.analysis import Analysis
    # rmax must be < half the smallest PERPENDICULAR cell width (NOT vector length --
    # the cell is non-orthogonal). width_i = volume / area of the opposite face.
    cell = np.array(atoms.cell)
    vol = abs(atoms.get_volume())
    widths = [vol / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)]
    min_w = float(min(widths))
    ana = Analysis(traj)
    rmax = min(RDF_RMAX, 0.45 * min_w)
    rdf_list = None
    for _ in range(8):
        try:
            rdf_list = ana.get_rdf(rmax, RDF_NBINS, return_dists=True)
            break
        except Exception:  # noqa: BLE001  (CellTooSmall -> shrink and retry)
            rmax -= 0.3
    print(f"  (using rmax = {rmax:.2f} A; min perpendicular width = {min_w:.2f} A)")
    if rdf_list is None:
        raise RuntimeError("RDF failed even after shrinking rmax")
    rdfs = np.array([r for r, _ in rdf_list])
    dists = rdf_list[0][1]
    rdf = rdfs.mean(axis=0)
    # first-peak position
    peak_i = int(np.argmax(rdf))
    print(f"  total RDF first-peak position = {dists[peak_i]:.2f} A  (height {rdf[peak_i]:.2f})")
    print(f"  (sanity: a molten chloride first cation-Cl peak sits ~2.5-3.0 A)")
    # save a plot + csv
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(dists, rdf, lw=1.5)
        plt.xlabel("r (A)"); plt.ylabel("g(r)"); plt.title(f"Total RDF, 2-body MD, {T_K:.0f} K")
        plt.tight_layout(); plt.savefig("results/md_smoke_rdf.png", dpi=130)
        print("  saved results/md_smoke_rdf.png")
    except Exception as exc:  # noqa: BLE001
        print(f"  (plot skipped: {exc})")
    np.savetxt("results/md_smoke_rdf.csv", np.column_stack([dists, rdf]),
               delimiter=",", header="r_ang,g_r", comments="")
    print("  saved results/md_smoke_rdf.csv")
except Exception as exc:  # noqa: BLE001
    print(f"  RDF computation failed: {type(exc).__name__}: {exc}")

print("\nDone.")
