"""
MoltenSaltCalc MD driver for a fitted UF3 model (the 'fit_uf3' backend).

First real MD-phase script: drives MoltenSaltCalc end-to-end with our UF3 model
(instead of the hand-rolled ASE loop in md_smoke_test.py) and validates the two
cheapest properties against a known melt:

  1. short NPT run  -> equilibrium DENSITY (box relaxes; density read from NPT traj)
  2. short NVT run from the equilibrated frame -> total + partial RDFs (from NVT traj)

The composition is DERIVED FROM THE MODEL'S OWN ELEMENTS (charge-balanced), so the
built melt always matches the model -- pointing at a Cs-Zr-Cl model builds a
Cs-Zr-Cl melt, a K-Rb-Cl model builds K-Rb-Cl, etc. (No more model/system mismatch.)

Run from the repo root (UF3 is CPU-only; always device='cpu', dispersion=None
because the DFT training data already includes D3):

    uv run python scripts/md_moltensaltcalc.py
    uv run python scripts/md_moltensaltcalc.py <model.json> <T_K> <npt_steps> <nvt_steps> [label]

Defaults = the local K-Rb-Cl 2-body model at 1100 K, ~100 atoms. SMOKE-TEST
lengths (a few ps) -> sanity, not a publishable number. Real benchmark: ~100 ps/T
(see md_properties_checklist.md).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.optimize import FIRE
from moltensaltcalc import MoltenSaltAnalyzer, MoltenSaltSimulator

np.random.seed(42)  # reproducible random start

# formal charges for the chloride-melt cations in this project (anion = Cl, -1)
CHARGE = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Zn": 2, "Zr": 4}
ANION = "Cl"
TARGET_ATOMS = 100


def _arg(i, default):  # pylint: disable=redefined-outer-name
    """argv[i] if given AND non-empty, else the default."""
    return sys.argv[i] if len(sys.argv) > i and sys.argv[i] != "" else default


def build_composition(cations: list[str], target: int = TARGET_ATOMS):  # pylint: disable=redefined-outer-name
    """Charge-balanced ~target-atom melt: equal cation counts m, Cl balances charge."""
    # pylint: disable=redefined-outer-name
    charge_per_unit = sum(CHARGE[c] for c in cations)  # Cl per 1-of-each-cation unit
    atoms_per_unit = len(cations) + charge_per_unit
    m = max(1, round(target / atoms_per_unit))
    n_cations = [m] * len(cations)
    n_cl = m * charge_per_unit
    return n_cations, n_cl


# ---- config ----
MODEL = _arg(1, "results/2body_ClKRb_n1000_c8_w0.3_seed42_377d4e/model.json")
T_K = float(_arg(2, 1100.0))
NPT_STEPS = int(_arg(3, 3000))
NVT_STEPS = int(_arg(4, 2000))
BAROSTAT = _arg(6, "mtknpt")  # mtknpt (rigorous) | nptberendsen (crude)
_TARGET_OVERRIDE = _arg(7, "")  # optional atom count override for finite-size check
_CATION_OVERRIDE = _arg(8, "")  # optional comma-sep cation subset, e.g. "Zr" for pure ZrCl4
DENSITY_GUESS = 2.5  # g/cm^3, rough starting box; NPT relaxes it
DT_FS = 1.0
# MoltenSaltCalc simulator parameters, pinned to the documented defaults
# (https://moltensaltcalc.readthedocs.io/en/latest/Simulator_API/); steps are CLI args.
TAUT_FS = 100.0  # NPT temperature coupling time (fs)
TAUP_FS = 1000.0  # NPT pressure coupling time (fs)
COMPRESSIBILITY = 5e-6  # 1/bar
PRESSURE_BAR = 1.01325  # ~1 atm
TCHAIN = 3  # MTK thermostat chain length
PCHAIN = 3  # MTK barostat chain length
TLOOP = 1  # MTK thermostat sub-steps
PLOOP = 1  # MTK barostat sub-steps
TDAMP_FS = 100.0  # NVT thermostat damping time (fs)
NVT_DYN = "nosehoover"
WRITE_EVERY = 10
PRINT_EVERY = 200

# ---- 1. simulator with the UF3 model ----
sim = MoltenSaltSimulator(
    model_name="fit_uf3",
    model_parameters={"model_path": MODEL},
    device="cpu",
    dispersion=None,  # DFT training data already has D3
)

# ---- derive the composition from the MODEL's elements (keeps system == model) ----
model_elements = list(sim.calc.elements)  # model's element symbols, exposed by the UF3 wrapper
cations = [e for e in model_elements if e != ANION]
if ANION not in model_elements:
    raise SystemExit(f"Model elements {model_elements} do not include the anion '{ANION}'.")
if _CATION_OVERRIDE:
    wanted = [c.strip() for c in _CATION_OVERRIDE.split(",") if c.strip()]
    missing = [c for c in wanted if c not in cations]
    if missing:
        raise SystemExit(f"Requested cations {missing} not in model elements {cations}.")
    cations = wanted
    print(f"[cation override] restricting melt to {cations} (subset of model elements)")
unknown = [c for c in cations if c not in CHARGE]
if unknown:
    raise SystemExit(f"No formal charge on record for {unknown}; add them to CHARGE.")
TARGET = int(_TARGET_OVERRIDE) if _TARGET_OVERRIDE else TARGET_ATOMS
n_cations, n_cl = build_composition(cations, target=TARGET)

LABEL = _arg(5, "".join(cations) + ANION + f"_{int(T_K)}K")
OUT = Path("results") / f"md_{LABEL}"
OUT.mkdir(parents=True, exist_ok=True)
NPT_TRAJ, NVT_TRAJ = str(OUT / "npt.traj"), str(OUT / "nvt.traj")

print(f"model : {MODEL}")
print(
    f"system: {dict(zip(cations, n_cations, strict=False))} + {{'{ANION}': {n_cl}}}  "
    f"(= {sum(n_cations) + n_cl} atoms, charge-balanced)"
)
print(f"T = {T_K:.0f} K,  dt = {DT_FS} fs,  NPT = {NPT_STEPS}, NVT = {NVT_STEPS} steps")
print(f"out   : {OUT}/\n")

# ---- 2. build the melt ----
atoms = sim.build_system(
    salt_anion=[ANION],
    salt_cation=cations,
    n_anions=[n_cl],
    n_cations=n_cations,
    density_guess=DENSITY_GUESS,
    lattice="random",
)
print(f"built {len(atoms)} atoms, initial box = {atoms.get_volume():.1f} A^3\n")

# ---- relax the random start: remove bad contacts so dynamics starts sane ----
# (a random packing has near-overlaps; releasing that energy is what spikes T.
#  A short FIRE minimisation relieves the worst contacts without freezing the melt.)
print("--- minimise (relax overlaps) ---")
atoms.calc = sim.calc
e0 = atoms.get_potential_energy()
FIRE(atoms, logfile=str(OUT / "min.log")).run(fmax=1.0, steps=200)
print(f"energy {e0:.1f} -> {atoms.get_potential_energy():.1f} eV  (worst contacts relieved)\n")

t0 = time.time()

# ---- 3. NPT: relax the box -> equilibrium density ----
print("--- NPT (equilibrium density) ---")
atoms_eq = sim.run_npt_simulation(
    atoms,
    T=T_K,
    steps=NPT_STEPS,
    timestep_fs=DT_FS,
    npt_dyn=BAROSTAT,
    taut_fs=TAUT_FS,
    taup_fs=TAUP_FS,
    compressibility_per_bar=COMPRESSIBILITY,
    pressure_bar=PRESSURE_BAR,
    tchain=TCHAIN,
    pchain=PCHAIN,
    tloop=TLOOP,
    ploop=PLOOP,
    write_interval=WRITE_EVERY,
    print_interval=PRINT_EVERY,
    traj_file=NPT_TRAJ,
    logfile=str(OUT / "npt_run.log"),
)

# ---- 4. NVT from the equilibrated frame -> RDF ----
print("\n--- NVT (RDF, at the relaxed box) ---")
sim.run_nvt_simulation(
    atoms_eq,
    T=T_K,
    steps=NVT_STEPS,
    timestep_fs=DT_FS,
    nvt_dyn=NVT_DYN,
    tdamp_fs=TDAMP_FS,
    write_interval=WRITE_EVERY,
    print_interval=PRINT_EVERY,
    traj_file=NVT_TRAJ,
    logfile=str(OUT / "nvt_run.log"),
)

wall = time.time() - t0
print(f"\nMD done in {wall:.0f} s ({wall / (NPT_STEPS + NVT_STEPS) * 1000:.0f} ms/step)\n")

# ---- 5. analyse: density (NPT) + RDF (NVT) ----
ana = MoltenSaltAnalyzer(
    traj_files_npt=NPT_TRAJ,
    temperatures_npt=[T_K],
    traj_files_nvt=NVT_TRAJ,
    temperatures_nvt=[T_K],
    timestep_fs=DT_FS,
)

rho = ana.compute_eq_density(T=T_K, eq_fraction=0.5)
# equilibration check: has the density stopped drifting? compare last 25% vs the 25% before it
rho_last25 = ana.compute_eq_density(T=T_K, eq_fraction=0.25)
dens_t, _ = ana.compute_density_vs_time(T=T_K)

_n = len(dens_t)
_q = max(1, _n // 4)
rho_prev25 = float(np.mean(dens_t[_n - 2 * _q : _n - _q]))
drift = rho_last25 - rho_prev25
print("=== DENSITY ===")
print(f"  equilibrium density (NPT, last 50%) = {rho:.3f} g/cm^3")
print(f"  last 25% = {rho_last25:.3f} vs preceding 25% = {rho_prev25:.3f}  ->  drift = {drift:+.3f} g/cm^3")
print(f"  barostat = {BAROSTAT}, atoms = {len(atoms)}, NPT steps = {NPT_STEPS}")
print("  (|drift| < ~0.02 = equilibrated; a large negative drift = box still expanding, was over-compressed)")
print("  (molten chlorides sit ~1.5-3.0 g/cm^3; compare to literature for this melt)")

print("\n=== RDF ===")
# adaptive rmax: must be < half the smallest PERPENDICULAR cell width (non-cubic cells)
_cell = np.array(atoms_eq.cell)
_vol = abs(atoms_eq.get_volume())
_widths = [_vol / np.linalg.norm(np.cross(_cell[(i + 1) % 3], _cell[(i + 2) % 3])) for i in range(3)]
_rmax = min(6.0, 0.49 * min(_widths))
print(f"  (RDF rmax = {_rmax:.2f} A; min perpendicular cell width = {min(_widths):.2f} A)")
rdf = ana.compute_rdf(T=T_K, rmax=_rmax, nbins=100)
summary = {
    "model": MODEL,
    "T_K": T_K,
    "npt_steps": NPT_STEPS,
    "nvt_steps": NVT_STEPS,
    "n_atoms": len(atoms),
    "composition": dict(zip(cations, n_cations, strict=False)) | {ANION: n_cl},
    "eq_density_g_cm3": float(rho),
    "rdf_first_peak": {},
}
RMIN = 1.8  # A: below this is the r->0 normalisation spike, not real structure
for pair, (dists, g, _std) in rdf.items():
    d = np.asarray(dists)
    gg = np.asarray(g)
    m = d >= RMIN
    i = int(np.where(m)[0][np.argmax(gg[m])])  # first real peak, artifact excluded
    label = f"{pair[0]}-{pair[1]}" if isinstance(pair, tuple) else str(pair)
    print(f"  {label:10s} first peak at {dists[i]:.2f} A  (g = {g[i]:.2f})")
    summary["rdf_first_peak"][label] = {"r_ang": float(dists[i]), "g": float(g[i])}
    np.savetxt(OUT / f"rdf_{label}.csv", np.column_stack([dists, g]), delimiter=",", header="r_ang,g_r", comments="")

with open(OUT / "summary.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)
print(f"\nwrote {OUT}/summary.json + rdf_*.csv + trajectories")

# ---- optional RDF plot ----
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    for pair, (dists, g, _std) in rdf.items():
        label = f"{pair[0]}-{pair[1]}" if isinstance(pair, tuple) else str(pair)
        plt.plot(dists, g, lw=1.3, label=label)
    plt.xlabel("r (A)")
    plt.ylabel("g(r)")
    plt.title(f"RDFs, UF3 {LABEL}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "rdf.png", dpi=130)
    print(f"wrote {OUT}/rdf.png")
except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
    print(f"(plot skipped: {exc})")

print("\nDone.")
