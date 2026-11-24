"""
SuperSalt MD driver (local validation) — the same pipeline as md_moltensaltcalc.py,
but driven by the SuperSalt MACE model instead of our UF3 fast backend.

Purpose: a LOCAL sanity check before committing cluster time. Run one or two pure
salts through MoltenSaltCalc with SuperSalt and confirm the density matches the
SuperSalt paper (NaCl 1200 K -> ~1.549, MgCl2 1200 K -> ~1.594). If it does, the
MACE-backend pipeline + our setup are validated, and the full 11-salt x 3-T matrix
can be handed to the cluster GPUs.

SuperSalt is a MACE model, so it is loaded as a MACECalculator exactly the way the
reproduction (force RMSE 20.3 vs the paper's 19.6) loaded it — NOT through the
mace_mp foundation-model path, which assumes a pretrained MP head. Dispersion is
left OFF (plain MACECalculator, dispersion=None): the SuperSalt training labels
already include D3, the same reason we pass dispersion=None for UF3. If the density
comes out systematically off, D3 handling is the first thing to revisit.

SuperSalt is heavy on CPU (~100x slower per step than the UF3 fast backend), so keep
this SMALL and SHORT — this validates the setup, it is not the production number.
The full matrix belongs on the cluster.

Run from the repo root:

    uv run python scripts/md_supersalt.py Na                 # NaCl at 1200 K, ~100 atoms
    uv run python scripts/md_supersalt.py Mg --temp 1200
    uv run python scripts/md_supersalt.py Na --npt 8000 --nvt 3000 --atoms 150

Needs mace-torch (already in the env) and data/SuperSalt-swa.model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase.calculators.mixing import SumCalculator
from ase.optimize import FIRE
from mace.calculators import MACECalculator
from moltensaltcalc import MoltenSaltAnalyzer, MoltenSaltSimulator
from moltensaltcalc.models.fit_uf3 import ZBLRepulsion

np.random.seed(42)  # reproducible random start

# formal charges for the chloride-melt cations (anion = Cl, -1)
CHARGE = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Zn": 2, "Zr": 4}
ANION = "Cl"

# SuperSalt paper reference densities (g/cm^3), for an at-a-glance check after the run.
# key = (cation, T_K) -> (SuperSalt, AIMD, experiment)
REF = {
    ("Na", 1200): (1.549, 1.566, 1.488),
    ("Na", 1500): (1.422, 1.44, 1.325),
    ("Mg", 1200): (1.594, 1.593, 1.622),
    ("Mg", 1500): (1.509, 1.516, 1.541),
    ("Rb", 1100): (2.101, 2.134, 2.149),
    ("Rb", 1200): (2.034, 2.065, 2.061),
    ("Ba", 1250): (2.907, 2.875, 3.174),
    ("Ba", 1450): (2.752, 2.777, 3.027),
}


def build_composition(cations: list[str], target: int, fracs: dict | None = None):
    """Charge-balanced ~target-atom melt. fracs=None -> equal cation counts (pure/equimolar);
    fracs given -> a BLEND at salt mole fractions (e.g. {'Na':0.57,'Mg':0.43}), normalised,
    scaled to target, Cl set to exactly charge-balance the rounded cation counts."""
    if fracs is None:
        charge_per_unit = sum(CHARGE[c] for c in cations)
        atoms_per_unit = len(cations) + charge_per_unit
        m = max(1, round(target / atoms_per_unit))
        return [m] * len(cations), m * charge_per_unit
    tot = sum(fracs[c] for c in cations)
    f = {c: fracs[c] / tot for c in cations}
    atoms_per_unit = 1.0 + sum(f[c] * CHARGE[c] for c in cations)
    m = max(1, round(target / atoms_per_unit))
    n_cations = [max(1, round(f[c] * m)) for c in cations]
    n_cl = sum(n * CHARGE[c] for n, c in zip(n_cations, cations, strict=False))
    return n_cations, n_cl


def main():
    ap = argparse.ArgumentParser(description="Local SuperSalt MD validation (density + RDF).")
    ap.add_argument("cations", help="cation(s): 'Na' (NaCl), 'Na,Mg' (equal counts), or a MOLE-FRACTION "
                                     "blend 'Na:0.57,Mg:0.43' (0.57 NaCl . 0.43 MgCl2; colon = blend mode)")
    ap.add_argument("--temp", type=float, default=1200.0, help="temperature in K (default 1200)")
    ap.add_argument("--npt", type=int, default=5000, help="NPT steps (default 5000)")
    ap.add_argument("--nvt", type=int, default=2000, help="NVT steps (default 2000)")
    ap.add_argument("--atoms", type=int, default=100, help="target atom count (default 100 — keep small, MACE is slow)")
    ap.add_argument("--dt", type=float, default=1.0, help="timestep in fs (use 0.5 for Mg/Ca/Zn)")
    ap.add_argument("--density-guess", type=float, default=1.5,
                    help="starting-box density g/cm^3 (default 1.5 — deliberately DILUTE so the random pack "
                         "has no close contacts; NPT compresses to the real density). SuperSalt/MACE has no "
                         "ZBL wall, so a too-dense random start produces huge forces and explodes.")
    ap.add_argument("--model", default="data/SuperSalt-swa.model", help="path to the SuperSalt .model file")
    ap.add_argument("--device", default="cpu", help="cpu (default) or mps (Apple GPU, experimental) or cuda")
    ap.add_argument("--barostat", default="nptberendsen",
                    help="nptberendsen (default, robust) | mtknpt (rigorous but can blow up from a cold start)")
    ap.add_argument("--eq", type=int, default=1500, help="NVT equilibration steps before NPT (default 1500)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    blend_fracs = None
    if ":" in args.cations:  # BLEND mode: "Na:0.57,Mg:0.43"
        blend_fracs = {}
        for part in args.cations.split(","):
            el, frac = part.split(":")
            blend_fracs[el.strip()] = float(frac)
        cations = list(blend_fracs)
    else:
        cations = [c.strip() for c in args.cations.split(",") if c.strip()]
    unknown = [c for c in cations if c not in CHARGE]
    if unknown:
        raise SystemExit(f"No formal charge on record for {unknown}.")
    T_K = args.temp
    density_guess = args.density_guess  # dilute start; NPT compresses to the real density (MACE has no ZBL wall)

    # ---- simulator + SuperSalt MACE calculator (loaded like the reproduction) ----
    # SuperSalt (MACE) has NO short-range repulsive wall, so a random pack or a minimisation can drive
    # two atoms together into a region it never trained on -> energy keeps dropping, force -> inf -> NaN.
    # Add a SHORT-range ZBL safety wall: full below r_in=1.0 A, off above r_out=1.8 A — safely under the
    # shortest real cation-Cl bond (~2.25 A for Zn-Cl), so it catches overlaps WITHOUT touching the
    # physical liquid structure. Combined via SumCalculator exactly as the UF3 backend combines UF3+ZBL.
    sim = MoltenSaltSimulator(device=args.device, dispersion=None)
    _mace = MACECalculator(model_paths=args.model, device=args.device, default_dtype="float64")
    _wall = ZBLRepulsion(r_in=1.0, r_out=1.8)
    sim.calc = SumCalculator([_mace, _wall])
    print(f"[backend] SuperSalt MACE + short-range ZBL safety wall (<1.8 A): {args.model}")

    n_cations, n_cl = build_composition(cations, target=args.atoms, fracs=blend_fracs)
    label = args.label or ("".join(cations) + ANION + f"_{int(T_K)}K")
    out = Path("results") / f"md_supersalt_{label}"
    out.mkdir(parents=True, exist_ok=True)
    npt_traj, nvt_traj = str(out / "npt.traj"), str(out / "nvt.traj")

    print(f"system: {dict(zip(cations, n_cations, strict=False))} + {{'{ANION}': {n_cl}}}  "
          f"(= {sum(n_cations) + n_cl} atoms, charge-balanced)")
    print(f"T = {T_K:.0f} K,  dt = {args.dt} fs,  NPT = {args.npt}, NVT = {args.nvt} steps")
    print(f"out   : {out}/\n")

    # ---- build the melt ----
    atoms = sim.build_system(
        salt_anion=[ANION], salt_cation=cations,
        n_anions=[n_cl], n_cations=n_cations,
        density_guess=density_guess, lattice="random",
    )
    print(f"built {len(atoms)} atoms, initial box = {atoms.get_volume():.1f} A^3\n")

    # ---- relax the random start ----
    print("--- minimise (relax overlaps) ---")
    atoms.calc = sim.calc
    e0 = atoms.get_potential_energy()
    FIRE(atoms, logfile=str(out / "min.log")).run(fmax=0.5, steps=300)
    print(f"energy {e0:.1f} -> {atoms.get_potential_energy():.1f} eV\n")

    # ---- NVT equilibration: thermalise the random melt at a fixed box BEFORE the barostat ----
    # A random pack releases a burst of energy as it settles into a liquid; if that hits the NPT
    # barostat directly, the temperature spikes (~4000 K) and the MTK Nose-Hoover chain overflows.
    # Letting NVT absorb it first means the barostat starts from an already-thermalised melt.
    print("--- NVT equilibration (thermalise at fixed box) ---")
    atoms = sim.run_nvt_simulation(
        atoms, T=T_K, steps=args.eq, timestep_fs=args.dt, nvt_dyn="nosehoover", tdamp_fs=100.0,
        write_interval=50, print_interval=200,
        traj_file=str(out / "eq.traj"), logfile=str(out / "eq_run.log"),
    ) or atoms

    t0 = time.time()

    # ---- NPT: equilibrium density ----
    print("--- NPT (equilibrium density) ---")
    atoms_eq = sim.run_npt_simulation(
        atoms, T=T_K, steps=args.npt, timestep_fs=args.dt, npt_dyn=args.barostat,
        taut_fs=100.0, taup_fs=1000.0, compressibility_per_bar=5e-6, pressure_bar=1.01325,
        tchain=3, pchain=3, tloop=1, ploop=1,
        write_interval=10, print_interval=200,
        traj_file=npt_traj, logfile=str(out / "npt_run.log"),
    )

    # ---- NVT from the equilibrated frame -> RDF ----
    print("\n--- NVT (RDF, at the relaxed box) ---")
    sim.run_nvt_simulation(
        atoms_eq, T=T_K, steps=args.nvt, timestep_fs=args.dt, nvt_dyn="nosehoover", tdamp_fs=100.0,
        write_interval=10, print_interval=200,
        traj_file=nvt_traj, logfile=str(out / "nvt_run.log"),
    )

    wall = time.time() - t0
    print(f"\nMD done in {wall:.0f} s ({wall / (args.npt + args.nvt) * 1000:.0f} ms/step)\n")

    # ---- analyse: density (NPT) + RDF (NVT) ----
    ana = MoltenSaltAnalyzer(
        traj_files_npt=npt_traj, temperatures_npt=[T_K],
        traj_files_nvt=nvt_traj, temperatures_nvt=[T_K], timestep_fs=args.dt,
    )
    rho = ana.compute_eq_density(T=T_K, eq_fraction=0.5)
    rho_last25 = ana.compute_eq_density(T=T_K, eq_fraction=0.25)
    dens_t, _ = ana.compute_density_vs_time(T=T_K)
    _n = len(dens_t); _q = max(1, _n // 4)
    rho_prev25 = float(np.mean(dens_t[_n - 2 * _q: _n - _q]))
    drift = rho_last25 - rho_prev25

    print("=== DENSITY ===")
    print(f"  equilibrium density (NPT, last 50%) = {rho:.3f} g/cm^3")
    print(f"  last 25% = {rho_last25:.3f} vs preceding 25% = {rho_prev25:.3f}  ->  drift = {drift:+.3f}")
    ref = REF.get((cations[0], int(T_K))) if len(cations) == 1 else None
    if ref:
        ss, aimd, exp = ref
        print(f"  --- SuperSalt paper, {cations[0]}Cl{'' if CHARGE[cations[0]] == 1 else '2' if CHARGE[cations[0]] == 2 else '4'} @ {int(T_K)} K:")
        print(f"      SuperSalt {ss:.3f}  |  AIMD {aimd:.3f}  |  exp {exp:.3f}")
        print(f"      -> local SuperSalt-MD {rho:.3f} vs paper {ss:.3f}  (Δ {100 * (rho - ss) / ss:+.1f}%)")
        print("      a few % and matching sign = pipeline validated; a large offset -> check D3 / dtype / box size")

    print("\n=== RDF ===")
    _cell = np.array(atoms_eq.cell); _vol = abs(atoms_eq.get_volume())
    _widths = [_vol / np.linalg.norm(np.cross(_cell[(i + 1) % 3], _cell[(i + 2) % 3])) for i in range(3)]
    _rmax = min(6.0, 0.49 * min(_widths))
    rdf = ana.compute_rdf(T=T_K, rmax=_rmax, nbins=100)
    summary = {
        "model": args.model, "T_K": T_K, "npt_steps": args.npt, "nvt_steps": args.nvt,
        "n_atoms": len(atoms), "composition": dict(zip(cations, n_cations, strict=False)) | {ANION: n_cl},
        "eq_density_g_cm3": float(rho), "rdf_first_peak": {},
    }
    rmin = 1.8
    for pair, (dists, g, _std) in rdf.items():
        d = np.asarray(dists); gg = np.asarray(g); mask = d >= rmin
        i = int(np.where(mask)[0][np.argmax(gg[mask])])
        plabel = f"{pair[0]}-{pair[1]}" if isinstance(pair, tuple) else str(pair)
        print(f"  {plabel:10s} first peak at {dists[i]:.2f} A  (g = {g[i]:.2f})")
        summary["rdf_first_peak"][plabel] = {"r_ang": float(dists[i]), "g": float(g[i])}

    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out}/summary.json + trajectories\nDone.")


if __name__ == "__main__":
    main()
