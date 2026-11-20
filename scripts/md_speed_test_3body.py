"""3-body MD speed test — OLD fit_uf3 vs the NEW fast 3-body backend.

The old `fit_uf3` backend runs 3-body MD with uf3's generic routines (forces dominated
by np.unique triplet sorting) + a NUMERICAL virial stress (~12 energy evals/step). The
new `fit_uf3_fast_3body` backend reuses uf3's trio splines but with one neighbour-list
triplet enumeration + an analytic virial (energy/forces/stress validated against the old
backend; its forces are energy-consistent, where uf3's forces_3b is not). This times old
vs fast 3-body (and the fast 2-body backend for reference) on the SAME MgCl2 melt.

  uv run python scripts/md_speed_test_3body.py [atoms] [n3] [n2]
     atoms : system size (default 100 — keep small, the OLD 3-body is slow)
     n3    : NPT steps timed on the OLD 3-body backend   (default 10)
     n2    : NPT steps timed on the fast backends        (default 100)

Models: 3-body = results/3body_MgCl_speedtest/model.json ; 2-body = results/2body_CaMgCl_zbl/model.json
"""

from __future__ import annotations

import sys
import time

from ase.optimize import FIRE
from fit_uf3_fast import UF3FastCalculator
from fit_uf3_fast_3body import UF3Fast3BodyCalculator
from moltensaltcalc import MoltenSaltSimulator
from moltensaltcalc.models.fit_uf3 import ZBLRepulsion
from uf3.forcefield.calculator import UFCalculator
from uf3.regression.least_squares import WeightedLinearModel

MODEL3 = "results/3body_MgCl_speedtest/model.json"
MODEL2 = "results/2body_CaMgCl_zbl/model.json"
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 100
N3 = int(sys.argv[2]) if len(sys.argv) > 2 else 10
N2 = int(sys.argv[3]) if len(sys.argv) > 3 else 100
PROD_STEPS = 120_000  # a production sweep = 60k NPT + 60k NVT

NPT_KW = {
    "timestep_fs": 1.0,
    "npt_dyn": "mtknpt",
    "taut_fs": 100.0,
    "taup_fs": 1000.0,
    "compressibility_per_bar": 5e-6,
    "pressure_bar": 1.01325,
    "tchain": 3,
    "pchain": 3,
    "tloop": 1,
    "ploop": 1,
    "print_interval": 10_000,
    "write_interval": 10_000,
    "print_status": False,
    "logfile": None,
    "T": 1100.0,
}

# 3-body sim (old backend) + its calc
sim3 = MoltenSaltSimulator(model_name="fit_uf3", model_parameters={"model_path": MODEL3}, device="cpu", dispersion=None)
old3 = sim3.calc
# fast 3-body calc, built directly (same MODEL3 as the old backend)
fast3 = UF3Fast3BodyCalculator(
    UFCalculator(WeightedLinearModel.from_json(MODEL3)), wall=ZBLRepulsion(r_in=4.0, r_out=5.0)
)
# 2-body fast calc, built directly (reference)
fast2 = UF3FastCalculator(UFCalculator(WeightedLinearModel.from_json(MODEL2)), wall=ZBLRepulsion(r_in=4.0, r_out=5.0))

# build a pure MgCl2 melt (Mg133-style ratio), size = TARGET
m = max(1, round(TARGET / 3))
n_cl = 2 * m
atoms = sim3.build_system(
    salt_anion=["Cl"], salt_cation=["Mg"], n_anions=[n_cl], n_cations=[m], density_guess=2.0, lattice="random"
)
print(f"system     : Mg{m} Cl{n_cl} = {len(atoms)} atoms")
print(f"3-body model: {MODEL3}  (trios present)")
print(f"2-body model: {MODEL2}  (fast backend)\n")
atoms.calc = fast2
FIRE(atoms, logfile=None).run(fmax=1.0, steps=200)


def breakdown_uf3(model_path, label):
    """Profile the OLD uf3 3-body calc with a FRESH instance per property.

    uf3's UFCalculator exposes SEPARATE energy / forces / stress methods, so timing
    each on its own fresh (uncached) instance isolates the three costs. The previous
    version timed the ASE SumCalculator wrapper, whose calculate() computes all three
    properties in one call -> get_potential_energy did the full compute and the later
    get_forces/get_stress returned cached ~0.1 ms values, mislabelling everything as
    'energy'. ZBL is negligible, so it is omitted here.
    """
    out = {}
    for name in ("energy", "forces", "stress"):
        calc = UFCalculator(WeightedLinearModel.from_json(model_path))  # fresh -> no cross-property caching
        a = atoms.copy()
        a.calc = calc
        fn = {"energy": a.get_potential_energy, "forces": a.get_forces, "stress": a.get_stress}[name]
        t0 = time.perf_counter()
        fn()
        out[name] = time.perf_counter() - t0
    print(
        f"  {label:12s}  energy {out['energy'] * 1000:9.1f} ms | forces {out['forces'] * 1000:9.1f} ms | "
        f"stress {out['stress'] * 1000:9.1f} ms  (stress = numerical, ~12 energy evals)"
    )
    return out


print("--- single-point cost breakdown of the OLD 3-body backend (what a fast backend must fix) ---")
b3 = breakdown_uf3(MODEL3, "3-body old")
print()


def time_npt(sim, calc, nsteps, label):
    """Time an NPT run with the given calc and return ms/step."""
    sim.calc = calc
    a = atoms.copy()
    a.calc = calc
    t0 = time.perf_counter()
    sim.run_npt_simulation(a, steps=nsteps, traj_file=f"results/_speed3_{label}.traj", **NPT_KW)
    wall = time.perf_counter() - t0
    ms = wall / nsteps * 1000
    print(
        f"  {label:12s}: {wall:7.1f} s / {nsteps} steps -> {ms:9.1f} ms/step   "
        f"({nsteps * 0.06 / wall:6.3f} ps/min, 1 fs dt)"
    )
    return ms


print("--- NPT timing (mtknpt, stress every step) ---")
ms3 = time_npt(sim3, old3, N3, "3-body old")
ms3f = time_npt(sim3, fast3, N2, "3-body fast")  # reuse sim3, swap calc (run attaches calc to atoms)
ms2 = time_npt(sim3, fast2, N2, "2-body fast")

print("\n=== VERDICT ===")
print(f"  fast 3-body speedup over old 3-body : {ms3 / ms3f:7.0f}x")
print(f"  fast 3-body vs fast 2-body          : {ms3f / ms2:7.1f}x slower (cost of the angular term)")
prod3_h = ms3 * PROD_STEPS / 1000 / 3600
prod3f_h = ms3f * PROD_STEPS / 1000 / 3600
prod2_h = ms2 * PROD_STEPS / 1000 / 3600
print(f"  one production sweep ({PROD_STEPS:,} steps) at {len(atoms)} atoms:")
print(f"     2-body fast : {prod2_h:8.2f} h")
print(f"     3-body fast : {prod3f_h:8.2f} h")
print(f"     3-body old  : {prod3_h:8.2f} h  ({prod3_h / 24:.1f} days)")
print(
    f"  (old 3-body cost breakdown: forces {b3['forces'] * 1000:.0f} ms, "
    f"numerical stress {b3['stress'] * 1000:.0f} ms, energy {b3['energy'] * 1000:.0f} ms)"
)
