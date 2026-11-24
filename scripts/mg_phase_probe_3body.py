"""MgCl2 phase probe — 3-BODY model. Did the angular term lift the boiling point?

The 2-body model has a real MgCl2 liquid but a COLLAPSED boiling point (~875 K, below the
melting point 987 K) -- see mg_phase_probe.py / md_results.md. The 3-body campaign runs at
1000-1300 K did NOT answer whether 3-body fixes this: they CRASHED numerically (T -> ~29000 K)
rather than boiling, because a random start + straight NPT at 0.5 fs is too aggressive for the
stiff 3-body Mg-Cl forces.

This probe runs the 3-body model STABLY and asks the real question: as T rises at 1 atm, how
high can the 3-body model hold a dense MgCl2 liquid? If it holds a liquid ABOVE ~987 K (the
melting point), the angular term lifted the boiling point back above T_m -> MgCl2 fixed. If it
still boils below T_m, 3-body cohesion is still too weak.

Numerics that keep it stable (vs the campaign crash):
  - 3-body FAST backend + ZBL wall (same as the campaign) but dt = 0.25 fs (0.5 fs blew up).
  - FIRE minimise -> SHORT NVT equilibration at fixed box (absorbs the startup heat spike)
    -> NPT. The campaign went FIRE -> straight NPT and the barostat saw the spike.
  - Each run try/except-wrapped; summary.json rewritten after every run (survives an interrupt).

Run it unattended (keeps the Mac awake, lets you close the terminal):
  caffeinate -i nohup uv run python scripts/mg_phase_probe_3body.py > mg3b.out 2>&1 &
  # defaults to results/3body_MgCl/model.json -- refit it first (see md_results.md) if deleted:
  #   NTRAIN=500 CATIONS="Mg" bash scripts/fit_3body_models.sh
  # or pass another model:  ... mg_phase_probe_3body.py results/3body_MgCl_speedtest/model.json

Outputs -> results/mg_phase_probe_3body/<run>/ + results/mg_phase_probe_3body/summary.json
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from ase.io import read
from ase.optimize import FIRE
from fit_uf3_fast_3body import UF3Fast3BodyCalculator  # 3-body fast backend
from moltensaltcalc import MoltenSaltSimulator
from moltensaltcalc.models.fit_uf3 import ZBLRepulsion
from uf3.forcefield.calculator import UFCalculator
from uf3.regression.least_squares import WeightedLinearModel

MODEL = sys.argv[1] if len(sys.argv) > 1 else "results/3body_MgCl/model.json"
CATION, ANION, CHARGE = "Mg", "Cl", 2
DT = 0.25  # fs -- 0.5 fs blew up the 3-body Mg-Cl dynamics
TARGET = 250
NPT_STEPS = 20000  # 5 ps -- enough to see hold-vs-boil + an equilibrium density
NVT_EQ_STEPS = 3000  # 0.75 ps fixed-box thermalisation before the barostat
RHO_EXP = 1.678  # Janz MgCl2 at 995 K (g/cm3)
AMU_A3_TO_G_CM3 = 1.6605390666
EV_A3_TO_BAR = 1.602176634e6

# the decisive temperature sweep at 1 atm: around the 2-body collapse (~875 K) up through the
# real melting point (987 K) into where real MgCl2 is molten.
T_SWEEP = (850, 950, 1000, 1050)

OUTROOT = Path("results/mg_phase_probe_3body")
OUTROOT.mkdir(parents=True, exist_ok=True)
SUMMARY = OUTROOT / "summary.json"
results = {
    "model": MODEL,
    "dt_fs": DT,
    "atoms_target": TARGET,
    "note": "3-body MgCl2: does the angular term lift T_boil above T_m (987 K)?",
    "runs": [],
}

m = max(1, round(TARGET / (1 + CHARGE)))
n_cl = m * CHARGE
fast = UF3Fast3BodyCalculator(
    UFCalculator(WeightedLinearModel.from_json(MODEL)), wall=ZBLRepulsion(r_in=4.0, r_out=5.0)
)
sim = MoltenSaltSimulator(device="cpu", dispersion=None)
sim.calc = fast
print(f"MgCl2 3-body phase probe | Mg{m} Cl{n_cl} = {m + n_cl} atoms | dt={DT} fs | model {MODEL}", flush=True)


def density(a):
    """Mass density of an Atoms object in g/cm3."""
    return float(a.get_masses().sum() / a.get_volume() * AMU_A3_TO_G_CM3)


def mean_pressure_bar(traj_path, skip_frac=0.5):
    """Mean +/- std scalar pressure (bar) over the second half of a trajectory."""
    frames = read(traj_path, index=":")
    frames = frames[int(len(frames) * skip_frac) :] or frames[-1:]
    ps = []
    for a in frames:
        a.calc = fast
        s = a.get_stress(voigt=True)
        ps.append(-(s[0] + s[1] + s[2]) / 3.0 * EV_A3_TO_BAR)
    return float(np.mean(ps)), float(np.std(ps))


def save():
    """Rewrite summary.json after every run so an interrupt still leaves partial results."""
    with open(SUMMARY, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)


def build(density_guess, T):
    """Dilute-ish random pack -> FIRE -> short NVT equilibration at T (fixed box)."""
    a = sim.build_system(
        salt_anion=[ANION],
        salt_cation=[CATION],
        n_anions=[n_cl],
        n_cations=[m],
        density_guess=density_guess,
        lattice="random",
    )
    a.calc = fast
    FIRE(a, logfile=str(OUTROOT / "min.log")).run(fmax=1.0, steps=200)
    # NVT equilibration at fixed box -- thermalise before the barostat sees any energy spike
    a = (
        sim.run_nvt_simulation(
            a,
            T=T,
            steps=NVT_EQ_STEPS,
            timestep_fs=DT,
            nvt_dyn="nosehoover",
            tdamp_fs=100.0,
            write_interval=50,
            print_interval=2000,
            logfile=str(OUTROOT / "eq.log"),
        )
        or a
    )
    return a


def run_npt(tag, T, p_bar, density_guess=2.0):
    """One NPT point: build -> equilibrate -> NPT; record equilibrium density and liquid/vapour."""
    out = OUTROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    rec = {"run": tag, "type": "NPT", "T_K": T, "P_bar": p_bar}
    try:
        np.random.seed(42)
        t0 = time.time()
        a = build(density_guess, T)
        aeq = sim.run_npt_simulation(
            a,
            T=T,
            steps=NPT_STEPS,
            timestep_fs=DT,
            npt_dyn="mtknpt",
            taut_fs=100.0,
            taup_fs=1000.0,
            compressibility_per_bar=5e-6,
            pressure_bar=p_bar,
            tchain=3,
            pchain=3,
            tloop=1,
            ploop=1,
            write_interval=20,
            print_interval=5000,
            traj_file=str(out / "npt.traj"),
            logfile=str(out / "npt_run.log"),
        )
        rec["eq_density_g_cm3"] = density(aeq)
        rec["minutes"] = round((time.time() - t0) / 60, 1)
        rec["liquid"] = bool(rec["eq_density_g_cm3"] > 1.0)
        print(
            f"  [{tag}] T={T:.0f} P={p_bar} bar -> rho={rec['eq_density_g_cm3']:.3f}  "
            f"{'LIQUID' if rec['liquid'] else 'vapour'}  [{rec['minutes']} min]",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        rec["error"] = repr(e)
        traceback.print_exc()
    results["runs"].append(rec)
    save()


def run_nvt_fixed_density(tag, T, rho):
    """NVT clamped at a target density; reports the mean internal pressure (coherence check)."""
    out = OUTROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    rec = {"run": tag, "type": "NVT@rho", "T_K": T, "target_rho": rho}
    try:
        np.random.seed(42)
        t0 = time.time()
        a = build(rho, T)
        rec["box_rho_g_cm3"] = density(a)
        sim.run_nvt_simulation(
            a,
            T=T,
            steps=NPT_STEPS,
            timestep_fs=DT,
            nvt_dyn="nosehoover",
            tdamp_fs=100.0,
            write_interval=20,
            print_interval=5000,
            traj_file=str(out / "nvt.traj"),
            logfile=str(out / "nvt_run.log"),
        )
        p_mean, p_std = mean_pressure_bar(out / "nvt.traj")
        rec["mean_pressure_bar"] = round(p_mean, 1)
        rec["pressure_std_bar"] = round(p_std, 1)
        rec["minutes"] = round((time.time() - t0) / 60, 1)
        print(
            f"  [{tag}] NVT @ rho={rec['box_rho_g_cm3']:.3f}, T={T:.0f} -> "
            f"mean P = {p_mean:.0f} ± {p_std:.0f} bar  [{rec['minutes']} min]",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        rec["error"] = repr(e)
        traceback.print_exc()
    results["runs"].append(rec)
    save()


# ---- A) the decisive T-sweep at 1 atm: how high can 3-body hold a liquid? ----
print("A) temperature sweep (NPT, 1 bar) -- the boiling-point test:", flush=True)
for t_k in T_SWEEP:
    run_npt(f"T{t_k}K_1bar", float(t_k), 1.01325)

# ---- B) sanity: NVT clamped at the experimental density (is the liquid coherent at 995 K?) ----
print("B) NVT at experimental density (rho=1.678, 995 K):", flush=True)
run_nvt_fixed_density("NVT_rhoexp_995K", 995.0, RHO_EXP)

# ---- verdict ----
liq = [r for r in results["runs"] if r.get("type") == "NPT" and r.get("liquid")]
hi = max((r["T_K"] for r in liq), default=None)
if hi is None:
    VERDICT = "3-body still boils MgCl2 at every T tested (>=850 K) -> cohesion still too weak."
elif hi >= 987:
    VERDICT = f"3-body HOLDS a liquid up to >= {hi:.0f} K (>= T_m 987 K) -> angular term lifted T_boil above melting. MgCl2 FIXED."
else:
    VERDICT = (
        f"3-body holds a liquid up to ~{hi:.0f} K but still below T_m (987 K) -> boiling point "
        f"raised vs 2-body (~875 K) but not enough; partial improvement."
    )
results["verdict"] = VERDICT
save()
print("\n=== VERDICT ===\n" + VERDICT + f"\nsummary -> {SUMMARY}", flush=True)
