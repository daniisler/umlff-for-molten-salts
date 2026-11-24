"""MgCl2 phase probe — is the 2-body collapse case A (no liquid exists) or
case B (a dense liquid exists but boils early, i.e. a collapsed boiling point)?

Runs UNATTENDED. ALL at dt = 0.5 fs (1 fs is a known integration artifact for the
stiff Mg-Cl bond — see Session 18). Fast backend. Pure MgCl2, 400 atoms.

Three parts, each run wrapped so one failure never kills the batch; summary.json is
rewritten after every run so partial results survive an interrupt.

  A) PRESSURE sweep (NPT, 995 K, P = 100 / 1000 / 5000 bar): does squeezing recover a
     dense liquid? If density climbs toward experiment (~1.68) as P rises, a liquid
     STATE exists -> case B (boiling-point collapse). If it stays vapour even at
     5000 bar -> case A.
  B) NVT at the EXPERIMENTAL density (rho = 1.678 g/cm3, 995 K): clamp the box at the
     real liquid density and read the internal pressure + whether the structure stays
     a homogeneous liquid. A modest pressure + intact Mg-Cl coordination -> a
     metastable liquid is present (case B). Huge tension / cavitation -> case A.
  C) TEMPERATURE sweep (NPT, 1 bar, T = 950/900/850/800/750 K): where (if anywhere)
     does the model first hold a liquid at 1 atm -> the model's collapsed boiling point.

  uv run python scripts/mg_phase_probe.py [results/2body_CaMgCl_zbl/model.json]

Outputs -> results/mg_phase_probe/<run>/{npt,nvt}.traj + results/mg_phase_probe/summary.json
"""
from __future__ import annotations
import sys, json, time, traceback
from pathlib import Path
import numpy as np
from ase.io import read
from ase.optimize import FIRE
from moltensaltcalc import MoltenSaltSimulator
from moltensaltcalc.models.fit_uf3 import ZBLRepulsion
from uf3.forcefield.calculator import UFCalculator
from uf3.regression.least_squares import WeightedLinearModel
from fit_uf3_fast import UF3FastCalculator

MODEL = sys.argv[1] if len(sys.argv) > 1 else "results/2body_CaMgCl_zbl/model.json"
CATION, ANION, CHARGE = "Mg", "Cl", 2
T0 = 995.0
DT = 0.5
TARGET = 400
NPT_STEPS = 40000
NVT_STEPS = 40000
RHO_EXP = 1.678                       # Janz MgCl2 at 995 K (g/cm3)
AMU_A3_TO_G_CM3 = 1.6605390666
EV_A3_TO_BAR = 1.602176634e6          # 1 eV/Å^3 = 1.602176634e6 bar

OUTROOT = Path("results/mg_phase_probe"); OUTROOT.mkdir(parents=True, exist_ok=True)
SUMMARY = OUTROOT / "summary.json"
results = {"model": MODEL, "dt_fs": DT, "note": "MgCl2 case A (no liquid) vs B (collapsed boiling point)", "runs": []}

m = max(1, round(TARGET / (1 + CHARGE))); n_cl = m * CHARGE
fast = UF3FastCalculator(UFCalculator(WeightedLinearModel.from_json(MODEL)), wall=ZBLRepulsion(r_in=4.0, r_out=5.0))
sim = MoltenSaltSimulator(model_name="fit_uf3", model_parameters={"model_path": MODEL}, device="cpu", dispersion=None); sim.calc = fast
print(f"MgCl2 phase probe | {CATION}{m} Cl{n_cl} = {m + n_cl} atoms | dt={DT} fs | model {MODEL}", flush=True)

def density(a):
    return float(a.get_masses().sum() / a.get_volume() * AMU_A3_TO_G_CM3)

def mean_pressure_bar(traj_path, skip_frac=0.5):
    frames = read(traj_path, index=":")
    frames = frames[int(len(frames) * skip_frac):] or frames[-1:]
    ps = []
    for a in frames:
        a.calc = fast
        s = a.get_stress(voigt=True)              # eV/Å^3: xx,yy,zz,yz,xz,xy
        ps.append(-(s[0] + s[1] + s[2]) / 3.0 * EV_A3_TO_BAR)
    return float(np.mean(ps)), float(np.std(ps))

def save():
    json.dump(results, open(SUMMARY, "w"), indent=2)

def build(density_guess):
    a = sim.build_system(salt_anion=[ANION], salt_cation=[CATION], n_anions=[n_cl], n_cations=[m],
                         density_guess=density_guess, lattice="random")
    a.calc = fast
    FIRE(a, logfile=str(OUTROOT / "min.log")).run(fmax=1.0, steps=200)
    return a

def run_npt(tag, T, P_bar):
    out = OUTROOT / tag; out.mkdir(parents=True, exist_ok=True)
    rec = {"run": tag, "type": "NPT", "T_K": T, "P_bar": P_bar}
    try:
        np.random.seed(42)
        t0 = time.time()
        a = build(2.5)
        aeq = sim.run_npt_simulation(a, T=T, steps=NPT_STEPS, timestep_fs=DT, npt_dyn="mtknpt",
            taut_fs=100.0, taup_fs=1000.0, compressibility_per_bar=5e-6, pressure_bar=P_bar,
            tchain=3, pchain=3, tloop=1, ploop=1, write_interval=10, print_interval=10000,
            traj_file=str(out / "npt.traj"), logfile=str(out / "npt_run.log"))
        rec["eq_density_g_cm3"] = density(aeq)
        rec["minutes"] = round((time.time() - t0) / 60, 1)
        rec["liquid"] = bool(rec["eq_density_g_cm3"] > 1.0)
        print(f"  [{tag}] T={T:.0f} P={P_bar} bar -> rho={rec['eq_density_g_cm3']:.3f}  "
              f"{'LIQUID' if rec['liquid'] else 'vapour'}  [{rec['minutes']} min]", flush=True)
    except Exception as e:
        rec["error"] = repr(e); traceback.print_exc()
    results["runs"].append(rec); save()

def run_nvt_fixed_density(tag, T, rho):
    out = OUTROOT / tag; out.mkdir(parents=True, exist_ok=True)
    rec = {"run": tag, "type": "NVT@rho", "T_K": T, "target_rho": rho}
    try:
        np.random.seed(42)
        t0 = time.time()
        a = build(rho)
        rec["box_rho_g_cm3"] = density(a)         # fixed through NVT
        sim.run_nvt_simulation(a, T=T, steps=NVT_STEPS, timestep_fs=DT, nvt_dyn="nosehoover",
            tdamp_fs=100.0, write_interval=10, print_interval=10000,
            traj_file=str(out / "nvt.traj"), logfile=str(out / "nvt_run.log"))
        p_mean, p_std = mean_pressure_bar(out / "nvt.traj")
        rec["mean_pressure_bar"] = round(p_mean, 1); rec["pressure_std_bar"] = round(p_std, 1)
        rec["minutes"] = round((time.time() - t0) / 60, 1)
        print(f"  [{tag}] NVT @ rho={rec['box_rho_g_cm3']:.3f}, T={T:.0f} -> "
              f"mean P = {p_mean:.0f} ± {p_std:.0f} bar  [{rec['minutes']} min]", flush=True)
    except Exception as e:
        rec["error"] = repr(e); traceback.print_exc()
    results["runs"].append(rec); save()

# ---- A) pressure sweep at 995 K ----
print("A) pressure sweep (NPT, 995 K):", flush=True)
for P in (100, 1000, 5000):
    run_npt(f"A_P{P}bar_995K", T0, P)

# ---- B) NVT clamped at experimental density ----
print("B) NVT at experimental density (rho=1.678, 995 K):", flush=True)
run_nvt_fixed_density("B_NVT_rhoexp_995K", T0, RHO_EXP)

# ---- C) temperature sweep at 1 bar ----
print("C) temperature sweep (NPT, 1 bar):", flush=True)
for T in (950, 900, 850, 800, 750):
    run_npt(f"C_T{T}K_1bar", float(T), 1.01325)

# ---- verdict ----
liq_P = [r for r in results["runs"] if r.get("type") == "NPT" and r.get("P_bar", 0) > 10 and r.get("liquid")]
liq_T = [r for r in results["runs"] if r.get("run", "").startswith("C_") and r.get("liquid")]
verdict = ("case B (liquid exists, boils early) — pressure and/or low-T recovers a dense liquid"
           if (liq_P or liq_T) else
           "case A-leaning — no liquid recovered by pressure or low T; check the NVT@rho pressure")
results["verdict"] = verdict
save()
print("\n=== VERDICT ===\n" + verdict, flush=True)
print(f"summary -> {SUMMARY}", flush=True)
