"""Rescue property sweep (FAST backend) for salts that evaporated at the cluster's
high temperatures. Reruns a pure-cation melt at a tailored lower range near the
melting point. Writes results/md_<prefix>_<T>K[/_dt<..>]/{npt,nvt}.traj + summary.json
in the SAME layout as the cluster runs, so analyze_properties.py works unchanged.

  uv run python scripts/rescue_sweep.py <model.json> <prefix> <cation> <T1> <T2> ... [--dt 0.5] [--nve-steps N]

  --dt <fs>        MD timestep in fs (default 1.0). A non-1.0 dt writes to a
                   separate dir md_<prefix>_<T>K_dt<dt> so it never overwrites the
                   1 fs runs. Use 0.5 to test whether the evaporation is an
                   integration-stability artifact rather than a thermodynamic
                   collapse (stiff light-cation Mg-Cl bonds make 1 fs borderline).
  --nve-steps <N>  Steps for the post-equilibration NVE energy-drift probe
                   (default 4000; set 0 to skip). The drift (meV/atom/ps) is the
                   real diagnostic: if 1 fs is leaking energy you will see a large
                   positive drift that shrinks at 0.5 fs; a flat drift at both means
                   the collapse is thermodynamic, not numerical.

e.g.
  uv run python scripts/rescue_sweep.py results/2body_CaMgCl_zbl/model.json CaCl2 Ca 1105 --dt 0.5
  uv run python scripts/rescue_sweep.py results/2body_CaMgCl_zbl/model.json MgCl2 Mg 995  --dt 0.5
  uv run python scripts/rescue_sweep.py results/2body_CaMgCl_zbl/model.json CaCl2 Ca 1105           # 1 fs baseline for comparison
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
from ase import units
from ase.optimize import FIRE
from ase.md.verlet import VelocityVerlet
from moltensaltcalc import MoltenSaltSimulator
from moltensaltcalc.models.fit_uf3 import ZBLRepulsion
from uf3.forcefield.calculator import UFCalculator
from uf3.regression.least_squares import WeightedLinearModel
from fit_uf3_fast import UF3FastCalculator

CHARGE={"Li":1,"Na":1,"K":1,"Rb":1,"Cs":1,"Mg":2,"Ca":2,"Sr":2,"Ba":2,"Zn":2,"Zr":4}
ANION="Cl"; TARGET=400; NPT_STEPS=40000; NVT_STEPS=40000

# ---- parse args (positional + optional flags) ----
args=sys.argv[1:]
DT=1.0; NVE_STEPS=4000
pos=[]; i=0
while i<len(args):
    a=args[i]
    if a=="--dt":         DT=float(args[i+1]);       i+=2; continue
    if a=="--nve-steps":  NVE_STEPS=int(args[i+1]);   i+=2; continue
    pos.append(a); i+=1
MODEL=pos[0]; PREFIX=pos[1]; CATION=pos[2]; TEMPS=[float(t) for t in pos[3:]]
DTSUF="" if abs(DT-1.0)<1e-9 else f"_dt{str(DT).replace('.','p')}"

cpu=CHARGE[CATION]; apu=1+cpu; m=max(1,round(TARGET/apu)); n_cl=m*cpu

fast=UF3FastCalculator(UFCalculator(WeightedLinearModel.from_json(MODEL)), wall=ZBLRepulsion(r_in=4.0,r_out=5.0))
sim=MoltenSaltSimulator(model_name="fit_uf3",model_parameters={"model_path":MODEL},device="cpu",dispersion=None); sim.calc=fast
print(f"{PREFIX}: {CATION}{m} Cl{n_cl} = {m+n_cl} atoms | model {MODEL} | dt={DT} fs | T = {TEMPS}")

def nve_drift(atoms0, calc, dt_fs, steps, sample=50):
    """Short NVE run; returns (drift_meV_per_atom_per_ps, Etot_start, Etot_end) in meV/atom."""
    at=atoms0.copy(); at.calc=calc
    dyn=VelocityVerlet(at, timestep=dt_fs*units.fs)
    n=len(at)
    def etot(): return (at.get_potential_energy()+at.get_kinetic_energy())/n*1000.0  # meV/atom
    ts=[0.0]; es=[etot()]; done=0
    while done<steps:
        step=min(sample,steps-done); dyn.run(step); done+=step
        ts.append(done*dt_fs/1000.0); es.append(etot())   # ps, meV/atom
    slope=float(np.polyfit(ts,es,1)[0])                    # meV/atom/ps
    return slope, es[0], es[-1]

for T in TEMPS:
    OUT=Path("results")/f"md_{PREFIX}_{int(T)}K{DTSUF}"; OUT.mkdir(parents=True,exist_ok=True)
    np.random.seed(42)
    atoms=sim.build_system(salt_anion=[ANION],salt_cation=[CATION],n_anions=[n_cl],n_cations=[m],density_guess=2.5,lattice="random")
    atoms.calc=fast
    FIRE(atoms,logfile=str(OUT/"min.log")).run(fmax=1.0,steps=200)
    t0=time.time()
    aeq=sim.run_npt_simulation(atoms,T=T,steps=NPT_STEPS,timestep_fs=DT,npt_dyn="mtknpt",
        taut_fs=100.0,taup_fs=1000.0,compressibility_per_bar=5e-6,pressure_bar=1.01325,tchain=3,pchain=3,tloop=1,ploop=1,
        write_interval=10,print_interval=10000,traj_file=str(OUT/"npt.traj"),logfile=str(OUT/"npt_run.log"))
    aeq.calc=fast
    sim.run_nvt_simulation(aeq,T=T,steps=NVT_STEPS,timestep_fs=DT,nvt_dyn="nosehoover",tdamp_fs=100.0,
        write_interval=10,print_interval=10000,traj_file=str(OUT/"nvt.traj"),logfile=str(OUT/"nvt_run.log"))
    rho=aeq.get_masses().sum()/aeq.get_volume()*1.6605390666  # amu/A^3 -> g/cm^3

    drift=etot0=etot1=None
    if NVE_STEPS>0:
        try:
            drift,etot0,etot1=nve_drift(aeq,fast,DT,NVE_STEPS)
        except Exception as e:
            print(f"  [nve-drift probe failed: {e}]")

    json.dump({"model":MODEL,"T_K":T,"n_atoms":len(aeq),"dt_fs":DT,"eq_density_g_cm3":rho,
               "nve_drift_meV_per_atom_per_ps":drift,"nve_etot_start_meV_atom":etot0,"nve_etot_end_meV_atom":etot1},
              open(OUT/"summary.json","w"),indent=2)
    dstr=f" | drift {drift:+.2f} meV/atom/ps" if drift is not None else ""
    print(f"  T={T:.0f}K dt={DT}fs  rho={rho:.3f} g/cm3  {'OK' if rho>1.0 else '*** EVAPORATED ***'}{dstr}  [{(time.time()-t0)/60:.1f} min] -> {OUT}")
