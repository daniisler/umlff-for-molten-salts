"""Viscosity production run: long weak-NVT trajectory for Green-Kubo.

FAST backend, built directly and swapped into a fit_uf3 simulator. Writes nvt.traj
with per-frame stress (the GK integrand), sampled every WRITE_EVERY fs. Transport
wants minimal thermostatting -> weak Nose-Hoover (large tdamp). Use several SEEDs for
independent replicas (GK viscosity is noisy; average the plateau across replicas).

  uv run python scripts/viscosity_run.py <model.json> <T_K> <atoms> <npt_steps> <nvt_steps> <write_every_fs> <label> <seed>
  defaults: NaCl 1100 K, 250 atoms, 30000 NPT, 120000 NVT, 5 fs, auto label, seed 1

Analyse:  uv run python scripts/viscosity.py results/md_<label> <T_K>
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
from ase.optimize import FIRE
from moltensaltcalc import MoltenSaltSimulator
from moltensaltcalc.models.fit_uf3 import ZBLRepulsion
from uf3.forcefield.calculator import UFCalculator
from uf3.regression.least_squares import WeightedLinearModel
from fit_uf3_fast import UF3FastCalculator

CHARGE={"Li":1,"Na":1,"K":1,"Rb":1,"Cs":1,"Mg":2,"Ca":2,"Sr":2,"Ba":2,"Zn":2,"Zr":4}
ANION="Cl"
def _arg(i,d): return sys.argv[i] if len(sys.argv)>i and sys.argv[i]!="" else d
def build_composition(c,t):
    cpu=sum(CHARGE[x] for x in c); apu=len(c)+cpu; m=max(1,round(t/apu)); return [m]*len(c), m*cpu

MODEL=_arg(1,"results/2body_NaCl_zbl/model.json"); T_K=float(_arg(2,1100.0)); TARGET=int(_arg(3,250))
NPT_STEPS=int(_arg(4,30000)); NVT_STEPS=int(_arg(5,120000)); WRITE_EVERY=int(_arg(6,5)); SEED=int(_arg(8,1))
DT_FS=1.0; TDAMP_FS=1000.0
np.random.seed(SEED)

fast=UF3FastCalculator(UFCalculator(WeightedLinearModel.from_json(MODEL)), wall=ZBLRepulsion(r_in=4.0,r_out=5.0))
sim=MoltenSaltSimulator(model_name="fit_uf3", model_parameters={"model_path":MODEL}, device="cpu", dispersion=None)
sim.calc=fast
elements=list(sim.calc.elements); cations=[e for e in elements if e!=ANION]
n_cat,n_cl=build_composition(cations,TARGET)
LABEL=_arg(7, "".join(cations)+ANION+f"_visc_{int(T_K)}K_s{SEED}")
OUT=Path("results")/f"md_{LABEL}"; OUT.mkdir(parents=True,exist_ok=True)
print(f"{LABEL}: {dict(zip(cations,n_cat))}+{{'Cl':{n_cl}}}={sum(n_cat)+n_cl} atoms, seed={SEED}, write every {WRITE_EVERY} fs")

atoms=sim.build_system(salt_anion=[ANION],salt_cation=cations,n_anions=[n_cl],n_cations=n_cat,density_guess=2.5,lattice="random")
atoms.calc=fast
FIRE(atoms,logfile=str(OUT/"min.log")).run(fmax=1.0,steps=200)
t0=time.time()
atoms_eq=sim.run_npt_simulation(atoms,T=T_K,steps=NPT_STEPS,timestep_fs=DT_FS,npt_dyn="mtknpt",
    taut_fs=100.0,taup_fs=1000.0,compressibility_per_bar=5e-6,pressure_bar=1.01325,tchain=3,pchain=3,tloop=1,ploop=1,
    write_interval=200,print_interval=10000,traj_file=str(OUT/"npt.traj"),logfile=str(OUT/"npt_run.log"))
atoms_eq.calc=fast
sim.run_nvt_simulation(atoms_eq,T=T_K,steps=NVT_STEPS,timestep_fs=DT_FS,nvt_dyn="nosehoover",tdamp_fs=TDAMP_FS,
    write_interval=WRITE_EVERY,print_interval=10000,traj_file=str(OUT/"nvt.traj"),logfile=str(OUT/"nvt_run.log"))
print(f"done in {(time.time()-t0)/60:.1f} min -> {OUT}/nvt.traj")
