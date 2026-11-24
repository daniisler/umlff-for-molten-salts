import sys, time, numpy as np
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from moltensaltcalc.simulator import MoltenSaltSimulator

model_path = sys.argv[1]
cation     = sys.argv[2] if len(sys.argv) > 2 else "Mg"
n_fu       = int(sys.argv[3]) if len(sys.argv) > 3 else 16
NSTEPS, T  = 10, 1100.0

syms = [cation]*n_fu + ["Cl"]*(2*n_fu)
L = (len(syms) / 0.033) ** (1/3)
rng = np.random.default_rng(42)
at = Atoms("".join(syms), positions=rng.random((len(syms),3))*L, cell=[L,L,L], pbc=True)

sim = MoltenSaltSimulator(model_name="fit_uf3",
                          model_parameters={"model_path": model_path},
                          dispersion=None)
at.calc = sim.calc
print(f"built fit_uf3 (3-body) calc; N={len(at)} atoms, elements={sim.calc.elements}", flush=True)

MaxwellBoltzmannDistribution(at, temperature_K=T)
dyn = Langevin(at, 1.0*units.fs, temperature_K=T, friction=0.01)
t0 = time.perf_counter(); last=[t0]
def report(a=at):
    now=time.perf_counter(); dt=now-last[0]; last[0]=now
    print(f"  step {dyn.nsteps:3d}  T={a.get_temperature():7.1f} K  "
          f"Epot={a.get_potential_energy():.1f} eV  {dt:5.1f} s/step", flush=True)
dyn.attach(report, interval=1)
print(f"running {NSTEPS} NVT steps at {T:.0f} K ...", flush=True)
dyn.run(NSTEPS)
tot=time.perf_counter()-t0
print(f"DONE: {NSTEPS} steps in {tot:.1f} s -> {tot/NSTEPS:.1f} s/step avg at N={len(at)}", flush=True)
