"""
Probe the 2-body UF3 pair potentials at short range.

Pushes two atoms together and prints the pair energy vs separation. A physical
potential must RISE STEEPLY (repulsive wall) as r -> small. If it flattens or
turns DOWN at short r, the model has no repulsion there -> under a barostat the
box collapses (what we saw for K-Rb-Cl).

Run:  uv run python scripts/probe_pair_potential.py <model.json>
"""
import sys
import numpy as np
from ase import Atoms
from uf3.regression.least_squares import WeightedLinearModel
from uf3.forcefield.calculator import UFCalculator

model_path = sys.argv[1] if len(sys.argv) > 1 else "results/2body_ClKRb_n1000_c8_w0.3_seed42_377d4e/model.json"
model = WeightedLinearModel.from_json(model_path)
calc = UFCalculator(model)
elements = [str(k) for k in calc.solutions if not isinstance(k, tuple)]
print(f"model: {model_path}\nelements: {elements}\n")

# every distinct pair
pairs = [(a, b) for i, a in enumerate(elements) for b in elements[i:]]
rs = np.arange(0.8, 5.01, 0.2)

for a, b in pairs:
    print(f"--- {a}-{b} pair energy vs r (relative to r=5.0) ---")
    es = []
    for r in rs:
        at = Atoms([a, b], positions=[[0, 0, 0], [r, 0, 0]], cell=[30, 30, 30], pbc=True)
        at.calc = calc
        es.append(at.get_potential_energy())
    es = np.array(es) - es[-1]  # reference to large separation
    for r, e in zip(rs, es):
        bar = "#" * min(60, max(0, int(e * 2))) if e > 0 else ("<" * min(40, int(-e * 2)))
        print(f"  r={r:4.1f}  dE={e:8.2f} eV  {bar}")
    # verdict
    wall = es[0] > es[len(es)//3]  # is it higher at r=0.8 than mid-range?
    print(f"  short-range repulsive wall at r=0.8? {'YES' if wall else 'NO  <-- PATHOLOGY (no wall, will collapse)'}\n")
