# MD property pipeline — handoff (UF3 + ZBL)

Run the liquid-property benchmark (density, structure, diffusion, Cp, thermal
expansion) on the cluster. Everything is in this repo — no files from a laptop.

## 1. Install the backend

Needs **Python 3.11–3.13** (MoltenSaltCalc requires `>=3.11,<3.14`; the project
targets 3.12). Do NOT use Python 3.14 — the backend will not install. Install the
`fit_uf3` backend (UF3 + a parameter-free ZBL short-range baseline) with the `uf3`
extra, into an env with `numpy < 2.3` (uf3 crashes on newer numpy):

```bash
uv venv --python 3.12 .venv-msc && source .venv-msc/bin/activate
uv pip install "numpy<2.3"
uv pip install "moltensaltcalc[fit_uf3] @ git+https://github.com/leiapple/MoltenSaltCalc.git@marios/fit-uf3-backend"
uv pip install matplotlib
```

(Without `uv`: make a venv with a Python 3.12 interpreter — `python3.12 -m venv ...` —
then `pip install` the same three lines. The driver needs `matplotlib` for the RDF plot.)

## 2. Get the scripts + models

```bash
git clone https://github.com/daniisler/umlff-for-molten-salts.git
cd umlff-for-molten-salts
git checkout marios/uf3-reproduction
```

The 9 fitted ZBL models are already committed (`results/2body_*_zbl/model.json`), so
there is nothing to fit and no training data needed.

## 3. Run the property pipeline (both tracks)

```bash
RUN="python" bash scripts/run_md_jobs.sh 1 11     # all 11 salts (both tracks)
RUN="python" bash scripts/run_md_jobs.sh 1 9      # just the 9 core salts
```

Salts are listed one per line in `scripts/md_jobs.txt`; the runner takes a job-index
range, so ranges can run on different nodes (e.g. `1 6` on one, `7 11` on another).

Per salt: a 3-temperature sweep (NPT + NVT at each T, 400 atoms, 60000 + 60000 steps),
then `analyze_properties.py` computes the five properties:

- **density**, **RDF / structure**, **self-diffusion** (per temperature)
- **heat capacity Cp** and **thermal expansion alpha_V** (across temperatures — these
  are slopes in T, which is why each salt is a sweep, not a single run)

Output: `results/<salt>_properties.json` (plus per-run `results/md_<salt>_<T>K/`).
Jobs **1–9** are the well-referenced core salts; **10–11** (ZnCl2, ZrCl4) are hard
cases, expected to look poor. It is a large job — launch it in the background or via
the scheduler. Bulk modulus is excluded (it is a 0 K solid property).

## 4. Extras (optional)

- **Viscosity** — Green-Kubo, noisy and expensive, so kept separate. Run a dedicated
  long NVT, then analyse:
  ```bash
  python scripts/md_moltensaltcalc.py results/2body_NaCl_zbl/model.json 1100 20000 200000 NaCl_visc_1100K mtknpt 400 Na
  python scripts/viscosity.py results/md_NaCl_visc_1100K 1100 --tmax-fs 20000
  ```
- **0 K bulk modulus** — `python scripts/eos_static.py <model.json> <npt.traj> --points 17 --range 0.75 1.05 --relax` (not part of the main pipeline).
- **Re-fit a model** — needs `data/Training.extxyz` (the SuperSalt training set):
  ```bash
  python scripts/make_subsystem.py --elements Cl Zn --out sub_ZnCl.extxyz
  python scripts/split_frames.py --input sub_ZnCl.extxyz
  python scripts/fit_uf3.py --mode 2body --train-file sub_ZnCl_train.extxyz \
    --test-file sub_ZnCl_test.extxyz --elements Cl Zn --cutoff2 8.0 --res2 20 \
    --weight 0.3 --zbl --run-name 2body_ZnCl_zbl
  ```
  Deterministic (seed 42) — reproduces the committed models exactly.
