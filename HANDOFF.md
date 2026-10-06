# Running the UF3 jobs — handoff

Everything needed to run the UF3 fit/evaluate script on the cluster. Full
rationale, job matrix and expected numbers are in `docs/uf3_cluster_run_plan.md`.

## What changed
- **Always use disjoint train/test files.** Sampling both from the *same* file
  leaks frames between fit and evaluation and flatters the error. Every command
  below now uses a `*_train` / `*_test` pair. Disjoint splits for two subsystems
  are already in `data/` (`sub_CsZrCl_*`, `sub_KRbCl_*`); make more with the new
  `scripts/split_frames.py` (seeded 80/20).
- **Output folders are self-naming now.** `run_name` encodes mode, elements,
  frames, cutoff, weight, seed and a hash of every other parameter, so no two
  configs overwrite each other (e.g. `3body_CsZrCl_n1500_c5_w0.5_seed42_1a2b3c`).
- **`fit_uf3.py` was cleaned up** (docstrings, single-pass sampling, unit-tagged
  metrics) — no change to the CLI or the fit math; still ruff-clean.

## Cores
Featurization is parallelized across CPU cores. By default it uses all
cores in your allocation (`len(os.sched_getaffinity)` on Linux); set it
explicitly with `--n-jobs N` on any run. So request several cores
(`--cpus-per-task 16`) and it will use them.

## Setup (once)
```bash
git clone https://github.com/daniisler/umlff-for-molten-salts.git
cd umlff-for-molten-salts
git checkout marios/uf3-reproduction
# put umlff_data_archive.zip in the repo root, then:
unzip umlff_data_archive.zip          # creates ./data/
uv sync                                # locked env, Python 3.12.14
```
If `uv` is missing: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `pip install uv`).

## Smoke test (~1 min — do this first)
```bash
uv run python scripts/fit_uf3.py --mode 2body --n-train 30 --n-test 15 \
  --train-file sub_KRbCl_train.extxyz --test-file sub_KRbCl_test.extxyz
```
Expect a `RESULT ...` line and a `results/2body_n30_c8_w0.5_seed42_<hash>/` folder.

## Runs

Key idea: **the 2-body is limited by the number of frames** (RAM grows with
frames), so subsample frames. **The full 3-body is limited by the number of
triplet features** (~111k → a ~100 GB matrix), which is independent of frames —
so subsampling frames does NOT make it fit. Different levers.

### 2-body — every ~10th frame (full coverage, cheap)
Every 10th of 68,400 ≈ 6,840 frames (~30 GB RAM, fits a normal node). The script
samples by count, not stride, but a seeded random 6,840 gives the same coverage.
The 2-body force error plateaus, so this is effectively the "all data" answer.
`Training.extxyz` and `Validation.extxyz` are already disjoint files:
```bash
uv run python scripts/fit_uf3.py --mode 2body --n-train 6840 --n-test 5820
```
(`--n-test 5820` = the full Validation set; use `2000` for a lighter test.)

### 2-body frame-count sweep — does more data help? (validation)
Open question from the Week-4 review: the 2-body error looks flat, but does it
keep improving — or drift UP — with more frames? To read a sweep you first need
the noise floor: **10 random 200-frame draws on Cs–Zr–Cl gave force RMSE
267.5 ± 1.4 meV/Å (total spread 4.8)**. So in the sweeps below, a change
**bigger than ~3 meV/Å is a real data-scaling effect; anything smaller is just
which frames got picked.** Keep the test set and the seed FIXED across each sweep.

**A. Cs–Zr–Cl subsystem (cheap; apples-to-apples with the noise floor).**
Capped by how many Cs–Zr–Cl frames exist in the DFT data (~2,975 total → ~2,380
train), so this only reaches ~2,000 — it refines the within-subsystem curve, it
does NOT reach "lots more frames" (that is part B). Runs fine on a laptop.
```bash
for n in 200 400 800 1600 2000; do
  uv run python scripts/fit_uf3.py --mode 2body --n-train $n --n-test 595 \
    --train-file sub_CsZrCl_train.extxyz --test-file sub_CsZrCl_test.extxyz \
    --elements Cl Cs Zr --seed 42 --no-predictions
done
```

**B. Full 12-element set (the real "more frames" curve — this is the cluster job).**
`Training.extxyz` (68,400 frames) and `Validation.extxyz` are already disjoint,
so no split needed. This is where more frames actually live. **RAM caution:** for
the 2-body, memory grows with FRAMES, not features — the design matrix is roughly
frames × (1 + 3N) × n_features × 8 bytes (~1,800 features at res2=20). ~10k frames
is tens of GB and the `--max-features` guard does NOT catch it (that only guards
feature count), so request a high-memory node (≥128 GB for ≥10k frames) and step
the sweep up while watching usage.
```bash
for n in 500 1000 2000 5000 10000 20000; do
  uv run python scripts/fit_uf3.py --mode 2body --n-train $n --n-test 2000 \
    --train-file Training.extxyz --test-file Validation.extxyz \
    --seed 42 --no-predictions
done
```
Fixed `--n-test 2000` (a slice of Validation) for comparability — drop to 500 if
the test featurization is heavy. No `--elements` → it auto-detects all 12.

**Lower-RAM option — `fit_uf3_batched.py` (the "split-RAM" fit).** This runs the
*exact same* 2-body fit (coefficients verified identical to ~1e-7) but accumulates
the gram X^T W X over frame batches, so peak RAM is one batch (`--frame-batch`,
e.g. 500) plus the small gram — **independent of --n-train**. Use it for the big
full-set sweeps and the ≥128 GB node is no longer required (a normal node works):
```bash
for n in 500 1000 2000 5000 10000 20000; do
  uv run python scripts/fit_uf3_batched.py --mode 2body --n-train $n --n-test 2000 \
    --train-file Training.extxyz --test-file Validation.extxyz \
    --frame-batch 500 --seed 42 --no-predictions
done
```
Only training is batched; keep `--n-test` modest (featurized in one pass). Same
CLI, outputs and metrics as `fit_uf3.py`.

**Reading it:** plot force RMSE vs n for each sweep. Flat within ±3 meV/Å →
data-saturated, the expected 2-body bias limit. A real climb of ≥5–10 → the rigid
pair curve is being pulled around by more/harder configs. Either outcome is a
clean learning-curve result; the ±1.4 floor is the yardstick for "real vs noise".

### Full 12-species 3-body — feasibility probe (feature-limited, not frame-limited)
Subsampling frames will not help here — the matrix is ~111k features wide
regardless. Run the probe; it prints the feature count + RAM instantly and does
not fit:
```bash
uv run python scripts/fit_uf3.py --mode 3body --probe-only --n-train 50
```
Only a reduced version might fit, on a very-high-memory node:
```bash
uv run python scripts/fit_uf3.py --mode 3body --n-train 2000 --n-test 500 \
  --cutoff3 3.5 --res3 4 --force
```

### 3-body on single-salt subsystems — the real 3-body result
Only 18 triplet types, so it is tractable. Each subsystem's train file has
~2,000 frames, so ~1,500 is plenty. Zr and K/Rb splits already exist; make Mg's
once with `split_frames.py`. Run all three:
```bash
# Mg (directional) — make the disjoint split once, then fit
uv run python scripts/split_frames.py --input sub_CaMgCl.extxyz
uv run python scripts/fit_uf3.py --mode 3body --n-train 1500 --n-test 400 \
  --train-file sub_CaMgCl_train.extxyz --test-file sub_CaMgCl_test.extxyz \
  --elements Ca Cl Mg --cutoff3 5.0 --res3 6

# Zr (directional) — split already in data/
uv run python scripts/fit_uf3.py --mode 3body --n-train 1500 --n-test 400 \
  --train-file sub_CsZrCl_train.extxyz --test-file sub_CsZrCl_test.extxyz \
  --elements Cl Cs Zr --cutoff3 5.0 --res3 6

# K/Rb (spherical control) — split already in data/
uv run python scripts/fit_uf3.py --mode 3body --n-train 1500 --n-test 400 \
  --train-file sub_KRbCl_train.extxyz --test-file sub_KRbCl_test.extxyz \
  --elements Cl K Rb --cutoff3 5.0 --res3 6
```

For a matched 2-body baseline on each subsystem, rerun the three with
`--mode 2body` and the SAME `*_train` / `*_test` pair (the difference is the
value of the 3-body term).

### 3-body learning curve on a subsystem (does the 3-body saturate too?)
The companion to the 2-body frame sweep. The 2-body force error *rises* with more
frames (it is misspecified — blind to angles). The question: does the 3-body, which
*can* see angles, instead keep improving or plateau low? Run it on Cs–Zr–Cl with the
batched script so RAM is a non-issue (subsystem gram ~3,174² ≈ 80 MB):
```bash
for n in 60 120 250 500 1000 1500; do
  uv run python scripts/fit_uf3_batched.py --mode 3body --n-train $n --n-test 400 \
    --train-file sub_CsZrCl_train.extxyz --test-file sub_CsZrCl_test.extxyz \
    --elements Cl Cs Zr --cutoff3 5.0 --res3 6 --frame-batch 300 --seed 42 --no-predictions
done
```
**Here the limit is TIME, not RAM** — 3-body featurization is ~11 s/frame, so the big
n take hours. This is a cluster/background job (`sbatch`), not a laptop one. For an
apples-to-apples contrast, run the matching 2-body curve on the SAME frames
(`--mode 2body`, same n / seed / test file) and plot the two force curves together —
2-body should rise, 3-body should fall or plateau low. Reference points already
measured (Cs–Zr–Cl, seed 42): **3-body n=60 → F 117.82 / E 3.85**; 2-body n=2000 →
F 296 — the 3-body with 60 frames already beats the 2-body with 2000.

### Add or enlarge subsystems (test more / bigger systems)
The `sub_*` files are just filtered slices of `Training.extxyz`. Build new or
larger subsystems with `make_subsystem.py`, split them, then fit:
```bash
# build a new subsystem (writes data/sub_CaClZn.extxyz)
uv run python scripts/make_subsystem.py --elements Cl Ca Zn

# split into disjoint train/test (writes data/sub_CaClZn_train.extxyz + _test.extxyz)
uv run python scripts/split_frames.py --input sub_CaClZn.extxyz

# check its 3-body size/RAM BEFORE fitting (more elements = more triplet types)
uv run python scripts/fit_uf3.py --mode 3body --probe-only \
  --train-file sub_CaClZn_train.extxyz --elements Ca Cl Zn --cutoff3 5.0 --res3 6

# fit it (use as many frames as the subsystem has)
uv run python scripts/fit_uf3.py --mode 3body --n-train 3000 --n-test 500 \
  --train-file sub_CaClZn_train.extxyz --test-file sub_CaClZn_test.extxyz \
  --elements Ca Cl Zn --cutoff3 5.0 --res3 6
```
A 4-element subsystem qualifies *more* frames (bigger dataset) but also has more
triplet types. Run `--probe-only` first with the SAME `--n-train`/`--elements`
you plan to use — it prints the estimated **peak RAM** (design matrix + gram) for
that exact run, so you can size the node before submitting. Good directional cations to add:
**Zn, Zr, Mg**; spherical controls: **Cs, Rb, K, Na**.

## Output
Each run writes `results/<run_name>/` (the folder name encodes the full config):
- `metrics.json` — config + energy MAE + force RMSE + timing
- `model.json` — the fitted coefficients
- `predictions.npz` — per-structure/atom predicted vs reference (for parity plots)

## Running the matrix in parallel
`scripts/run_jobs.sh` is a **plain shell runner** (no scheduler needed): it runs each
line of `scripts/jobs.txt` through the batched (split-RAM) fit, N at a time, so many
large runs go at once and big frame counts fit in memory. `jobs.txt` holds the matrix:
- **Track A (lines 1-12):** 2-body on the full 12-element set, n = 2k/5k/10k/20k x 3
  seeds each -- "different large sets" at scale (learning curve + seed spread).
- **Track B (lines 13-21):** 3-body on Cs-Zr-Cl, n = 500/1000/1500 x 3 seeds,
  cutoff3=5 res3=6 -- checks the 3-body result with more data.

One-time setup (also builds the fixed 2k test slice Track A needs):
```bash
uv sync && unzip umlff_data_archive.zip
uv run python -c "from ase.io import read,write; import numpy as np; \
  fr=read('data/Validation.extxyz',':'); i=sorted(np.random.default_rng(0).permutation(len(fr))[:2000]); \
  write('data/Validation_test2k.extxyz',[fr[k] for k in i])"
```
Then just run it (tune `N_PARALLEL`/`CORES_PER_JOB` to the box; `RUN` overrides how
python is launched, default `uv run python`):
```bash
bash scripts/run_jobs.sh                 # everything, N_PARALLEL at a time
N_PARALLEL=8 bash scripts/run_jobs.sh    # 8 jobs at once
bash scripts/run_jobs.sh 1 12            # 2-body only (lines 1..12)
bash scripts/run_jobs.sh 13 21           # 3-body only (lines 13..21)
```
Each job's stdout/stderr goes to `logs/job_NN.log`. (For an actual SLURM cluster later,
`scripts/run_slurm.sh` wraps the same jobs.txt as a job array — adapt its `#SBATCH`
lines to that site; the plain runner above is all you need on a normal server.)
Each task writes `results/<run_name>/metrics.json` (self-naming, no clobber). Gather
them into one table when done:
```bash
uv run python -c "import json,glob,csv; \
  R=[json.load(open(p)) for p in glob.glob('results/*/metrics.json')]; \
  w=csv.writer(open('results/summary.csv','w',newline='')); \
  w.writerow(['mode','elements','n_train','seed','force_rmse_meV_A','energy_mae_meV_at','wall_s']); \
  [w.writerow([r['config']['mode'],'-'.join(r['elements']),r['n_train'],r['config']['seed'], \
    r['metrics']['force_rmse_meV_per_ang'],r['metrics']['energy_mae_meV_per_atom'],r['wall_time_s']]) for r in R]; \
  print('wrote results/summary.csv', len(R), 'runs')"
```
Add more experiments by appending flag-lines to `jobs.txt` (e.g. the cutoff3/res3
sweep once the local preview picks a range). UF3 is CPU-bound -- no `--gres=gpu`.
