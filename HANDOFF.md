# Running the UF3 jobs — handoff

Everything needed to run the UF3 fit/evaluate script on the cluster. Full
rationale, job matrix and expected numbers are in `docs/uf3_cluster_run_plan.md`.

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
  --train-file sub_KRbCl.extxyz --test-file sub_KRbCl.extxyz
```
Expect a `RESULT ...` line and a `results/2body_n30_seed42/` folder.

## Runs

Key idea: **the 2-body is limited by the number of frames** (RAM grows with
frames), so subsample frames. **The full 3-body is limited by the number of
triplet features** (~111k → a ~100 GB matrix), which is independent of frames —
so subsampling frames does NOT make it fit. Different levers.

### 2-body — every ~10th frame (full coverage, cheap)
Every 10th of 68,400 ≈ 6,840 frames (~30 GB RAM, fits a normal node). The script
samples by count, not stride, but a seeded random 6,840 gives the same coverage.
The 2-body force error plateaus, so this is effectively the "all data" answer.
```bash
uv run python scripts/fit_uf3.py --mode 2body --n-train 6840 --n-test 5820
```
(`--n-test 5820` = the full Validation set; use `2000` for a lighter test.)

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
Only 18 triplet types, so it is tractable. Each subsystem file has ~2,500 frames,
so ~1,500 is plenty. Run all three:
```bash
# Mg (directional)
uv run python scripts/fit_uf3.py --mode 3body --n-train 1500 --n-test 400 \
  --train-file sub_CaMgCl.extxyz --test-file sub_CaMgCl.extxyz \
  --elements Ca Cl Mg --cutoff3 5.0 --res3 6

# Zr (directional)
uv run python scripts/fit_uf3.py --mode 3body --n-train 1500 --n-test 400 \
  --train-file sub_CsZrCl.extxyz --test-file sub_CsZrCl.extxyz \
  --elements Cl Cs Zr --cutoff3 5.0 --res3 6

# K/Rb (spherical control)
uv run python scripts/fit_uf3.py --mode 3body --n-train 1500 --n-test 400 \
  --train-file sub_KRbCl.extxyz --test-file sub_KRbCl.extxyz \
  --elements Cl K Rb --cutoff3 5.0 --res3 6
```

For a matched 2-body baseline on each subsystem, rerun the three with
`--mode 2body` (the difference is the value of the 3-body term).

### Add or enlarge subsystems (test more / bigger systems)
The `sub_*` files are just filtered slices of `Training.extxyz`. Build new or
larger subsystems with `make_subsystem.py`, then fit them:
```bash
# build a new subsystem (writes data/sub_CaClZn.extxyz)
uv run python scripts/make_subsystem.py --elements Cl Ca Zn

# check its 3-body size/RAM BEFORE fitting (more elements = more triplet types)
uv run python scripts/fit_uf3.py --mode 3body --probe-only \
  --train-file sub_CaClZn.extxyz --elements Ca Cl Zn --cutoff3 5.0 --res3 6

# fit it (use as many frames as the subsystem has)
uv run python scripts/fit_uf3.py --mode 3body --n-train 3000 --n-test 500 \
  --train-file sub_CaClZn.extxyz --test-file sub_CaClZn.extxyz \
  --elements Ca Cl Zn --cutoff3 5.0 --res3 6
```
A 4-element subsystem qualifies *more* frames (bigger dataset) but also has more
triplet types, so always run `--probe-only` first. Good directional cations to add:
**Zn, Zr, Mg**; spherical controls: **Cs, Rb, K, Na**.

## Output
Each run writes `results/<run_name>/`:
- `metrics.json` — config + energy MAE + force RMSE + timing
- `model.json` — the fitted coefficients
- `predictions.npz` — per-structure/atom predicted vs reference (for parity plots)

Send the `results/` folder back.

## SLURM
Edit `scripts/run_slurm.sh` for the cluster's partition/module names, pick a run,
then `sbatch scripts/run_slurm.sh`.
