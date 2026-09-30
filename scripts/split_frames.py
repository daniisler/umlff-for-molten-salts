"""Split an .extxyz trajectory into disjoint train/test files (seeded, frame-level).

Sampling train and test frames from the *same* file leaks information: a frame
used to fit the model can reappear in evaluation, so the reported error looks
better than the model really is. This script makes the split once, up front, so
train and test never share a frame.

Frames are shuffled with a seeded generator and cut at ``1 - test_frac`` : ``test_frac``
(default 80/20). Given the same input, seed and fraction, the split is identical
every time. Output goes next to the input as ``<stem>_train.extxyz`` and
``<stem>_test.extxyz`` (override with ``--out-prefix``).

Typical use::

    python scripts/split_frames.py --input sub_CaMgCl.extxyz
    python scripts/split_frames.py --input sub_CaMgCl.extxyz --test-frac 0.2 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def split_indices(n_frames: int, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint (train, test) index arrays from a seeded shuffle of ``range(n_frames)``."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_frames)
    n_test = int(round(n_frames * test_frac))
    test_idx = np.sort(order[:n_test])
    train_idx = np.sort(order[n_test:])
    assert set(train_idx.tolist()).isdisjoint(test_idx.tolist())  # never share a frame
    return train_idx, test_idx


def main() -> None:
    """Parse arguments, read the frames, and write the disjoint train/test files."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="input .extxyz file name inside --data-dir")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="folder holding the data files")
    parser.add_argument("--test-frac", type=float, default=0.2, help="fraction of frames held out for testing")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the reproducible shuffle")
    parser.add_argument("--out-prefix", default=None, help="output stem (default: input stem, e.g. sub_CaMgCl)")
    args = parser.parse_args()

    in_path = args.data_dir / args.input
    frames = read(str(in_path), index=":")  # read every frame
    train_idx, test_idx = split_indices(len(frames), args.test_frac, args.seed)

    stem = args.out_prefix or in_path.stem
    train_path = args.data_dir / f"{stem}_train.extxyz"
    test_path = args.data_dir / f"{stem}_test.extxyz"
    write(str(train_path), [frames[i] for i in train_idx])
    write(str(test_path), [frames[i] for i in test_idx])

    print(
        f"[split] {args.input}: {len(frames)} frames -> "
        f"{len(train_idx)} train + {len(test_idx)} test (seed {args.seed})"
    )
    print(f"[split] wrote {train_path.name} and {test_path.name}")


if __name__ == "__main__":
    main()
