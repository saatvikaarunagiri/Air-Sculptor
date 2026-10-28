"""
Generate SYNTHETIC gesture data so you can test the full train/evaluate
pipeline before (or without) collecting real webcam data.

Each gesture gets a characteristic hand pose with a little motion and noise. This
is NOT a substitute for real data — it just proves the pipeline end to end and
lets you see what good metrics look like. Collect real data with collect_data.py
for the actual app.

    python make_synthetic_data.py --per-class 120
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import LABELS, NUM_LANDMARKS                  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# A rough canonical hand (wrist at origin-ish); each gesture perturbs fingertips.
_BASE = np.array([
    [0.50, 0.90, 0.0],   # 0 wrist
    [0.42, 0.80, 0.0], [0.38, 0.72, 0.0], [0.35, 0.65, 0.0], [0.33, 0.59, 0.0],  # thumb
    [0.48, 0.66, 0.0], [0.47, 0.55, 0.0], [0.46, 0.48, 0.0], [0.46, 0.42, 0.0],  # index
    [0.52, 0.65, 0.0], [0.52, 0.53, 0.0], [0.52, 0.45, 0.0], [0.52, 0.39, 0.0],  # middle
    [0.56, 0.66, 0.0], [0.57, 0.55, 0.0], [0.58, 0.48, 0.0], [0.58, 0.43, 0.0],  # ring
    [0.60, 0.68, 0.0], [0.62, 0.59, 0.0], [0.63, 0.53, 0.0], [0.64, 0.49, 0.0],  # pinky
], dtype=np.float32)

THUMB_TIP, INDEX_TIP = 4, 8


def _pose(label, t):
    """Return a (21,3) pose for `label` at phase t in [0,1]."""
    h = _BASE.copy()
    if label == "draw":           # thumb + index pinch closing over time
        close = 0.08 * (1 - t)
        h[INDEX_TIP, :2] = [0.40 + close, 0.60]
        h[THUMB_TIP, :2] = [0.40 - close, 0.60]
    elif label == "stop":         # open palm, fingers spread wide
        h[[8, 12, 16, 20], 0] += np.array([-0.05, 0, 0.05, 0.10])
        h[[8, 12, 16, 20], 1] -= 0.04
    elif label == "clear":        # fist: fingertips curl toward palm
        for tip in [8, 12, 16, 20]:
            h[tip, 1] += 0.22
    elif label == "save":         # thumbs up: thumb up, fingers curled
        h[THUMB_TIP, 1] -= 0.18
        for tip in [8, 12, 16, 20]:
            h[tip, 1] += 0.20
    # idle: base pose as-is
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--frames", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    for label in LABELS:
        d = os.path.join(DATA_DIR, label)
        os.makedirs(d, exist_ok=True)
        for i in range(args.per_class):
            T = args.frames + int(rng.integers(-3, 4))      # varying length
            seq = []
            for f in range(T):
                pose = _pose(label, f / max(T - 1, 1))
                pose = pose + rng.normal(0, 0.012, pose.shape).astype(np.float32)
                pose[:, :2] += rng.normal(0, 0.03, 2)        # whole-hand shift
                seq.append(pose)
            arr = np.stack(seq).astype(np.float32)
            np.save(os.path.join(d, f"syn_{i}_{int(time.time()*1e6)%10**6}.npy"),
                    arr)
    print("Synthetic data written. Per-class counts:")
    for label in LABELS:
        print(f"  {label:6}: {len(os.listdir(os.path.join(DATA_DIR, label)))}")


if __name__ == "__main__":
    main()
