"""
Generate SYNTHETIC face-landmark data to dry-run the pipeline without a webcam.

Builds a rough neutral face mesh, then perturbs mouth/eye/brow regions per
expression. NOT a substitute for real data — it just proves train/evaluate run.

    python make_synthetic_data.py --per-class 250
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import LABELS, NUM_LANDMARKS                   # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Approx landmark indices for expressive regions (MediaPipe FaceMesh).
MOUTH = [61, 291, 0, 17, 13, 14, 78, 308]       # corners, top/bottom lips
BROW = [55, 65, 105, 285, 295, 334]             # inner/outer brows
EYE = [159, 145, 386, 374]                       # upper/lower eyelids


# One fixed canonical face shared by ALL samples, so the only consistent signal
# across a class is its expression perturbation (plus small per-sample jitter).
# Generating a fresh random face per sample would drown the expression in noise.
_CANON = np.random.default_rng(12345).normal(0.5, 0.1,
                                             (NUM_LANDMARKS, 3)).astype(np.float32)
_CANON[1] = [0.5, 0.5, 0.0]                       # nose at center
_CANON[33] = [0.38, 0.45, 0.0]                    # left eye corner
_CANON[263] = [0.62, 0.45, 0.0]                   # right eye corner


def _base_face(rng):
    """The canonical face with per-sample variation simulating different
    people/poses, so features aren't degenerate (real faces vary everywhere)."""
    face = _CANON.copy()
    face += rng.normal(0, 0.02, face.shape).astype(np.float32)   # person/pose
    face *= (1.0 + rng.uniform(-0.05, 0.05))                     # size wobble
    face[:, :2] += rng.normal(0, 0.02, 2)                        # shift
    return face


def _apply(face, label, rng):
    f = face.copy()
    if label == "happy":                         # mouth corners up & wide
        for i in (61, 291):
            f[i, 1] -= 0.09; f[i, 0] += (0.07 if i == 291 else -0.07)
        f[13, 1] -= 0.03
    elif label == "sad":                         # mouth corners down, inner brow up
        for i in (61, 291):
            f[i, 1] += 0.09
        for i in (55, 285):
            f[i, 1] -= 0.06
    elif label == "surprised":                   # mouth open, brows up, eyes wide
        f[14, 1] += 0.10; f[13, 1] -= 0.07       # lips apart
        for i in BROW:
            f[i, 1] -= 0.10
        for i in EYE:
            f[i, 1] += (0.05 if i in (145, 374) else -0.05)
    # neutral: unchanged
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    for label in LABELS:
        d = os.path.join(DATA_DIR, label)
        os.makedirs(d, exist_ok=True)
        for i in range(args.per_class):
            face = _apply(_base_face(rng), label, rng)
            face += rng.normal(0, 0.006, face.shape).astype(np.float32)
            np.save(os.path.join(d, f"syn_{i}_{int(time.time()*1e6)%10**6}.npy"),
                    face.astype(np.float32))
    print("Synthetic face data written:")
    for label in LABELS:
        print(f"  {label:10}: {len(os.listdir(os.path.join(DATA_DIR, label)))}")


if __name__ == "__main__":
    main()
