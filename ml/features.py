"""
Feature engineering for hand-gesture recognition.

MediaPipe Hands gives 21 landmarks per frame, each (x, y, z) -> 63 numbers.
Raw coordinates are useless to a classifier on their own: they depend on where
the hand is in the frame and how big it looks. So we normalise every frame to be
*translation- and scale-invariant*:

  1. Translate so the wrist (landmark 0) is the origin.
  2. Scale so the distance wrist -> middle-finger MCP (landmark 9) is 1.
  3. Flatten to a 63-vector.

A gesture is a short *sequence* of these frames (a window), so the model sees
motion, not just a pose. This module is pure numpy and fully unit-testable.
"""

import numpy as np

NUM_LANDMARKS = 21
NUM_COORDS = 3
FEATURES_PER_FRAME = NUM_LANDMARKS * NUM_COORDS          # 63
WINDOW = 16                                              # frames per sample
WRIST = 0
MIDDLE_MCP = 9

# Gesture vocabulary (index == class id). Order is fixed once you train.
# To add a gesture: append a name here, recollect data, retrain. The model and
# collector size themselves from this list automatically.
LABELS = ["idle", "draw", "stop", "clear", "save", "undo"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}


def normalize_frame(landmarks):
    """landmarks: (21, 3) array -> (63,) translation/scale-invariant vector."""
    lm = np.asarray(landmarks, dtype=np.float32).reshape(NUM_LANDMARKS, NUM_COORDS)
    lm = lm - lm[WRIST]                                  # wrist at origin
    scale = np.linalg.norm(lm[MIDDLE_MCP]) + 1e-6        # hand size
    lm = lm / scale
    return lm.reshape(-1)


def normalize_window(frames):
    """frames: (T, 21, 3) or (T, 63) -> (WINDOW, 63) float32.

    Normalises each frame, then resamples the time axis to exactly WINDOW frames
    so gestures of slightly different speed line up."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim == 3:
        frames = np.stack([normalize_frame(f) for f in frames])
    elif frames.ndim == 2 and frames.shape[1] == FEATURES_PER_FRAME:
        frames = np.stack([normalize_frame(f) for f in frames])
    else:
        raise ValueError(f"unexpected window shape {frames.shape}")

    t = frames.shape[0]
    if t == WINDOW:
        return frames
    # linear resample along time
    src = np.linspace(0, 1, t)
    dst = np.linspace(0, 1, WINDOW)
    out = np.stack([np.interp(dst, src, frames[:, c])
                    for c in range(FEATURES_PER_FRAME)], axis=1)
    return out.astype(np.float32)


def augment_window(window, rng, jitter=0.02, scale=0.08, time_shift=2):
    """Light data augmentation for training robustness:
    small gaussian jitter, a global scale wobble, and a tiny time shift."""
    w = window.copy()
    w += rng.normal(0, jitter, w.shape).astype(np.float32)   # sensor noise
    w *= (1.0 + rng.uniform(-scale, scale))                  # hand-size wobble
    s = rng.integers(-time_shift, time_shift + 1)
    if s:
        w = np.roll(w, s, axis=0)
    return w
