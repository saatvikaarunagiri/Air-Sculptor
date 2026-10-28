"""
Feature engineering for facial-expression recognition.

MediaPipe FaceMesh gives 468 landmarks per frame, each (x, y, z). Expression is
a *pose* (a single frame suffices), so each sample is one normalised face:

  1. Translate so the nose tip (landmark 1) is the origin.
  2. Scale by the inter-ocular distance (outer eye corners 33 & 263) so it's
     invariant to how close/far the face is.
  3. Flatten to a 1404-vector (468 * 3).

Pure numpy and unit-testable.
"""

import numpy as np

NUM_LANDMARKS = 468
NUM_COORDS = 3
FEATURES = NUM_LANDMARKS * NUM_COORDS        # 1404
NOSE = 1
EYE_L, EYE_R = 33, 263                        # outer eye corners

# Expression vocabulary (index == class id). Extend + recollect + retrain.
LABELS = ["neutral", "happy", "sad", "surprised"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}


def normalize_face(landmarks):
    """landmarks: (468, 3) -> (1404,) translation/scale-invariant vector."""
    lm = np.asarray(landmarks, dtype=np.float32).reshape(NUM_LANDMARKS, NUM_COORDS)
    lm = lm - lm[NOSE]                                    # nose at origin
    eye_dist = np.linalg.norm(lm[EYE_L] - lm[EYE_R]) + 1e-6
    lm = lm / eye_dist                                    # scale by face width
    return lm.reshape(-1)


def augment_face(vec, rng, jitter=0.01, scale=0.05):
    """Light augmentation for the training split only."""
    v = vec.copy()
    v = v + rng.normal(0, jitter, v.shape).astype(np.float32)
    v = v * (1.0 + rng.uniform(-scale, scale))
    return v.astype(np.float32)
