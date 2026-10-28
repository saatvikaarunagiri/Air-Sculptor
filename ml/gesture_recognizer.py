"""
Live inference wrapper — the bridge between the trained model and the app.

Feed it one frame of landmarks at a time via push(); it keeps a rolling window
and returns the current smoothed gesture prediction. Loads lazily and fails
soft: if torch or the model file is missing, available=False and the app falls
back to the classic pinch-threshold.
"""

import os
import sys
from collections import deque, Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import WINDOW, LABELS, normalize_window      # noqa: E402

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


class GestureRecognizer:
    def __init__(self, smooth=5, min_conf=0.6):
        self.available = False
        self.labels = LABELS
        self.status = "not loaded"
        self._buf = deque(maxlen=WINDOW)
        self._recent = deque(maxlen=smooth)      # majority-vote smoothing
        self.min_conf = min_conf
        self._torch = None
        self._model = None
        self._device = None
        self._load()

    def _load(self):
        try:
            import torch
            from model import GestureNet, pick_device
            path = os.path.join(ART, "gesture_model.pt")
            if not os.path.exists(path):
                self.status = "no model file (train first)"
                return
            self._torch = torch
            self._device = pick_device()
            ckpt = torch.load(path, map_location=self._device)
            self.labels = ckpt.get("labels", LABELS)
            self._model = GestureNet(num_classes=len(self.labels)).to(self._device)
            self._model.load_state_dict(ckpt["state_dict"])
            self._model.eval()
            self.available = True
            self.status = "ready"
        except Exception as e:
            self.status = f"unavailable ({type(e).__name__})"

    def push(self, landmarks_21x3):
        """Add one frame; return (label, confidence) or (None, 0.0)."""
        if not self.available:
            return None, 0.0
        self._buf.append(np.asarray(landmarks_21x3, dtype=np.float32))
        if len(self._buf) < WINDOW:
            return None, 0.0
        window = normalize_window(np.stack(self._buf))
        with self._torch.no_grad():
            x = self._torch.from_numpy(window[None]).to(self._device)
            probs = self._torch.softmax(self._model(x), dim=1)[0].cpu().numpy()
        cls = int(probs.argmax())
        conf = float(probs[cls])
        label = self.labels[cls] if conf >= self.min_conf else "idle"
        self._recent.append(label)
        # majority vote over the last few predictions
        voted = Counter(self._recent).most_common(1)[0][0]
        return voted, conf

    def reset(self):
        self._buf.clear()
        self._recent.clear()
