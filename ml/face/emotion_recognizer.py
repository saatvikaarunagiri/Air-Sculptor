"""
Live inference wrapper for facial-expression recognition.

Feed it one frame of face landmarks via push(); returns the current smoothed
expression. Fails soft: missing torch/model -> available=False and the app
simply doesn't show a mood.
"""

import os
import sys
from collections import deque, Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import normalize_face, LABELS                  # noqa: E402

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


class EmotionRecognizer:
    def __init__(self, smooth=8, min_conf=0.5):
        self.available = False
        self.labels = LABELS
        self.status = "not loaded"
        self.min_conf = min_conf
        self._recent = deque(maxlen=smooth)
        self._torch = None
        self._model = None
        self._device = None
        self._load()

    def _load(self):
        try:
            import torch
            from model import FaceEmotionNet, pick_device
            path = os.path.join(ART, "emotion_model.pt")
            if not os.path.exists(path):
                self.status = "no model file (train first)"
                return
            self._torch = torch
            self._device = pick_device()
            ckpt = torch.load(path, map_location=self._device)
            self.labels = ckpt.get("labels", LABELS)
            self._model = FaceEmotionNet(num_classes=len(self.labels)).to(self._device)
            self._model.load_state_dict(ckpt["state_dict"])
            self._model.eval()
            self.available = True
            self.status = "ready"
        except Exception as e:
            self.status = f"unavailable ({type(e).__name__})"

    def push(self, landmarks_468x3):
        """Return (label, confidence) or (None, 0.0)."""
        if not self.available:
            return None, 0.0
        vec = normalize_face(landmarks_468x3)
        with self._torch.no_grad():
            x = self._torch.from_numpy(vec[None]).to(self._device)
            probs = self._torch.softmax(self._model(x), dim=1)[0].cpu().numpy()
        cls = int(probs.argmax())
        conf = float(probs[cls])
        label = self.labels[cls] if conf >= self.min_conf else "neutral"
        self._recent.append(label)
        voted = Counter(self._recent).most_common(1)[0][0]
        return voted, conf

    def reset(self):
        self._recent.clear()
