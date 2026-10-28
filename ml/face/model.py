"""
Facial-expression classifier: an MLP over normalised face landmarks.

Input : (batch, 1404)   -- one normalised face
Output: (batch, num_classes)

Expression is a static pose, so a straightforward MLP on the landmark vector
works well and stays tiny. Trains in seconds on an M-series GPU (MPS).
"""

import torch
import torch.nn as nn

from features import FEATURES, LABELS


class FaceEmotionNet(nn.Module):
    def __init__(self, num_classes=len(LABELS), in_feats=FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
