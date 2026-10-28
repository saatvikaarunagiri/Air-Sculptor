"""
The gesture classifier: a small 1-D temporal CNN.

Input : (batch, 63, WINDOW)   -- 63 landmark features over time
Output: (batch, num_classes)  -- gesture logits

A 1-D CNN over the time axis learns short motion patterns (a pinch closing, a
palm opening) directly from landmark trajectories. It's tiny (~50k params), so
it trains in seconds on an M5 and runs in well under a millisecond per window.
"""

import torch
import torch.nn as nn

from features import FEATURES_PER_FRAME, WINDOW, LABELS


class GestureNet(nn.Module):
    def __init__(self, num_classes=len(LABELS), in_feats=FEATURES_PER_FRAME):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_feats, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),            # global pooling over time
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, WINDOW, 63) -> (batch, 63, WINDOW) for Conv1d
        if x.dim() == 3 and x.shape[1] == WINDOW:
            x = x.transpose(1, 2)
        return self.head(self.net(x))


def pick_device():
    """Prefer Apple Silicon GPU (MPS) on an M-series Mac, else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
