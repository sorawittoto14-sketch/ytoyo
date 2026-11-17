"""Small CNN that scores a 5-s low-frequency spectrogram for rumble presence."""

import torch
from torch import nn


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class RumbleCNN(nn.Module):
    """Input (B, 1, F, T) -> logit (B,). Pools over time with mean+max so a
    rumble occupying only part of the window still triggers."""

    def __init__(self, width: int = 16, dropout: float = 0.3):
        super().__init__()
        w = width
        self.features = nn.Sequential(_block(1, w), _block(w, 2 * w), _block(2 * w, 4 * w), _block(4 * w, 6 * w))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(12 * w, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).mean(dim=2)            # average over frequency -> (B, C, T')
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)  # mean + max over time
        return self.head(h).squeeze(1)
