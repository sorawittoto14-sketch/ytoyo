"""SincNet front-end + bidirectional LSTM for elephant rumble detection.

Raw waveform (250 Hz) --> SincConv (learnable band-pass filters kept inside
the elephant infrasound band, 10-60 Hz) --> |.| --> pooling + log compression
--> two 1-D conv blocks (down to 5 frames/s) --> 2-layer BiLSTM that follows
the temporal shape of the call --> per-frame rumble logit.

The window ("clip") score is the maximum frame score, so the same network
gives both "is there a rumble in this window?" and where it starts/ends.

SincConv follows Ravanelli & Bengio, "Speaker Recognition from Raw Waveform
with SincNet" (2018).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import BAND_HZ, FRAME_HOP, SAMPLE_RATE


class SincConv(nn.Module):
    """1-D convolution whose kernels are parametrised band-pass sinc filters.

    Only two numbers per filter are learned: the low cut-off and the bandwidth.
    Every filter is constrained to lie inside [f_low, f_high].
    """

    def __init__(self, n_filters=32, kernel_size=251, sample_rate=SAMPLE_RATE,
                 f_low=BAND_HZ[0], f_high=BAND_HZ[1], min_band_hz=2.0):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size, self.sample_rate = kernel_size, sample_rate
        self.f_low, self.f_high, self.min_band_hz = f_low, f_high, min_band_hz

        edges = np.geomspace(f_low, f_high, n_filters + 1)
        self.low_hz_ = nn.Parameter(torch.tensor(edges[:-1] - f_low, dtype=torch.float32).view(-1, 1))
        band = np.maximum(np.diff(edges) - min_band_hz, 0.5)
        self.band_hz_ = nn.Parameter(torch.tensor(band, dtype=torch.float32).view(-1, 1))

        half = kernel_size // 2
        n_lin = torch.linspace(0, half - 1, steps=half)
        self.register_buffer("window_", 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size))
        self.register_buffer("n_", 2 * math.pi * torch.arange(-half, 0).view(1, -1) / sample_rate)

    def cutoffs(self):
        low = torch.clamp(self.f_low + self.low_hz_.abs(), max=self.f_high - self.min_band_hz)
        high = torch.clamp(low + self.min_band_hz + self.band_hz_.abs(), max=self.f_high)
        return low, high

    def filters(self):
        low, high = self.cutoffs()
        band = (high - low)[:, 0]
        left = (torch.sin(high @ self.n_) - torch.sin(low @ self.n_)) / (self.n_ / 2) * self.window_
        center = 2 * band.view(-1, 1)
        bp = torch.cat([left, center, torch.flip(left, dims=[1])], dim=1) / (2 * band[:, None])
        return bp.view(-1, 1, self.kernel_size)

    def forward(self, x):  # (B, 1, S) -> (B, n_filters, S)
        return F.conv1d(x, self.filters(), padding=self.kernel_size // 2)


def _conv_block(cin, cout, pool):
    return nn.Sequential(
        nn.Conv1d(cin, cout, 5, padding=2, bias=False), nn.BatchNorm1d(cout), nn.LeakyReLU(0.2),
        nn.MaxPool1d(pool),
    )


class SincLSTM(nn.Module):
    """Input (B, S) waveform at SAMPLE_RATE -> frame logits (B, S // FRAME_HOP)."""

    def __init__(self, n_filters=32, kernel_size=251, channels=64, hidden=64, lstm_layers=2, dropout=0.2):
        super().__init__()
        assert FRAME_HOP == 5 * 5 * 2
        self.sinc = SincConv(n_filters, kernel_size)
        self.norm0 = nn.BatchNorm1d(n_filters)
        self.conv = nn.Sequential(_conv_block(n_filters, channels, 5), _conv_block(channels, channels, 2))
        self.lstm = nn.LSTM(channels, hidden, num_layers=lstm_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.frame_head = nn.Linear(2 * hidden, 1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = wav - wav.mean(-1, keepdim=True)
        x = x / (x.std(-1, keepdim=True) + 1e-6)          # gain invariance
        h = self.sinc(x.unsqueeze(1)).abs()                # band envelopes
        h = F.avg_pool1d(h, 5)                             # 250 -> 50 Hz
        h = self.norm0(torch.log1p(10 * h))                # log compression
        h = self.conv(h)                                   # 50 -> 5 frames/s
        h, _ = self.lstm(h.transpose(1, 2))                # (B, T, 2*hidden)
        return self.frame_head(self.drop(h)).squeeze(-1)   # (B, T)

    @staticmethod
    def clip_logit(frame_logits: torch.Tensor) -> torch.Tensor:
        return frame_logits.amax(-1)
