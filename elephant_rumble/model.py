"""SincNet front-end + bidirectional LSTM for elephant rumble detection.

Raw waveform (1 kHz) --> SincConv (learnable band-pass filters, initialised on
a log scale over 5-250 Hz, so most filters sit in the infrasound/rumble band)
--> |.| --> pooling + log compression --> two 1-D conv blocks (down to
20 frames/s) --> 2-layer BiLSTM that follows the temporal shape of the call
(rumbles last ~1-6 s) --> per-frame rumble logit.

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

from .config import FRAME_HOP, SAMPLE_RATE


class SincConv(nn.Module):
    """1-D convolution whose kernels are parametrised band-pass sinc filters.

    Only two numbers per filter are learned: the low cut-off and the bandwidth.
    """

    def __init__(self, n_filters=32, kernel_size=251, sample_rate=SAMPLE_RATE,
                 f_low=5.0, f_high=250.0, min_low_hz=2.0, min_band_hz=3.0):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size, self.sample_rate = kernel_size, sample_rate
        self.min_low_hz, self.min_band_hz = min_low_hz, min_band_hz

        edges = np.geomspace(f_low, f_high, n_filters + 1)
        self.low_hz_ = nn.Parameter(torch.tensor(edges[:-1] - min_low_hz, dtype=torch.float32).view(-1, 1))
        band = np.maximum(np.diff(edges) - min_band_hz, 0.5)
        self.band_hz_ = nn.Parameter(torch.tensor(band, dtype=torch.float32).view(-1, 1))

        half = kernel_size // 2
        n_lin = torch.linspace(0, half - 1, steps=half)
        self.register_buffer("window_", 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size))
        self.register_buffer("n_", 2 * math.pi * torch.arange(-half, 0).view(1, -1) / sample_rate)

    def cutoffs(self):
        low = self.min_low_hz + self.low_hz_.abs()
        high = torch.clamp(low + self.min_band_hz + self.band_hz_.abs(), self.min_low_hz, self.sample_rate / 2)
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
        h = F.avg_pool1d(h, 5)                             # 1000 -> 200 Hz
        h = self.norm0(torch.log1p(10 * h))                # log compression
        h = self.conv(h)                                   # 200 -> 20 frames/s
        h, _ = self.lstm(h.transpose(1, 2))                # (B, T, 2*hidden)
        return self.frame_head(self.drop(h)).squeeze(-1)   # (B, T)

    @staticmethod
    def clip_logit(frame_logits: torch.Tensor) -> torch.Tensor:
        return frame_logits.amax(-1)
