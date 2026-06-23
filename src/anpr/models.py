from __future__ import annotations

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, *, batch_norm: bool = True) -> None:
        layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class SmallCRNN(nn.Module):
    """Stable lightweight CRNN.

    This is the safer baseline for small datasets. It borrows the idea of a
    simpler CRNN variant from notebook-style examples, but keeps the code reusable.
    """

    def __init__(self, vocab_size: int, in_channels: int = 1, hidden_size: int = 128, dropout: float = 0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(in_channels, 64, batch_norm=False),
            nn.MaxPool2d(2, 2),              # 32x128 -> 16x64
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),              # 16x64 -> 8x32
            ConvBlock(128, 256),
            nn.MaxPool2d(kernel_size=(2, 1)), # 8x32 -> 4x32
            ConvBlock(256, 256),
        )
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)               # [B,C,H,W]
        features = features.mean(dim=2)      # [B,C,W], collapse height robustly
        features = features.permute(0, 2, 1) # [B,W,C]
        seq, _ = self.rnn(features)
        logits = self.classifier(seq)
        return logits.log_softmax(2).permute(1, 0, 2)  # [T,B,C]


class DeepCRNN(nn.Module):
    """Heavier CRNN similar to classical OCR CRNN: CNN + 2 BiLSTM + CTC."""

    def __init__(self, vocab_size: int, in_channels: int = 1, hidden_size: int = 256, dropout: float = 0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(in_channels, 64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1)),
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        features = features.mean(dim=2)
        features = features.permute(0, 2, 1)
        seq, _ = self.rnn(features)
        logits = self.classifier(seq)
        return logits.log_softmax(2).permute(1, 0, 2)


def build_crnn(architecture: str, vocab_size: int, in_channels: int = 1, hidden_size: int | None = None) -> nn.Module:
    architecture = architecture.lower()
    if architecture in {"small", "simple", "small_crnn", "simple_crnn"}:
        return SmallCRNN(vocab_size=vocab_size, in_channels=in_channels, hidden_size=hidden_size or 128)
    if architecture in {"deep", "crnn", "deep_crnn"}:
        return DeepCRNN(vocab_size=vocab_size, in_channels=in_channels, hidden_size=hidden_size or 256)
    raise ValueError(f"Unknown CRNN architecture: {architecture}")
