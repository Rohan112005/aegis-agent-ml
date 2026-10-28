"""Autoencoder model and per-sample reconstruction loss."""

import torch
import torch.nn as nn


class PacketAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

        # CICIDS training uses MinMaxScaler, so training targets are in [0, 1].
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def compute_reconstruction_loss(
    model: nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    outputs = model(inputs)
    return torch.mean((inputs - outputs) ** 2, dim=1)
