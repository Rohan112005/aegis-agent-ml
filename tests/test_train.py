
"""Unit tests for the training pipeline (synthetic data + short train run)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_engine.autoencoder import (
    PacketAutoencoder,
    compute_reconstruction_loss,
)
from src.ml_engine.train import (
    generate_synthetic_normal_traffic,
    train,
)


def test_generate_synthetic_shape_and_range() -> None:
    """Check the shape, data type, and range of generated traffic."""
    data = generate_synthetic_normal_traffic(
        n_samples=100,
        input_dim=10,
        seed=1,
    )

    assert data.shape == (100, 10)
    assert data.dtype == np.float32
    assert data.min() >= 0.0
    assert data.max() <= 1.0


def test_generate_synthetic_reproducible() -> None:
    """Check that the same seed generates identical data."""
    data_a = generate_synthetic_normal_traffic(
        n_samples=50,
        input_dim=10,
        seed=99,
    )

    data_b = generate_synthetic_normal_traffic(
        n_samples=50,
        input_dim=10,
        seed=99,
    )

    np.testing.assert_array_equal(data_a, data_b)


def test_short_training_saves_checkpoint(tmp_path: Path) -> None:
    """Run a short training loop and verify that the checkpoint is saved."""
    model_path = tmp_path / "autoencoder.pth"

    model = train(
        input_dim=10,
        epochs=2,
        batch_size=32,
        n_samples=128,
        model_path=model_path,
        seed=7,
    )

    # Check that the model was saved.
    assert model_path.exists()

    # Check that the returned model has the expected type.
    assert isinstance(model, PacketAutoencoder)

    # Load the saved checkpoint.
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    assert "model_state_dict" in checkpoint
    assert checkpoint["input_dim"] == 10
    assert "final_mse" in checkpoint

    # Load the saved weights into a fresh model.
    fresh_model = PacketAutoencoder(input_dim=10)

    fresh_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    fresh_model.eval()

    # Test reconstruction loss using the reloaded model.
    sample_data = torch.rand(2, 10)

    with torch.no_grad():
        loss = compute_reconstruction_loss(
            fresh_model,
            sample_data,
        )

    assert loss.shape == (2,)
