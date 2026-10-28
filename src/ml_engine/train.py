"""
Training script for the PacketAutoencoder.

Teaches the model what "normal" network traffic looks like by minimising
reconstruction error on synthetic (or real) feature vectors, then saves
the weights to models/autoencoder.pth.

Usage (from project root):
    python -m src.ml_engine.train
    python -m src.ml_engine.train --epochs 30 --samples 8000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.ml_engine.autoencoder import PacketAutoencoder, compute_reconstruction_loss


def generate_synthetic_normal_traffic(
    n_samples: int = 5000, input_dim: int = 10, seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic 'normal' traffic features in [0, 1].

    In a real deployment replace this with features extracted from
    Zeek/CICFlowMeter/pcap pipelines (packet sizes, inter-arrival times,
    protocol flags, etc.).
    """
    rng = np.random.default_rng(seed)
    # Slightly correlated features so the autoencoder has structure to learn
    half = input_dim // 2
    base = rng.normal(loc=0.5, scale=0.15, size=(n_samples, half))
    noise = rng.normal(loc=0.0, scale=0.05, size=(n_samples, half))
    features = np.hstack([base, base + noise])
    if input_dim % 2 == 1:
        extra = rng.normal(loc=0.5, scale=0.1, size=(n_samples, 1))
        features = np.hstack([features, extra])
    return np.clip(features, 0.0, 1.0).astype(np.float32)


def train(
    input_dim: int = 10,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    n_samples: int = 5000,
    model_path: str | Path = "models/autoencoder.pth",
    seed: int = 42,
) -> PacketAutoencoder:
    """
    Train PacketAutoencoder on normal traffic and persist weights.

    Returns the trained model (already on CPU after save).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    print(f"[train] Samples={n_samples}, input_dim={input_dim}, epochs={epochs}")

    data = generate_synthetic_normal_traffic(
        n_samples=n_samples, input_dim=input_dim, seed=seed
    )
    tensor_data = torch.from_numpy(data)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PacketAutoencoder(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    history: list[float] = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(dataset)
        history.append(avg_loss)
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch {epoch:03d}/{epochs} | MSE Loss: {avg_loss:.6f}")

    # Quick sanity check: mean reconstruction loss on training set should be low
    model.eval()
    with torch.no_grad():
        sample = tensor_data[:256].to(device)
        recon_losses = compute_reconstruction_loss(model, sample)
        mean_recon = recon_losses.mean().item()
    print(f"[train] Mean reconstruction loss (sample): {mean_recon:.6f}")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "final_mse": history[-1] if history else None,
            "mean_recon_loss": mean_recon,
        },
        model_path,
    )
    print(f"[train] Model saved → {model_path.resolve()}")

    return model.cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AegisAgent PacketAutoencoder")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=10)
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/autoencoder.pth",
        help="Where to write the trained weights",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    import sys

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    args = parse_args()
    train(
        input_dim=args.input_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_samples=args.samples,
        model_path=args.model_path,
        seed=args.seed,
    )
