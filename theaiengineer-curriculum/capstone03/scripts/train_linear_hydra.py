"""Hydra variant of tiny linear regression training script.

Defines a config schema and supports command line overrides via Hydra
"""

from __future__ import annotations  # postpone annotation evaluation

from dataclasses import dataclass # structured config

import numpy as np  # numeric arrays
from hydra import main # Hydra entry
from omegaconf import DictConfig # config type


def make_data(seed: int, n: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic (x, y) with y = 3x + noise."""
    rng = np.random.default_rng(seed)  # RNG with reproducible seed
    x = rng.uniform(-1.0, 1.0, size=(n, 1))  # inputs in [-1, 1]
    noise = rng.normal(0.0, 0.2, size=(n, 1))  # Gaussian noise
    y = 3.0 * x + noise  # linear target with noise
    return x.astype(np.float32), y.astype(np.float32)  # cast to float32


def train(seed: int, epochs: int, lr: float) -> float:
    """Fit y ≈ w*x with gradient descent; return final MSE loss."""
    x, y = make_data(seed)  # synthetic dataset
    w = np.array([[0.0]], dtype=np.float32)  # model parameter (1x1)
    for _ in range(epochs):  # simple training loop
        y_hat = x @ w  # prediction
        err = y_hat - y  # residuals
        grad = (2.0 / x.shape[0]) * (x.T @ err)  # d/dw MSE gradient
        w -= lr * grad  # gradient descent step
    mse = float(np.mean((x @ w - y) ** 2))  # final mean squared error
    return mse  # scalar loss

@dataclass
class TrainCfg:
    seed: int = 7 # RNG seed
    epochs: int = 30 # training epochs
    lr: float = 0.1 # learning rate

@main(config_name=None, version_base=None)
def run(cfg: DictConfig) -> None:
    # Convert nested config into dataclass for clarity if needed
    c = TrainCfg(**cfg) # type: ignore[arg-type]
    final_loss = train(c.seed, c.epochs, c.lr)
    print(f"final_loss={final_loss:.6f}")  # concise console metric


if __name__ == "__main__":  # only run when executed as a script
    run()  # Hydra handles CLI overrides and working dir