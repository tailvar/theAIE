"""Tiny linear regression training with NumPy and JSON run tracking.

Generates synthetic data, fits a 1D linear model via gradient descent, prints
final loss, and writes a JSON record under code/capstone04/runs/ with config + metrics.
"""

from __future__ import annotations  # postpone annotation evaluation

import argparse  # parse command-line flags
from pathlib import Path  # filesystem paths
from typing import Tuple  # simple type alias

import numpy as np  # numeric arrays

from tracking import RunRecord, save_run  # tiny JSON tracker


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


def build_parser() -> argparse.ArgumentParser:
    """Create argument parser for the training script."""
    p = argparse.ArgumentParser(description="Train linear model and log a run.")
    p.add_argument("--seed", type=int, default=7, help="RNG seed (int).")
    p.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    p.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    return p  # return configured parser


def main() -> None:
    """Entry point: train, print final loss, and save a run record."""
    args = build_parser().parse_args()  # parse flags from sys.argv
    final_loss = train(args.seed, args.epochs, args.lr)  # run training
    print(f"final_loss={final_loss:.6f}")  # concise metric for the console
    out = save_run(Path(__file__).with_suffix("").parent / "runs",
                   RunRecord(seed=args.seed, epochs=args.epochs,
                             lr=args.lr, final_loss=final_loss))  # save JSON
    print(f"saved={out}")  # path to the JSON run record


if __name__ == "__main__":  # only run when executed as a script
    main()  # invoke entry point