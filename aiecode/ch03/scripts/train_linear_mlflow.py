"""MLflow variant of the tiny linear regression training script.

Logs parameters and final loss to an MLflow tracking server.
"""

from __future__ import annotations

import argparse  # parse command-line flags
from typing import Tuple

import numpy as np  # numeric arrays

import mlflow  # type: ignore[import]  # external, optional dependency
# Point to your tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# Make sure we use (or create) this experiment
mlflow.set_experiment("experiment no.1")


def make_data(seed: int, n: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)  # reproducible RNG
    x = rng.uniform(-1.0, 1.0, size=(n, 1))  # inputs in [-1, 1]
    noise = rng.normal(0.0, 0.2, size=(n, 1))  # Gaussian noise
    y = 3.0 * x + noise  # linear target with noise
    return x.astype(np.float32), y.astype(np.float32)


def train(seed: int, epochs: int, lr: float) -> float:
    x, y = make_data(seed)
    w = np.array([[0.0]], dtype=np.float32)  # parameter
    for _ in range(epochs):
        y_hat = x @ w
        err = y_hat - y
        grad = (2.0 / x.shape[0]) * (x.T @ err)
        w -= lr * grad
    mse = float(np.mean((x @ w - y) ** 2))
    return mse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train linear model and log to MLflow.")
    p.add_argument("--seed", type=int, default=7, help="RNG seed (int).")
    p.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    p.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    with mlflow.start_run():  # start a run context
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Active Experiment: {mlflow.get_experiment_by_name('my-first-experiment')}")
        mlflow.log_param("seed", args.seed)  # log parameters
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("lr", args.lr)
        final_loss = train(args.seed, args.epochs, args.lr)
        mlflow.log_metric("final_loss", final_loss)  # log metric
        print(f"final_loss={final_loss:.6f}")


if __name__ == "__main__":
    main()