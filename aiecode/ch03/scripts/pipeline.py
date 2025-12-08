
"""A tiny CSV → features → model pipeline with artifacts and a run report.

Steps:
  1) ingest_csv: load (x, y) from a CSV file.
  2) make_features: build a feature matrix from x (add bias column).
  3) train_linear: fit y ≈ X @ theta via normal equation.
Outputs:
  - features.npy, model.npz, report.json under an output directory.
"""

from __future__ import annotations  # future-friendly typing

import argparse  # parse command-line flags
import csv  # read input CSV
import json  # write a small run report
from dataclasses import asdict, dataclass  # structured report
from pathlib import Path  # filesystem paths
from time import perf_counter  # simple timings
from typing import Tuple  # type hints

import numpy as np  # numeric arrays


@dataclass
class Report:
    seed: int  # RNG seed for reproducibility
    n_rows: int  # number of data rows
    mse: float  # mean squared error on the training data
    features_path: str  # saved features file
    model_path: str  # saved model file


def ingest_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load columns x, y from a CSV with a header."""
    rows = []  # hold numeric rows
    with path.open("r", encoding="utf-8") as f:  # open input file
        reader = csv.DictReader(f)  # read named columns
        for row in reader:  # each row is a dict
            rows.append((float(row["x"]), float(row["y"])) )  # parse to float
    if not rows:  # empty input is invalid
        raise SystemExit("ingest_csv: no data rows found")  # exit early
    arr = np.array(rows, dtype=np.float32)  # to array
    x = arr[:, :1]  # shape (n, 1)
    y = arr[:, 1:2]  # shape (n, 1)
    return x, y  # inputs and targets


def make_features(x: np.ndarray) -> np.ndarray:
    """Build a feature matrix: bias and x (shape (n, 2))."""
    bias = np.ones_like(x)  # column of ones
    X = np.concatenate([bias, x], axis=1)  # [1, x]
    return X  # shape (n, 2)


def train_linear(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit y ≈ X @ theta via normal equation; return (theta, mse)."""
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y  # closed-form solution
    mse = float(np.mean((X @ theta - y) ** 2))  # mean squared error
    return theta, mse  # model and metric


def build_parser() -> argparse.ArgumentParser:
    """CLI parser for the pipeline."""
    p = argparse.ArgumentParser(
        description="Run a tiny CSV → features → model pipeline."
    )
    p.add_argument(
        "--data", type=Path, required=True, help="Path to CSV with columns x,y."
    )
    p.add_argument(
        "--out", type=Path, required=True, help="Output directory for artifacts."
    )
    p.add_argument(
        "--seed", type=int, default=7, help="Seed for any randomized steps."
    )
    return p  # return configured parser

def main() -> None:
    """Run the pipeline end to end and save artifacts + report."""
    args = build_parser().parse_args()  # parse flags
    args.out.mkdir(parents=True, exist_ok=True)  # ensure output dir exists

    t0 = perf_counter()  # start timing
    x, y = ingest_csv(args.data)  # load data
    X = make_features(x)  # feature matrix
    theta, mse = train_linear(X, y)  # fit model
    dt = perf_counter() - t0  # elapsed seconds

    # Save artifacts
    features_path = args.out / "features.npy"  # features file path
    np.save(features_path, X)  # write features
    model_path = args.out / "model.npz"  # model file path
    np.savez(model_path, theta=theta)  # write model parameters

    # Save a compact JSON report with key run info
    report = Report(
        seed=args.seed, n_rows=int(X.shape[0]), mse=mse,
        features_path=str(features_path), model_path=str(model_path)
    )
    with (args.out / "report.json").open("w", encoding="utf-8") as f:  # write JSON
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)  # pretty JSON

    # Echo a concise summary for humans
    msg = f"rows={report.n_rows} mse={report.mse:.6f} saved={args.out}"
    print(msg)  # user output


if __name__ == "__main__":  # only run when executed as script
    main()  # invoke entry