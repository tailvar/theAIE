"""A tiny CLI that computes a reproducible mean and stdev.

Run:
  python code/capstone01/mini_cli.py --seed 42 --n 5
"""

from __future__ import annotations  # postpone annotations

import argparse  # parse command-line flags
import random  # deterministic RNG with a seed
import statistics as stats  # mean and stdev utilities
from mymltool.core import compute_values


# def compute_values(seed: int, n: int) -> tuple[list[float], float, float]:
#     """Generate n pseudo-random floats deterministically and summarize.

#     Args:
#         seed: RNG seed for reproducibility.
#         n: number of values to generate (> 0).

#     Returns:
#         A tuple of (values, mean, stdev).
#     """
#     rng = random.Random(seed)  # create an isolated RNG
#     values = [rng.random() for _ in range(n)]  # n floats in [0, 1)
#     mu = stats.fmean(values)  # compute mean (float)
#     # Use population stdev for a tiny demo; either is fine when stated.
#     sigma = stats.pstdev(values)  # population standard deviation
#     return values, mu, sigma  # return both raw values and summary


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Compute a reproducible mean and stdev."
    )
    p.add_argument(
        "--seed", type=int, default=123, help="RNG seed (int)."
    )
    p.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of values to generate (>0).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="print configuration details before running"
    )
    return p  # return configured parser


def main() -> None:
    """Entry point for the CLI."""
    args = build_parser().parse_args()  # parse flags from sys.argv

    if args.verbose:
        print(f"[config] seed = {args.seed} n = {args.n}")
    
    if args.n <= 0:  # validate the input
        raise SystemExit(f"--n must be > 0; was given {args.n}")  # exit with a clear error

    if args.seed < 0:
        raise SystemExit(f"--seed must be >=0; got {args.seed}")

        
    values, mu, sigma = compute_values(args.seed, args.n)  # run computation
    # Print a concise, verifiable summary (copy-pastable into tests if needed).
    print(
        f"seed={args.seed} n={args.n} mean={mu:.6f} stdev={sigma:.6f}"
    )  # user-facing output


if __name__ == "__main__":  # only run when executed as a script
    main()  