"""
Core functionality for the Ch2 package example

This keeps logic separate from the CLI so that it can be imported and tested
"""
from __future__ import annotations

import random # determinstic RNG with seed
import statistics as stats  # mean and stdev utilities

def compute_values(seed: int, n: int) -> tuple[list[float], float, float]:
    """Generate ``n`` pseudo-random floats deterministically and summarize.

    Args:
        seed: RNG seed for reproducibility.
        n: number of values to generate (> 0).

    Returns:
        A tuple of (values, mean, stdev).
    """
    if n <= 0:
        raise ValueError("n must be >0")
    rng = random.Random(seed)  # create an isolated RNG
    values = [rng.random() for _ in range(n)]  # n floats in [0, 1)
    mu = stats.fmean(values)  # compute mean (float)
    # Use population stdev for a tiny demo; either is fine when stated.
    sigma = stats.pstdev(values)  # population standard deviation
    return values, mu, sigma  # return both raw values and summary