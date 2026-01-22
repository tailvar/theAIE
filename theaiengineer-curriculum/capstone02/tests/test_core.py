"""Minimal test for packaged compute_values"""
import pytest
from mymltool.core import compute_values

def test_determinism() -> None:
    seed, n = 42, 5 
    _, mu1, s1 = compute_values(seed, n)
    _, mu2, s2 = compute_values(seed, n)
    assert mu1 == mu2 and s1 == s2

def test_negative_n_raises_error() -> None:
    # n < 0 should not be allowed
    with pytest.raises(ValueError):
        compute_values(seed=42, n=-1)