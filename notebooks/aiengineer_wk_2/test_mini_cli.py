from mini_cli import compute_values

def test_compute_values():
    """For a fixed seed and n, compute_values should be reproducible"""
    seed = 42
    n = 5

    # call the function twice with identical inputs
    values1, mu1, sigma1 = compute_values(seed, n)
    values2, mu2, sigma2 = compute_values(seed, n)

    # check mean and standard deviation match exactly
    assert mu1 == mu2
    assert sigma1 == sigma2

    # optional: check raw values match
    assert values1 == values2