"""CLI that calls the packaged core function

"""
import argparse  # parse command-line flags
from mymltool.core import compute_values

def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Compute a reproducible mean and stdev."
    )
    p.add_argument("--seed", type=int, default=123, help="RNG seed (int).")
    p.add_argument("--n", type=int, default=5, help="Number of values to generate (>0).")
    p.add_argument("--verbose",action="store_true",help="print configuration details before running")
    return p  # return configured parser


def main() -> None:
    """Entry point for the CLI."""
    args = build_parser().parse_args()  # parse flags from sys.argv
    if args.verbose:
        print(f"seed={args.seed} n={args.n}")
            
    _, mu, sigma = compute_values(args.seed, args.n)  # run computation
    print(f"mean={mu:.6f} stdev={sigma:.6f}")  # user-facing output


if __name__ == "__main__":  # only run when executed as a script
    main()  # invoke the CLI entry