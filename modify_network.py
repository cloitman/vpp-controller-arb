"""
modify_network.py — scale and perturb r, x, I_max values in a network CSV.

Each non-zero entry in r, x, and I_max is multiplied by the chosen scale factor
and then perturbed with independent multiplicative Gaussian noise so that values
across rows are no longer identical.

Usage examples
--------------
# Halve all resistances, add 15% node-to-node variation, keep I_max unchanged:
python modify_network.py \
    --input  data/networks/homework3bus_no_gen.csv \
    --output data/networks/homework3bus_modified.csv \
    --r-scale 0.5 \
    --noise 0.15

# Scale everything down, no noise:
python modify_network.py \
    --input  data/networks/homework3bus_no_gen.csv \
    --output data/networks/homework3bus_modified.csv \
    --r-scale 0.4 --x-scale 0.4 --imax-scale 0.8
"""

import argparse
import ast
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_vec(s: str) -> np.ndarray:
    return np.array(ast.literal_eval(s), dtype=float)


def _format_vec(arr: np.ndarray) -> str:
    parts = ["0" if v == 0.0 else f"{v:.6g}" for v in arr]
    return "[" + ", ".join(parts) + "]"


def _perturb(arr: np.ndarray, scale: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    """Scale non-zero entries then add multiplicative Gaussian noise."""
    result = arr.copy()
    mask = arr != 0.0
    n = int(mask.sum())
    if n == 0:
        return result
    result[mask] = arr[mask] * scale
    if noise > 0.0:
        factors = 1.0 + noise * rng.standard_normal(n)
        # Clip so values never go negative or flip to zero
        factors = np.maximum(factors, 0.01)
        result[mask] *= factors
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scale and perturb r, x, I_max in a network CSV."
    )
    parser.add_argument("--input", required=True, help="Path to input network CSV")
    parser.add_argument("--output", required=True, help="Path for output CSV")
    parser.add_argument(
        "--r-scale", type=float, default=1.0,
        help="Multiplier applied to every non-zero r entry (default 1.0 = no change)",
    )
    parser.add_argument(
        "--x-scale", type=float, default=1.0,
        help="Multiplier applied to every non-zero x entry (default 1.0 = no change)",
    )
    parser.add_argument(
        "--imax-scale", type=float, default=1.0,
        help="Multiplier applied to every non-zero I_max entry (default 1.0 = no change)",
    )
    parser.add_argument(
        "--noise", type=float, default=0.0,
        help=(
            "Fractional standard deviation of multiplicative noise applied to each "
            "non-zero entry independently. E.g. 0.10 gives roughly ±10%% variation "
            "across nodes for the same branch type. Applied after scaling."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default 42)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        sys.exit(f"ERROR: input file not found: {args.input}")

    column_scales = [
        ("r",     args.r_scale),
        ("x",     args.x_scale),
        ("I_max", args.imax_scale),
    ]

    for col, scale in column_scales:
        if col not in df.columns:
            print(f"WARNING: column '{col}' not found in CSV, skipping.")
            continue
        df[col] = [
            _format_vec(_perturb(_parse_vec(val), scale, args.noise, rng))
            for val in df[col]
        ]

    df.to_csv(args.output, index=False)

    print(f"Saved to: {args.output}")
    print(f"  r-scale={args.r_scale}  x-scale={args.x_scale}  "
          f"imax-scale={args.imax_scale}  noise={args.noise}  seed={args.seed}")


if __name__ == "__main__":
    main()
