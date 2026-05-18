#!/usr/bin/env python3
"""
Run DANN domain-adversarial LOSO experiments.

Usage:
    # Full 50-subject DANN LOSO (long running)
    python scripts/run_dann_experiment.py --config configs/experiment/xwstroke_dann_loso.yaml

    # Subset 15 quick validation
    python scripts/run_dann_experiment.py --config configs/experiment/xwstroke_dann_subset15.yaml

    # Override domain grouping strategy
    python scripts/run_dann_experiment.py --config configs/experiment/xwstroke_dann_loso.yaml \
        --domain-grouping stroke_location_2class

    # Grid search over DANN hyperparameters
    python scripts/run_dann_experiment.py --config configs/experiment/xwstroke_dann_loso.yaml \
        --grid-search
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.run_experiment import main as run_experiment_main


def main():
    parser = argparse.ArgumentParser(description="Run DANN LOSO experiments")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--domain-grouping",
        type=str,
        default=None,
        choices=[
            "stroke_location_4class",
            "stroke_location_2class",
            "paralysis_side",
            "nihss_3class",
        ],
        help="Override domain grouping strategy",
    )
    parser.add_argument(
        "--grl-gamma",
        type=float,
        default=None,
        help="Override GRL gamma parameter",
    )
    parser.add_argument(
        "--domain-loss-weight",
        type=float,
        default=None,
        help="Override domain loss weight",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run grid search over DANN hyperparameters",
    )

    args = parser.parse_args()

    if args.grid_search:
        run_grid_search(args.config)
    else:
        run_single(args)


def run_single(args):
    """Run a single DANN experiment."""
    # Build command-line overrides for run_experiment_main
    overrides = []

    if args.domain_grouping is not None:
        overrides.extend(["--set", f"trainer.args.domain_grouping={args.domain_grouping}"])

    if args.grl_gamma is not None:
        overrides.extend(["--set", f"trainer.args.grl_gamma={args.grl_gamma}"])

    if args.domain_loss_weight is not None:
        overrides.extend(["--set", f"trainer.args.domain_loss_weight={args.domain_loss_weight}"])

    # Forward to standard experiment runner
    sys.argv = [sys.argv[0], "--config", args.config] + overrides
    run_experiment_main()


def run_grid_search(config_path: str):
    """
    Grid search over DANN hyperparameters.

    Searches:
        - domain_grouping: stroke_location_4class, stroke_location_2class
        - grl_gamma: 5.0, 10.0, 15.0
        - domain_loss_weight: 0.5, 1.0, 2.0
    """
    import subprocess

    domain_groupings = ["stroke_location_4class", "stroke_location_2class"]
    grl_gammas = [5.0, 10.0, 15.0]
    domain_loss_weights = [0.5, 1.0, 2.0]

    total = len(domain_groupings) * len(grl_gammas) * len(domain_loss_weights)
    print(f"Starting DANN grid search: {total} configurations")
    print("=" * 60)

    for dg in domain_groupings:
        for gamma in grl_gammas:
            for weight in domain_loss_weights:
                exp_name = f"DANN_grid_{dg}_g{gamma}_w{weight}"
                print(f"\n>>> Running: {exp_name}")

                cmd = [
                    sys.executable,
                    "scripts/run_dann_experiment.py",
                    "--config",
                    config_path,
                    "--domain-grouping",
                    dg,
                    "--grl-gamma",
                    str(gamma),
                    "--domain-loss-weight",
                    str(weight),
                ]

                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error running {exp_name}: {e}")
                    continue

    print("\n" + "=" * 60)
    print("Grid search complete!")


if __name__ == "__main__":
    main()
