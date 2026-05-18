"""
DANN Diagnostic Grid Search on Subset-20.

Runs multiple DANN configurations sequentially and saves histories for comparison.
Use this to systematically diagnose why DANN failed and find a working configuration.

Usage:
    python scripts/run_dann_diagnostic_grid.py

The script will run the following grid on Subset-20:
    - domain_grouping: [ParalysisSide, stroke_location_2class, subject_id]
    - domain_loss_weight: [0.01, 0.05, 0.1, 0.3]
    - Total: 12 configurations × ~20 min each = ~4 hours on GPU

After running, analyze with:
    python scripts/analyze_dann_histories.py --exp_dir experiments/XWStroke_Subset20_DANN_Diagnostic_*
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Grid definition
DOMAIN_GROUPINGS = [
    "ParalysisSide",
    "stroke_location_2class",
    # "subject_id",  # Uncomment if you want to try per-subject domain (20 domains)
]

DOMAIN_LOSS_WEIGHTS = [0.01, 0.05, 0.1, 0.3]

BASE_CONFIG = "configs/experiment/xwstroke_subset20_dann_diagnostic.yaml"


def run_single_config(grouping, weight, config_path):
    """Run one DANN config via train.py with CLI overrides."""
    exp_name = f"XWStroke_Subset20_DANN_{grouping}_w{weight}"
    
    cmd = [
        sys.executable, "train.py",
        "--config", str(config_path),
        "--name", exp_name,
        "--device", "cuda",
    ]
    
    # Inject overrides via environment or temporary config copy
    # train.py may not support nested CLI overrides, so we use a temp config file
    import yaml
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    cfg["experiment"]["name"] = exp_name
    cfg["trainer"]["args"]["domain_grouping"] = grouping
    cfg["trainer"]["args"]["domain_loss_weight"] = weight
    
    temp_config = Path(f"configs/experiment/_temp_dann_{grouping}_w{weight}.yaml")
    with open(temp_config, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    
    cmd = [
        sys.executable, "train.py",
        "--config", str(temp_config),
        "--name", exp_name,
        "--device", "cuda",
    ]
    
    print("\n" + "=" * 70)
    print(f"RUNNING: {exp_name}")
    print(f"  domain_grouping = {grouping}")
    print(f"  domain_loss_weight = {weight}")
    print("=" * 70)
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    # Clean up temp config
    temp_config.unlink(missing_ok=True)
    
    return result.returncode == 0


def main():
    print("DANN Diagnostic Grid Search")
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Total configs: {len(DOMAIN_GROUPINGS)} × {len(DOMAIN_LOSS_WEIGHTS)} = {len(DOMAIN_GROUPINGS) * len(DOMAIN_LOSS_WEIGHTS)}")
    print("Estimated total time: ~4 hours on GPU")
    print("=" * 70)
    
    success_count = 0
    fail_count = 0
    results = []
    
    for grouping in DOMAIN_GROUPINGS:
        for weight in DOMAIN_LOSS_WEIGHTS:
            ok = run_single_config(grouping, weight, BASE_CONFIG)
            results.append((grouping, weight, ok))
            if ok:
                success_count += 1
            else:
                fail_count += 1
    
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE")
    print("=" * 70)
    for grouping, weight, ok in results:
        status = "✓ OK" if ok else "✗ FAIL"
        print(f"  {status}  {grouping:25s}  weight={weight:.2f}")
    print(f"\nSuccess: {success_count}/{len(results)}")
    print(f"Failed:  {fail_count}/{len(results)}")
    print(f"End time: {datetime.now().isoformat()}")
    print("\nNext step: Run analyze_dann_histories.py to compare configurations.")


if __name__ == "__main__":
    main()
