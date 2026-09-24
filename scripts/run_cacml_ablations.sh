#!/bin/bash
# Run CACML ablation pretraining experiments.
# Usage: bash scripts/run_cacml_ablations.sh

set -e

cd /home/user_lyh/python_project/EEGBCI-SEF

# Activate conda environment
source /home/opt/anaconda/etc/profile.d/conda.sh
conda activate torch

CONFIGS=(
    "configs/experiment/xwstroke_cacml_split38_class_aware_pretrain.yaml"
    "configs/experiment/xwstroke_cacml_split38_no_contrastive_pretrain.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    echo "============================================================"
    echo "Running pretraining: $cfg"
    echo "============================================================"
    python train.py --config "$cfg"
    echo ""
done

echo "All pretraining experiments completed."
