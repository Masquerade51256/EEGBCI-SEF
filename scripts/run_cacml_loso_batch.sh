#!/bin/bash
# Batch LOSO evaluation for a given pretrained checkpoint.
# Usage: bash scripts/run_cacml_loso_batch.sh <pretrained_checkpoint_path>
# Example:
#   bash scripts/run_cacml_loso_batch.sh \
#     experiments/cacml_split38_clinical_informed/XWStroke_CACML_20260618_095759/checkpoints/cacml_pretrained_best.pt

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <pretrained_checkpoint_path>"
    exit 1
fi

CKPT="$1"

if [ ! -f "$CKPT" ]; then
    echo "Checkpoint not found: $CKPT"
    exit 1
fi

cd /home/user_lyh/python_project/EEGBCI-SEF

# Activate conda environment
source /home/opt/anaconda/etc/profile.d/conda.sh
conda activate torch

BASE_CFG="configs/experiment/xwstroke_cacml_split38_clinical_informed_loso.yaml"
TMP_CFG="/tmp/cacml_loso_batch.yaml"

SHOTS=(0 5 10 20)

for n in "${SHOTS[@]}"; do
    echo "============================================================"
    echo "Running LOSO ${n}-shot"
    echo "============================================================"

    # Create temporary config with updated pretrained_path and n_shot
    sed -e "s|pretrained_path:.*|pretrained_path: \"$CKPT\"|" \
        -e "s|n_shot:.*|n_shot: $n|" \
        "$BASE_CFG" > "$TMP_CFG"

    python train.py --config "$TMP_CFG"
    echo ""
done

rm -f "$TMP_CFG"
echo "All LOSO evaluations completed."
