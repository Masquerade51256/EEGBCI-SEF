#!/bin/bash
# Run LOSO for all three CACML ablation checkpoints.

set -e

cd /home/user_lyh/python_project/EEGBCI-SEF

source /home/opt/anaconda/etc/profile.d/conda.sh
conda activate torch

CHECKPOINTS=(
    "experiments/cacml_split38_clinical_informed/XWStroke_CACML_20260618_095759/checkpoints/cacml_pretrained_best.pt"
    "experiments/cacml_split38_class_aware/XWStroke_CACML_20260618_103657/checkpoints/cacml_pretrained_best.pt"
    "experiments/cacml_split38_no_contrastive/XWStroke_CACML_20260618_103943/checkpoints/cacml_pretrained_best.pt"
)

NAMES=(
    "clinical_informed"
    "class_aware"
    "no_contrastive"
)

BASE_CFG="configs/experiment/xwstroke_cacml_split38_clinical_informed_loso.yaml"
TMP_CFG="/tmp/cacml_loso_all.yaml"

for idx in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$idx]}"
    NAME="${NAMES[$idx]}"

    echo "============================================================"
    echo "Running LOSO for: $NAME"
    echo "Checkpoint: $CKPT"
    echo "============================================================"

    for n in 0 5 10 20; do
        echo "--- ${n}-shot ---"
        sed -e "s|pretrained_path:.*|pretrained_path: \"$CKPT\"|" \
            -e "s|n_shot:.*|n_shot: $n|" \
            -e "s|root_dir:.*|root_dir: \"./results/cacml_loso_results\"|" \
            "$BASE_CFG" > "$TMP_CFG"

        # Update experiment name to avoid overwriting
        sed -i "s|name: \"XWStroke_CACML_Split38_ClinicalInformed_LOSO\"|name: \"XWStroke_CACML_${NAME}_${n}shot_LOSO\"|" "$TMP_CFG"

        python train.py --config "$TMP_CFG"
    done
    echo ""
done

rm -f "$TMP_CFG"
echo "All LOSO evaluations completed."
