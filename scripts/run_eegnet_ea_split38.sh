#!/bin/bash
# Run EEGNet + Euclidean Alignment baseline on XWStroke Split38.
# 1) Pretrain on 38 source subjects.
# 2) 0/5/10/20-shot LOSO evaluation on 12 held-out subjects.

set -e

cd /home/user_lyh/python_project/EEGBCI-SEF

source /home/opt/anaconda/etc/profile.d/conda.sh
conda activate torch

PRETRAIN_CFG="configs/experiment/xwstroke_eegnet_split38_pretrain.yaml"
BASE_LOSO_CFG="configs/experiment/xwstroke_eegnet_split38_loso.yaml"
TMP_CFG="/tmp/eegnet_ea_split38_loso.yaml"

echo "============================================================"
echo "Step 1: Pretrain EEGNet+EA on Split38 source subjects"
echo "============================================================"
python train.py --config "$PRETRAIN_CFG"

# Infer the actual pretrain checkpoint path (auto-generated timestamp dir)
CKPT=$(find ./experiments/eegnet_ea_split38 -name "pretrained_best.pt" | head -n 1)
if [ -z "$CKPT" ]; then
    echo "ERROR: Pretrained checkpoint not found."
    exit 1
fi

echo "Pretrained checkpoint: $CKPT"

echo ""
echo "============================================================"
echo "Step 2: N-shot LOSO evaluation"
echo "============================================================"

for n in 0 5 10 20; do
    echo ""
    echo "--- ${n}-shot ---"

    sed -e "s|pretrained_path:.*|pretrained_path: \"$CKPT\"|" \
        -e "s|n_shot:.*|n_shot: $n|" \
        -e "s|name: \"XWStroke_EEGNet_Split38_LOSO_Nshot_EA\"|name: \"XWStroke_EEGNet_Split38_LOSO_${n}shot_EA\"|" \
        "$BASE_LOSO_CFG" > "$TMP_CFG"

    python train.py --config "$TMP_CFG"
done

rm -f "$TMP_CFG"

echo ""
echo "All EEGNet+EA Split38 experiments completed."
