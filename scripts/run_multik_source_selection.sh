#!/bin/bash
# Batch runner: multi-K clinical source-selection LOSO (EEGNet, LR labels).
# Response to Reviewer m6Dh #3. Runs sequentially; each run ~13 min.
PY=/home/opt/anaconda/envs/torch/bin/python
cd /home/user_lyh/python_project/EEGBCI-SEF

CONFIGS=(
  configs/experiment/xwstroke_eegnet_full_loso_lr_noEA_source_sel_k5.yaml
  configs/experiment/xwstroke_eegnet_full_loso_lr_noEA_source_sel_k10.yaml
  configs/experiment/xwstroke_eegnet_full_loso_lr_noEA_source_sel_k15.yaml
  configs/experiment/xwstroke_eegnet_full_loso_lr_noEA_source_sel_k20.yaml
  configs/experiment/xwstroke_eegnet_full_loso_lr_noEA_source_sel_k25.yaml
  configs/experiment/xwstroke_eegnet_full_loso_lr_ea_source_sel_k5.yaml
  configs/experiment/xwstroke_eegnet_full_loso_lr_ea_source_sel_k10.yaml
  configs/experiment/xwstroke_eegnet_full_loso_lr_ea_source_sel_k15.yaml
  configs/experiment/xwstroke_eegnet_full_loso_lr_ea_source_sel_k20.yaml
  configs/experiment/xwstroke_eegnet_full_loso_lr_ea_source_sel_k25.yaml
)

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  log="experiments/${name}.log"
  echo "[$(date '+%F %T')] START $cfg"
  "$PY" train.py --config "$cfg" > "$log" 2>&1
  rc=$?
  # Record the experiment output dir for later aggregation
  exp_dir=$(grep -m1 "Output directory:" "$log" | awk '{print $NF}')
  echo "[$(date '+%F %T')] DONE $cfg exit=$rc dir=$exp_dir"
  if [ $rc -ne 0 ]; then
    echo "ERROR: $cfg failed with exit $rc; aborting batch."
    exit $rc
  fi
done
echo "[$(date '+%F %T')] ALL RUNS COMPLETE"
