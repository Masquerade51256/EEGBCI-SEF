#!/bin/bash
# Queue runner: ADFCNN noEA-pipeline LOSO reruns with the paper's noaug
# protocol (the historical ADFCNN noEA runs used augmentation=0.3 + FP16;
# the EA runs already match the protocol and are reused as-is).
# Waits for the ShallowConvNet batch (PID given as $1) to finish first.
PY=/home/opt/anaconda/envs/torch/bin/python
cd /home/user_lyh/python_project/EEGBCI-SEF

WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
  echo "[$(date '+%F %T')] Waiting for ShallowConvNet batch (pid $WAIT_PID) to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[$(date '+%F %T')] ShallowConvNet batch done. Starting ADFCNN noEA reruns."
fi

CONFIGS=(
  configs/experiment/xwstroke_adfcnn_full_loso_lr_noEA.yaml
  configs/experiment/xwstroke_adfcnn_full_loso_affalign_noEA.yaml
)

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  log="experiments/${name}.log"
  echo "[$(date '+%F %T')] START $cfg"
  "$PY" train.py --config "$cfg" > "$log" 2>&1
  rc=$?
  exp_dir=$(grep -m1 "Output directory:" "$log" | awk '{print $NF}')
  echo "[$(date '+%F %T')] DONE $cfg exit=$rc dir=$exp_dir"
  if [ $rc -ne 0 ]; then
    echo "ERROR: $cfg failed with exit $rc; aborting batch."
    exit $rc
  fi
done
echo "[$(date '+%F %T')] ALL RUNS COMPLETE"
