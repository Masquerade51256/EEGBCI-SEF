#!/bin/bash
# Batch runner: ShallowConvNet 4-pipeline LOSO (architecture robustness check,
# response to Reviewer vf3F #4 / m6Dh #4).
# Waits for the multi-K source-selection batch (PID given as $1) to finish
# before starting, to avoid GPU contention.
PY=/home/opt/anaconda/envs/torch/bin/python
cd /home/user_lyh/python_project/EEGBCI-SEF

WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
  echo "[$(date '+%F %T')] Waiting for multi-K batch (pid $WAIT_PID) to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[$(date '+%F %T')] Multi-K batch done. Starting ShallowConvNet runs."
fi

CONFIGS=(
  configs/experiment/xwstroke_shallowconvnet_full_loso_lr_noEA.yaml
  configs/experiment/xwstroke_shallowconvnet_full_loso_lr_ea.yaml
  configs/experiment/xwstroke_shallowconvnet_full_loso_affalign_noEA.yaml
  configs/experiment/xwstroke_shallowconvnet_full_loso_ea_affalign.yaml
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
