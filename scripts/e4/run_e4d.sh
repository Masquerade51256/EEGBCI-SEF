#!/bin/bash
# E4 第 1.5 步门控修复：gate_fix 单条件，XW 50 被试，并发上限 6
cd /home/user_lyh/python_project/EEGBCI-SEF || exit 1
git pull origin main -q
P=/home/opt/anaconda/envs/torch/bin/python
mkdir -p logs_e4d
MAXJ=6
for s in $(seq 1 50); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
  "$P" scripts/e4/e4d_gate_fix.py --dataset xw --subjects "$s" \
     > "logs_e4d/xw_gate_fix_s${s}.log" 2>&1 &
done
wait
echo ALL_DONE_E4D
