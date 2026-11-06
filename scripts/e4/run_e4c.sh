#!/bin/bash
# E4 第 1 步双分支门控：并行被试调度器（共享服务器，并发上限 6）
# BCIC 只跑 gate_l01 + uniform（实现效度验证：门控应偏振荡支、融合不低于 EEGNet 水平）
# XW 跑全部 6 臂（主实验）
cd /home/user_lyh/python_project/EEGBCI-SEF || exit 1
git pull origin main -q
P=/home/opt/anaconda/envs/torch/bin/python
mkdir -p logs_e4c
MAXJ=6

run_ds () {  # $1=dataset $2=arm $3..=subjects
  local ds=$1 arm=$2; shift 2
  for s in "$@"; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
    "$P" scripts/e4/e4c_dualbranch_gate.py --dataset "$ds" --arm "$arm" --subjects "$s" \
       > "logs_e4c/${ds}_${arm}_s${s}.log" 2>&1 &
  done
  wait
}

for arm in gate_l01 uniform; do
  run_ds bcic "$arm" 1 2 3 4 5 6 7 8 9
done
echo BCIC_DONE

for arm in gate_l01 uniform slow_only osc_only gate_l0 gate_l10; do
  run_ds xw "$arm" $(seq 1 50)
done
echo ALL_DONE_E4C
