#!/bin/bash
# E4 第 3 步 S3-C：XW 主实验（50 被试 × 3 seeds）
# 前提：BCIC 效度已通过（87.94% >= 76%，2026-11-10）
cd /home/user_lyh/python_project/EEGBCI-SEF || exit 1
git pull origin main -q
P=/home/opt/anaconda/envs/torch/bin/python
mkdir -p logs_e4f
MAXJ=3

for s in $(seq 1 50); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
  "$P" scripts/e4/e4f_step3_conn.py --dataset xw --subjects "$s" \
     > "logs_e4f/xw_S3C_s${s}.log" 2>&1 &
done
wait
echo XW_S3C_DONE
