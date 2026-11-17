#!/bin/bash
# E4 第 3 步 S3-R：XW 主实验（50 被试 × 3 seeds，融合 + 单体）
# 前提：BCIC 效度已通过（S3R 87.77% / S3Ronly 87.22%，2026-11-17）
cd /home/user_lyh/python_project/EEGBCI-SEF || exit 1
git pull origin main -q
P=/home/opt/anaconda/envs/torch/bin/python
mkdir -p logs_e4g
MAXJ=3

for s in $(seq 1 50); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
  "$P" scripts/e4/e4g_step3_resid.py --dataset xw --subjects "$s" \
     > "logs_e4g/xw_S3R_s${s}.log" 2>&1 &
done
wait
echo XW_S3R_DONE

for s in $(seq 1 50); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  "$P" scripts/e4/e4g_step3_resid.py --dataset xw --subjects "$s" --resid-only \
     > "logs_e4g/xw_S3Ronly_s${s}.log" 2>&1 &
done
wait
echo XW_S3RONLY_DONE
