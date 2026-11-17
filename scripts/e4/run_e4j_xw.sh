#!/bin/bash
# E4 第 3 步 S3-C v3（协方差池化分支）：XW 主实验（融合 + 单体）
# 前提：BCIC 效度已通过（融合 88.38% / 单体 77.94%，2026-11-17）
cd /home/user_lyh/python_project/EEGBCI-SEF || exit 1
git pull origin main -q
P=/home/opt/anaconda/envs/torch/bin/python
mkdir -p logs_e4j
MAXJ=3

for s in $(seq 1 50); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
  "$P" scripts/e4/e4j_step3_cov.py --dataset xw --subjects "$s" \
     > "logs_e4j/xw_S3COV_s${s}.log" 2>&1 &
done
wait
echo XW_S3COV_DONE

for s in $(seq 1 50); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  "$P" scripts/e4/e4j_step3_cov.py --dataset xw --subjects "$s" --cov-only \
     > "logs_e4j/xw_S3COVonly_s${s}.log" 2>&1 &
done
wait
echo XW_S3COVONLY_DONE
