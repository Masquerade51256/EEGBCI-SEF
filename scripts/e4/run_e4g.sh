#!/bin/bash
# E4 第 3 步 S3-R（B 主干 + 残差分支）：BCIC 效度（9 被试 × 3 seeds）
# 含单体条件 S3Ronly（2026-11-11 规则：每个分支必配单体，区分提取能力与冗余）
cd /home/user_lyh/python_project/EEGBCI-SEF || exit 1
git pull origin main -q
P=/home/opt/anaconda/envs/torch/bin/python
mkdir -p logs_e4g
MAXJ=3

for s in 1 2 3 4 5 6 7 8 9; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
  "$P" scripts/e4/e4g_step3_resid.py --dataset bcic --subjects "$s" \
     > "logs_e4g/bcic_S3R_s${s}.log" 2>&1 &
done
wait
echo BCIC_S3R_DONE

for s in 1 2 3 4 5 6 7 8 9; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  "$P" scripts/e4/e4g_step3_resid.py --dataset bcic --subjects "$s" --resid-only \
     > "logs_e4g/bcic_S3Ronly_s${s}.log" 2>&1 &
done
wait
echo BCIC_S3RONLY_DONE
