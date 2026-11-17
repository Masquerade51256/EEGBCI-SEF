#!/bin/bash
# E4 第 3 步 S3-C v3（协方差池化分支）：BCIC 效度（融合 + 单体）
cd /home/user_lyh/python_project/EEGBCI-SEF || exit 1
git pull origin main -q
P=/home/opt/anaconda/envs/torch/bin/python
mkdir -p logs_e4j
MAXJ=3

for s in 1 2 3 4 5 6 7 8 9; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
  "$P" scripts/e4/e4j_step3_cov.py --dataset bcic --subjects "$s" \
     > "logs_e4j/bcic_S3COV_s${s}.log" 2>&1 &
done
wait
echo BCIC_S3COV_DONE

for s in 1 2 3 4 5 6 7 8 9; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  "$P" scripts/e4/e4j_step3_cov.py --dataset bcic --subjects "$s" --cov-only \
     > "logs_e4j/bcic_S3COVonly_s${s}.log" 2>&1 &
done
wait
echo BCIC_S3COVONLY_DONE
