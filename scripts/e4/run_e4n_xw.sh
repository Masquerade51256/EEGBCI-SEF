#!/bin/bash
# E4 第 3 步 S3-D（蒸馏分支：非周期 + 区域网格 wPLI）：XW 主实验（融合 + 单体）
cd /home/user_lyh/python_project/EEGBCI-SEF || exit 1
git pull origin main -q
P=/home/opt/anaconda/envs/torch/bin/python
mkdir -p logs_e4n
MAXJ=3

for s in $(seq 1 50); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
  "$P" scripts/e4/e4n_step3_distill.py --dataset xw --subjects "$s" \
     > "logs_e4n/xw_S3D_s${s}.log" 2>&1 &
done
wait
echo XW_S3D_DONE

for s in $(seq 1 50); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  "$P" scripts/e4/e4n_step3_distill.py --dataset xw --subjects "$s" --distill-only \
     > "logs_e4n/xw_S3Donly_s${s}.log" 2>&1 &
done
wait
echo XW_S3DONLY_DONE
