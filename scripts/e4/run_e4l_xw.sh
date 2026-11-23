#!/bin/bash
# E4 第 3 步 S3-C v4（iCoh 虚部相干分支）：XW 主实验（融合 + 单体）
cd /home/user_lyh/python_project/EEGBCI-SEF || exit 1
git pull origin main -q
P=/home/opt/anaconda/envs/torch/bin/python
mkdir -p logs_e4l
MAXJ=3

for s in $(seq 1 50); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
  "$P" scripts/e4/e4l_step3_icoh.py --dataset xw --subjects "$s" \
     > "logs_e4l/xw_S3ICOH_s${s}.log" 2>&1 &
done
wait
echo XW_S3ICOH_DONE

for s in $(seq 1 50); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  "$P" scripts/e4/e4l_step3_icoh.py --dataset xw --subjects "$s" --icoh-only \
     > "logs_e4l/xw_S3ICOHonly_s${s}.log" 2>&1 &
done
wait
echo XW_S3ICOHONLY_DONE
