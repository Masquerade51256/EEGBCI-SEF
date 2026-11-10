#!/bin/bash
# E4 第 3 步 S3-C 机制闭环：连接分支遮蔽（XW 50 被试，加载存盘模型，仅推理）
cd /home/user_lyh/python_project/EEGBCI-SEF || exit 1
git pull origin main -q
P=/home/opt/anaconda/envs/torch/bin/python
mkdir -p logs_e4h
MAXJ=3

for s in $(seq 1 50); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
  "$P" scripts/e4/e4h_step3_conn_masking.py --dataset xw --subjects "$s" \
     > "logs_e4h/xw_S3C_mask_s${s}.log" 2>&1 &
done
wait
echo XW_S3C_MASK_DONE
