#!/bin/bash
# E4 第 3 步：连接分支 v2 单体（S3C2conn）——BCIC + XW
# 目的：区分"分支学不到连接信息"与"被 B 主干冗余覆盖"
cd /home/user_lyh/python_project/EEGBCI-SEF || exit 1
git pull origin main -q
P=/home/opt/anaconda/envs/torch/bin/python
mkdir -p logs_e4i
MAXJ=3

for s in 1 2 3 4 5 6 7 8 9; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  "$P" scripts/e4/e4i_step3_conn2.py --dataset bcic --subjects "$s" --conn-only \
     > "logs_e4i/bcic_S3C2conn_s${s}.log" 2>&1 &
done
wait
echo BCIC_CONN_ONLY_DONE

for s in $(seq 1 50); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  "$P" scripts/e4/e4i_step3_conn2.py --dataset xw --subjects "$s" --conn-only \
     > "logs_e4i/xw_S3C2conn_s${s}.log" 2>&1 &
done
wait
echo XW_CONN_ONLY_DONE
