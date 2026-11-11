#!/bin/bash
# E4 第 3 步 S3-C v2（token 富化 + 窗级注意力）：BCIC 效度先行（9 被试 × 3 seeds）
cd /home/user_lyh/python_project/EEGBCI-SEF || exit 1
git pull origin main -q
P=/home/opt/anaconda/envs/torch/bin/python
mkdir -p logs_e4i
MAXJ=3

for s in 1 2 3 4 5 6 7 8 9; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
  "$P" scripts/e4/e4i_step3_conn2.py --dataset bcic --subjects "$s" \
     > "logs_e4i/bcic_S3C2_s${s}.log" 2>&1 &
done
wait
echo BCIC_S3C2_DONE
