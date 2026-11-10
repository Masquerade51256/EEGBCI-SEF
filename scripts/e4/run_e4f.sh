#!/bin/bash
# E4 第 3 步 S3-C（B 主干 + 连接分支）：BCIC 效度先行（9 被试 × 3 seeds）
# 效度线（BCIC >= ~76%）通过后再启动 XW——见 E4_实验设计_草案_20261105.md §13.6
cd /home/user_lyh/python_project/EEGBCI-SEF || exit 1
git pull origin main -q
P=/home/opt/anaconda/envs/torch/bin/python
mkdir -p logs_e4f
MAXJ=3

for s in 1 2 3 4 5 6 7 8 9; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
  "$P" scripts/e4/e4f_step3_conn.py --dataset bcic --subjects "$s" \
     > "logs_e4f/bcic_S3C_s${s}.log" 2>&1 &
done
wait
echo BCIC_S3C_DONE
