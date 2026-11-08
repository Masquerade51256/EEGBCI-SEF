#!/bin/bash
# E4 第 2 步因子检验：先 BCIC 效度（4 臂 × 9 被试），后 XW 主实验（4 臂 × 50 被试）
cd /home/user_lyh/python_project/EEGBCI-SEF || exit 1
git pull origin main -q
P=/home/opt/anaconda/envs/torch/bin/python
mkdir -p logs_e4e
MAXJ=6

run_ds () {  # $1=dataset $2=arm $3..=subjects
  local ds=$1 arm=$2; shift 2
  for s in "$@"; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
    "$P" scripts/e4/e4e_step2_factorial.py --dataset "$ds" --arm "$arm" --subjects "$s" \
       > "logs_e4e/${ds}_${arm}_s${s}.log" 2>&1 &
  done
  wait
}

for arm in B C D E; do
  run_ds bcic "$arm" 1 2 3 4 5 6 7 8 9
done
echo BCIC_DONE

for arm in B C D E; do
  run_ds xw "$arm" $(seq 1 50)
done
echo ALL_DONE_E4E
