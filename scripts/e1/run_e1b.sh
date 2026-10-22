#!/bin/bash
# E1b 升级队列：同阶段患者源 LOSO few-shot（XW + HS × finetune/scratch）
# 看门狗模式：每任务最多 3 次重试；脚本内置增量 CSV + resume，重跑安全。
# 用法：nohup bash scripts/e1/run_e1b.sh > logs/e1b.log 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
PY=/home/opt/anaconda/envs/torch/bin/python
mkdir -p logs

run3() {  # run3 <logname> <cmd...>
  local name="$1"; shift
  for i in 1 2 3; do
    echo "[$(date +%H:%M:%S)] RUN $name (attempt $i)"
    if "$@"; then echo "[$(date +%H:%M:%S)] OK  $name"; return 0; fi
    echo "[$(date +%H:%M:%S)] RETRY $name failed (attempt $i)"; sleep 20
  done
  echo "[$(date +%H:%M:%S)] GAVE UP $name"; return 1
}

# HS 人少（14），先跑早出数；XW（50 人）随后
for ds in hs xw; do
  for a in finetune scratch; do
    run3 "e1b-$ds-$a" $PY scripts/e1/e1b_loso_fewshot.py --dataset "$ds" --arm "$a" --seeds 42,123,2026
  done
done

echo "[$(date +%H:%M:%S)] ALL DONE"
