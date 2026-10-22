#!/bin/bash
# E1c+ 健康<->患者迁移队列：zero-shot（CSP + EEGNet×3seeds）+ 健康->患者 few-shot
# 看门狗模式：每个任务最多重试 3 次；脚本内置增量 CSV + resume，重跑安全。
# 用法：nohup bash scripts/e1/run_e1cplus.sh > logs/e1cplus.log 2>&1 &
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

# --- 1) zero-shot CSP（CPU，快）---
for d in bcic2xw bcic2hs xw2bcic hs2bcic; do
  run3 "csp-$d" $PY scripts/e1/e1c_transfer.py --direction "$d" --arm csp --seeds 42,123,2026
done

# --- 2) zero-shot EEGNet ×3 seeds ---
for d in bcic2xw bcic2hs xw2bcic hs2bcic; do
  run3 "eegnet-$d" $PY scripts/e1/e1c_transfer.py --direction "$d" --arm eegnet --seeds 42,123,2026
done

# --- 3) 健康->患者 few-shot（finetune / scratch）---
for d in bcic2xw bcic2hs; do
  for a in finetune scratch; do
    run3 "fewshot-$d-$a" $PY scripts/e1/e1_fewshot.py --direction "$d" --arm "$a" --seeds 42,123,2026
  done
done

echo "[$(date +%H:%M:%S)] ALL DONE"
