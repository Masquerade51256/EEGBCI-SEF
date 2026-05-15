#!/bin/bash
#
# 模型归因对照实验一键启动脚本
# 
# 运行前请确保：
#   1. 已在 /home/user_lyh/python_project/EEGBCI-SEF 目录下
#   2. conda 环境 'torch' 可用
#
# 用法（VS Code 终端直接执行）：
#   bash scripts/start_model_ablation.sh
#
# 后台运行（推荐，关闭终端也不中断）：
#   nohup bash scripts/start_model_ablation.sh > model_ablation.log 2>&1 &
#   # 然后可以用: tail -f model_ablation.log 查看实时进度

set -e  # 遇到错误立即退出

echo "=============================================================="
echo "  Model Ablation / Attribution Experiments"
echo "=============================================================="
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 定位项目目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "Project dir: $PROJECT_DIR"
echo ""

# 激活 conda 环境
echo "[Step 0] Activating conda environment 'torch'..."
source /home/opt/anaconda/etc/profile.d/conda.sh
conda activate torch

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# 检查已有结果，避免重复运行（如需强制重跑，可删除对应 experiments/ 目录）
echo "[Step 0.5] Checking existing results..."
for exp_name in "BCICIV2a_ADFCNN_StreamingLOSO" "XWStroke_Full50_EEGNet_LOSO_LR" "XWStroke_Full50_EEGNet_LOSO_Affected_Aligned"; do
    res_file="experiments/$exp_name/results/results.json"
    if [ -f "$res_file" ]; then
        mean=$(python -c "import json; d=json.load(open('$res_file')); print(f\"{d['overall_mean']*100:.2f}%\")")
        echo "  ⚠️  $exp_name already exists (mean=$mean). Skip if you want to keep it."
        echo "      To force rerun: rm -rf experiments/$exp_name"
    fi
done
echo ""

# 运行归因实验 Python 脚本
echo "[Step 1] Starting all attribution experiments..."
echo ""

python scripts/run_model_ablation.py

echo ""
echo "=============================================================="
echo "  All experiments completed!"
echo "  End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="
echo ""
echo "Results location:"
echo "  - experiments/BCICIV2a_ADFCNN_StreamingLOSO/        (E1: healthy baseline)"
echo "  - experiments/XWStroke_Full50_EEGNet_LOSO_LR/       (E2: EEGNet + LR)"
echo "  - experiments/XWStroke_Full50_EEGNet_LOSO_Affected_Aligned/ (E3: EEGNet + Aff+Align)"
echo "  - results/model_ablation/ablation_summary.json      (auto comparison)"
echo ""
