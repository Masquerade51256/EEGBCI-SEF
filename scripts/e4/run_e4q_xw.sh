#!/bin/bash
# S3-D v3a（阶段1正则化三件套）XW 主实验：融合 → 单体
cd /home/user_lyh/python_project/EEGBCI-SEF
PY=/home/opt/anaconda/envs/torch/bin/python
$PY scripts/e4/e4q_distill3_reg.py --dataset xw
echo "XW_S3DREG_DONE"
$PY scripts/e4/e4q_distill3_reg.py --dataset xw --distill-only
echo "XW_S3DREGONLY_DONE"
