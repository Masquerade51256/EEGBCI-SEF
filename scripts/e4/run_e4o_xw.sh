#!/bin/bash
# S3-D 蒸馏保真度诊断（XW，50 被试全量）
cd /home/user_lyh/python_project/EEGBCI-SEF
PY=/home/opt/anaconda/envs/torch/bin/python
$PY scripts/e4/e4o_distill_fidelity.py --dataset xw
echo "E4O_XW_DONE"
