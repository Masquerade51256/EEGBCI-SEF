#!/bin/bash
# S3-D v3a（阶段1正则化三件套）BCIC 效度线
cd /home/user_lyh/python_project/EEGBCI-SEF
PY=/home/opt/anaconda/envs/torch/bin/python
$PY scripts/e4/e4q_distill3_reg.py --dataset bcic
echo "BCIC_S3DREG_DONE"
$PY scripts/e4/e4q_distill3_reg.py --dataset bcic --distill-only
echo "BCIC_S3DREGONLY_DONE"
