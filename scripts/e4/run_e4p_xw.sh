#!/bin/bash
# S3-D v2a（教师 PCA 降维）XW 主实验：融合 → 单体
cd /home/user_lyh/python_project/EEGBCI-SEF
PY=/home/opt/anaconda/envs/torch/bin/python
$PY scripts/e4/e4p_distill2_pca.py --dataset xw
echo "XW_S3DPCA_DONE"
$PY scripts/e4/e4p_distill2_pca.py --dataset xw --distill-only
echo "XW_S3DPCAONLY_DONE"
