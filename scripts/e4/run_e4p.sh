#!/bin/bash
# S3-D v2a（教师 PCA 降维）BCIC 效度线
cd /home/user_lyh/python_project/EEGBCI-SEF
PY=/home/opt/anaconda/envs/torch/bin/python
$PY scripts/e4/e4p_distill2_pca.py --dataset bcic
echo "BCIC_S3DPCA_DONE"
$PY scripts/e4/e4p_distill2_pca.py --dataset bcic --distill-only
echo "BCIC_S3DPCAONLY_DONE"
