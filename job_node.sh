#!/bin/bash

set -euo pipefail

source /etc/profile.d/modules.sh
module load gcc/8.4.0
module load python/3.8.3
module load cuda/11.6.2
module load cudnn
module load nccl
module load openmpi_cuda/4.1.2

source /home/z44406a/.pyenv/versions/vdvae2/bin/activate

# https://github.com/pytorch/pytorch/issues/37377
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1

# distributed setting
MY_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_ADDR=$(head -n 1 ${PJM_O_NODEINF})
NODE_RANK=$(cat ${PJM_O_NODEINF} | awk '{print NR-1 " " $1}' | grep ${MY_ADDR}$ | awk '{print $1}')
#echo "MY_ADDR=${MY_ADDR}"
#echo "MASTER_ADDR=${MASTER_ADDR}"
#echo "NODE_RANK=${NODE_RANK}"

for i in 0 1 2 3; do
(
    export RANK=$((${i} + 4 * ${NODE_RANK}))
    export LOCAL_RANK=$i
    echo "MASTER_ADDR=${MASTER_ADDR}, MY_ADDR=${MY_ADDR}, NODE_RANK=${NODE_RANK}, RANK=${RANK}, LOCAL_RANK=${LOCAL_RANK}"
    python train.py --hps bev256 --port 12356 --save_dir ./saved_models_256_1node_05
) & 
done
wait
