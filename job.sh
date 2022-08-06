#!/bin/bash -x
#PJM -L rscgrp=cx-small
#PJM -L node=4
#PJM -L elapse=128:00:00
#PJM -j
#PJM -S

module load gcc/8.4.0
module load python/3.8.3
module load cuda/11.6.2
module load cudnn
module load nccl
module load openmpi_cuda/4.1.2

source /home/z44406a/.pyenv/versions/vdvae2/bin/activate

# Distributed setting
# -np == #nodes

export WORLD_SIZE=16
export MASTER_PORT=12356
echo "WORLD_SIZE=${WORLD_SIZE}"
echo "MASTER_PORT=${MASTER_PORT}"


mpirun \
    -np 4 \
    -npernode 1 \
    -bind-to none \
    -map-by slot \
    -x NCCL_SOCKET_IFNAME="ib0" \
    -mca pml ob1 \
    -mca btl ^openib \
    -mca btl_tcp_if_include ib0 \
    -mca plm_rsh_agent /bin/pjrsh \
    -machinefile ${PJM_O_NODEINF} \
    /home/z44406a/projects/vdvae/job_node.sh

#    -x NCCL_DEBUG=INFO \