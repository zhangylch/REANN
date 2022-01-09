#!/bin/sh
#PBS -V
#PBS -q gpu
#PBS -N co2+ni100
#PBS -l nodes=1:ppn=1
source /share/home/bjiangch/group-zyl/.bash_profile
# conda environment
conda_env=PyTorch-190
export OMP_NUM_THREADS=16
#path to save the code
path="/home/home/zyl/pytorch/2021_8_1/eann-8/"

#Number of processes per node to launch
NPROC_PER_NODE=1

#Number of process in all modes
WORLD_SIZE=`expr $PBS_NUM_NODES \* $NPROC_PER_NODE`

MASTER=`/bin/hostname -s`

MPORT=`ss -tan | awk '{print $5}' | cut -d':' -f2 | \
        grep "[2-9][0-9]\{3,3\}" | sort | uniq | shuf -n 1`

#You will want to replace this
COMMAND="$path "
conda activate $conda_env 
cd $PBS_O_WORKDIR 
#python3 -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --max_restarts=0 --nnodes=$PBS_NUM_NODES --rdzv_id=$PBS_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER:$MPORT $COMMAND > out
python3 -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --max_restarts=0 --nnodes=1 --standalone $COMMAND > out

