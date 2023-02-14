#!/bin/sh
#SBATCH -J dos
#SBATCH --gpus=1
#SBATCH -N 1
##SBATCH --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
echo Running on hosts
echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
# Your conda environment
conda_env=Pytorch
export OMP_NUM_THREADS=8

module add cuda/10.2
module add cudnn/7.6.5.32_cuda10.2

#ATTENTION! HERE MUSTT BE ONE LINE,OR ERROR!
source ~/.bashrc
source activate pt110
cd $PWD

#Number of processes per node to launch (20 for CPU, 2 for GPU)
NPROC_PER_NODE=1

#The path you place your code
path="/data/home/scv2201/run/zyl/program/REANN/reann/"
#This command to run your pytorch script
#You will want to replace this
COMMAND="$path"

#We want names of master and slave nodes
MASTER=`/bin/hostname -s`

MPORT=`ss -tan | awk '{print $4}' | cut -d':' -f2 | \
      grep "[2-9][0-9]\{3,3\}" | grep -v "[0-9]\{5,5\}" | \
      sort | uniq | shuf`

python -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --nnodes=$SLURM_JOB_NUM_NODES --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER:11416 $COMMAND >out
#python3 -m torch.distributed.run --nproc_per_node=1 --standalone --nnodes=1 --max_restarts=0 $COMMAND>out
