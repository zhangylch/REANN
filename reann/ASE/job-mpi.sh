#!/bin/sh
#PBS -V
#PBS -q testv5
#PBS -N h2o-0-reann
#PBS -l nodes=1:ppn=12
#export CUDA_VISIBLE_DEVICES="0,1"
export OMP_NUM_THREADS=12
source /public/home/group_zyl/.bashrc
#export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH #for specify the cuda path to overcome error: failed to open libnvrtc-builtins.so.11.1
cd $PBS_O_WORKDIR
conda activate pt200
python3 npt.py >q 
