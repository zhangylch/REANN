#!/bin/sh
#PBS -V
#PBS -q gpu
#PBS -N test_lammps_64
#PBS -l nodes=1:ppn=1
export CUDA_VISIBLE_DEVICES=[1,0]
export OMP_NUM_THREADS=1
source /share/home/bjiangch/group-zyl/.bash_profile
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH #for specify the cuda path to overcome error: failed to open libnvrtc-builtins.so.11.1
cd $PBS_O_WORKDIR
mpirun -n 1 ./lmp_mpi -in in.h2o >out
