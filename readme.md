Recursively embedded atom neural network 
=================================================
**Introduction:**
___________________________
  Recursively embedded atom neural network (REANN) is a python package implemented based on the PyTorch framework used to train interatomic potentials, dipole moments, transition dipole moments and polarizabilities of various systems. Making use of the autograd and distributed parallelism implementations in PyTorch, REANN can run spanning multiple nodes on the GPU/CPU with high efficiency. The output model can be compiled to a serialized file via torchscript, which can be loaded in C++ for convenience and efficient interfacing with molecular dynamics software. Currently, REANN model has been interfaced with LAMMPS as a new "pair_style". More details can be found in the manual.

**Requirements:**
___________________________________
* PyTorch 1.9.0
* LibTorch 1.9.0
* cmake 3.1.0
* opt_einsum 3.2.0

**References:**
1. The original EANN model: Yaolong Zhang, Ce Hu and Bin Jiang *J. Phys. Chem. Lett.* 10, 4962-4967 (2019).
2. The EANN model for dipole/transition dipole/polarizability: Yaolong Zhang  Sheng Ye, Jinxiao Zhang, Jun Jiang and Bin Jiang *J. Phys. Chem. B*  124, 7284–7290 (2020).
3. The REANN model: Yaolong Zhang, Junfan xia and Bin Jiang *Phys. Rev. Lett.* 127, 156002 (2021).
