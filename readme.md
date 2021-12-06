Recursively embedded atom neural network 
=================================================
**Introduction:**
___________________________
  Recursively embedded atom neural network (REANN) is a PyTorch-based end-to-end multi-functional Deep Neural Network Package for Molecular, Reactive and Periodic Systems. Currently, REANN can be used to train interatomic potentials, dipole moments, transition dipole moments, and polarizabilities. Taking advantage of Distributed DataParallel features embedded in PyTorch, the training process is highly parallelized on both GPU and CPU. For the convenience of MD simulation, an interface to LAMMPS has been constructed by creating a new pair_style invoking this representation for highly efficient MD simulations. More details can be found in the manual saved in the "manual" folder and an example has been placed in "reann/example/" for constructing the ML potential of the co2+Ni(100) system using the data located in the "data/" folder.

**Requirements:**
___________________________________
* PyTorch 1.9.0
* LibTorch 1.9.0
* cmake 3.1.0
* opt_einsum 3.2.0

**References:**
_________________________________________________
If you use this package, please cite these works.
1. The original EANN model: Yaolong Zhang, Ce Hu and Bin Jiang *J. Phys. Chem. Lett.* 10, 4962-4967 (2019).
2. The EANN model for dipole/transition dipole/polarizability: Yaolong Zhang  Sheng Ye, Jinxiao Zhang, Jun Jiang and Bin Jiang *J. Phys. Chem. B*  124, 7284–7290 (2020).
3. The theory of REANN model: Yaolong Zhang, Junfan xia and Bin Jiang *Phys. Rev. Lett.* 127, 156002 (2021).
4. The details about the implementation of REANN: Yaolong Zhang, Junfan xia and Bin Jiang *arXiv:2112.01774*.
