# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:52:34 2021

@author: Jia-lan Chen


modified by Yaolong Zhang for a better efficiency
"""

import torch
import ase.io.vasp
from ase import Atoms
from ase.calculators.reann import REANN
import getneigh as getneigh
from ase.io import extxyz
import time


fileobj=open("h2o.extxyz",'r')
configuration = extxyz.read_extxyz(fileobj,index=slice(0,100))
#--------------the type of atom, which is the same as atomtype which is in para/input_denisty--------------
atomtype = ['O','H']
#-----------------the device is cpu or gpu( cpu is default)---------------
device='cpu'
#-------------------------pbc([1,1,1] is default)---------------------
maxneigh=25000  # maximal number of the neighbor atoms for the configuration (summation of neighbor atoms for each center atom)
#----------------------------reann (if you use REANN package ****recommend****)---------------------------------
calc=REANN(atomtype,maxneigh, getneigh, potential = "PES.pt", device=device, dtype = torch.float32)
start=time.time()
num=0.0
for atoms in configuration:
    calc.reset()
    atoms.calc=calc
    ene = atoms.get_potential_energy(apply_constraint=False)
    force = atoms.get_forces()
    num+=ene
print(num)
end=time.time()
print(end-start)
