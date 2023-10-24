# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:52:34 2021

@author: Jia-lan Chen


modified by Yaolong Zhang for a better efficiency
"""

import torch
import ase.io.vasp
from ase import Atoms, units
#from ase.calculators.reann import REANN
import getneigh as getneigh
from ase.calculators.reann import REANN
from ase.io import extxyz
import time

from ase.optimize import BFGS,FIRE
from ase.constraints import ExpCellFilter
from ase.md.langevin import Langevin

fileobj=open("h2o.extxyz",'r')
configuration = extxyz.read_extxyz(fileobj,index=slice(0,1))
#--------------the type of atom, which is the same as atomtype which is in para/input_denisty--------------
atomtype = ['O','H']
#-----------------the device is cpu or gpu( cpu is default)---------------
device='cpu'
#-------------------------pbc([1,1,1] is default)---------------------
maxneigh=25000# maximal number of the neighbor atoms for the configuration (summation of neighbor atoms for each center atom)
#----------------------------reann (if you use REANN package ****recommend****)---------------------------------
calc=REANN(atomtype,maxneigh, getneigh, potential = "PES.pt", device=device, dtype = torch.float32)
start=time.time()
num=0.0
for atoms in configuration:
    calc.reset()
    atoms.calc=calc
    #ene = atoms.get_potential_energy(apply_constraint=False)

    #force = atoms.get_forces()
#    num+=ene
    dyn = Langevin(atoms, 0.5*units.fs, temperature_K=300, friction=5e-3)
    #dyn = FIRE(atoms= atoms, trajectory=  'test.traj')
    dyn.run(steps=50)

