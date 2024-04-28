# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:52:34 2021

@author: Jia-lan Chen


modified by Yaolong Zhang for a better efficiency
"""

import torch
import ase.io.vasp
from ase import Atoms, units
import getneigh as getneigh
from ase.calculators.reann import REANN
from ase.io import extxyz
from ase.io.trajectory import Trajectory
import time

from ase.optimize import BFGS,FIRE
from ase.constraints import ExpCellFilter
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen as NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution as MBD
import numpy as np

ef=np.zeros(3)
ef[2]=0.0
fileobj=open("h2o.extxyz",'r')
configuration = extxyz.read_extxyz(fileobj,index=slice(0,1))
#--------------the type of atom, which is the same as atomtype which is in para/input_denisty--------------
atomtype = ['O','H']
#-----------------the device is cpu or gpu( cpu is default)---------------
device='cpu'
maxneigh=50000# maximal number of the neighbor atoms for the configuration (summation of neighbor atoms for each center atom)
#----------------------------reann (if you use REANN package ****recommend****)---------------------------------
calc=REANN(atomtype,maxneigh, getneigh, properties=['energy', 'forces', 'stress'], nn = "PES.pt", device=device, dtype = torch.float32)
start=time.time()
num=0.0
for atoms in configuration:
    calc.reset()
    atoms.calc=calc
    MBD(atoms,temperature_K=300)
    traj = Trajectory('h2o.traj', 'w', atoms)
    dyn = NPT(atoms, timestep=0.25 * units.fs, temperature_K=300,
                   taut=500 * units.fs, pressure_au=1.01325 * units.bar, fixcm=True,
                   taup=1000 * units.fs, compressibility_au=5e-6 / units.bar, logfile="md.log",loginterval=10)
    dyn.attach(traj.write, interval=200)
    dyn.run(steps=100000)
    traj.close()
