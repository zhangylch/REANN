# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:52:34 2021

@author: Jia-lan Chen
"""

import ase.io.vasp
from ase import Atoms
from ase.calculators.eann import EANN
from ase.calculators.reann import REANN
#import os
#import re
from ase.optimize.minimahopping import MinimaHopping
from ase.optimize.minimahopping import MHPlot
from ase.optimize import LBFGS
from ase.io.trajectory import Trajectory

cell1 = ase.io.vasp.read_vasp("POSCAR")
atoms = Atoms(cell1)
#print(atoms.positions)
#--------------the type of atom, which is the same as atomtype which is in para/input_denisty--------------
atomtype = ['Pt','Fe','O']
#atomtype = ['Cu','Ce','O','C']
#-----------------the device is cpu or gpu( cpu is default)---------------
device='cpu'
#-------------------------pbc([1,1,1] is default)---------------------
period=[1,1,1]
#---------------nn file('EANN_PES_DOUBLE.pt' is default in eann,'REANN_PES_DOUBLE.pt' is default in reann)----------------------------
nn = 'REANN_PES_DOUBLE.pt'
#----------------------eann (if you use EANN package )--------------------------------
#atoms.calc = EANN(device=device,atomtype=atomtype,period=period,nn = nn)
#----------------------------reann (if you use REANN package ****recommend****)---------------------------------
atoms.calc = REANN(device=device,atomtype=atomtype,period=period,nn = nn)
print(atoms)
dyn = LBFGS(atoms,trajectory='atom2.traj')
dyn.run(fmax=0.2,steps=100)
traj = Trajectory('atom2.traj')
atoms = traj[-1]
print(atoms.get_potential_energy())
ase.io.write('POSCAR-final', atoms, format='vasp', vasp5='True')
#print(atoms.get_potential_energy())
#print(atoms)
#e= atoms.get_potential_energy()
#f = atoms.get_forces()
#print(e,f)
