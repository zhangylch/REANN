# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:52:34 2021

@author: Jialan Chen
"""

import ase.io.vasp
from ase import Atoms
#from ase.calculators.eann import EANN
from ase.calculators.reann import REANN
from ase.optimize import LBFGS
from ase.io.trajectory import Trajectory
import numpy as np

cell1 = ase.io.vasp.read_vasp("POSCAR")
atoms = Atoms(cell1)
#--------------the type of atom, which is the same atomtype as para/input_density--------------
atomtype = ['Pt','Fe','O']
#atomtype = ['Cu','Ce','O','C']
#-----------------the device is cpu or gpu( cpu is default)---------------
device='cpu'
#-------------------------pbc([1,1,1] is default)---------------------
period=[1,1,1]
#---------------nn file('EANN_PES_DOUBLE.pt' is default)----------------------------
nn = 'REANN_PES_DOUBLE.pt'
#----------------------eann --------------------------------
atoms.calc = REANN(device=device,atomtype=atomtype,period=period,nn=nn)
#atoms.calc = EANN()
#----------------------------reann---------------------------------
#atoms.calc = REANN(atomtype=atomtype,period=[1,1,0],nn = 'EANN_PES_DOUBLE.pt')
#print(atoms.get_potential_energy())
dyn = LBFGS(atoms,trajectory='atom2.traj')
dyn.run(fmax=0.2,steps=100)
traj = Trajectory('atom2.traj')
atoms = traj[-1]
ase.io.write('POSCAR-final', atoms, format='vasp', vasp5='True')
f = atoms.get_forces()
print(np.max(f))
