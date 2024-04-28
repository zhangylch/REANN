#----------------reann interface is for REANN package-------------------------------


import numpy as np
import os
import torch
import re
#from gpu_sel import *
from ase.data import chemical_symbols, atomic_numbers
from ase.units import Bohr
from ase.calculators.calculator import (Calculator, all_changes,
                                        PropertyNotImplementedError)

class REANN(Calculator):

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, atomtype, maxneigh, getneigh, properties=['energy', 'forces'], nn = 'PES.pt',device="cpu",dtype=torch.float32,**kwargs):
        Calculator.__init__(self, **kwargs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.atomtype = atomtype
        self.maxneigh=maxneigh
        self.getneigh=getneigh
        pes=torch.jit.load(nn)
        pes.to(self.device).to(self.dtype)
        pes.eval()
        self.cutoff=pes.cutoff
        self.tcell=[]
        self.properties=properties
        self.pes=torch.jit.optimize_for_inference(pes)
        self.table=0
        #self.pes=torch.compile(pes)
    
    def calculate(self,atoms=None, properties=['energy','force'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        cell=np.array(self.atoms.cell)
        if "cell" in system_changes:
            if cell.ndim==1:
                cell=np.diag(cell)
            self.getneigh.init_neigh(self.cutoff,self.cutoff/2.0,cell.T)
            self.tcell=torch.from_numpy(cell).to(self.dtype).to(self.device)
        icart = self.atoms.get_positions()
        cart,neighlist,shiftimage,scutnum=self.getneigh.get_neigh(icart.T,self.maxneigh)
        cart=torch.from_numpy(cart.T).contiguous().to(self.device).to(self.dtype)
        neighlist=torch.from_numpy(neighlist[:,:scutnum]).contiguous().to(self.device).to(torch.long)
        shifts=torch.from_numpy(shiftimage.T[:scutnum,:]).contiguous().to(self.device).to(self.dtype)
        symbols = list(self.atoms.symbols)
        species = [self.atomtype.index(i) for i in symbols]
        species = torch.tensor(species,device=self.device,dtype=torch.long)
        disp_cell = torch.zeros_like(self.tcell)
        if "forces" in self.properties:
            cart.requires_grad=True
        else:
            cart.requires_grad=False

        if "stress" in self.properties:
            disp_cell.requires_grad=True
        else:
            disp_cell.requires_grad=False

        energy = self.pes(self.tcell,disp_cell,cart,neighlist,shifts,species)
        self.results['energy'] = float(energy.detach().cpu().numpy())
           
        if "forces" in properties and "stress" in properties:
            forces,virial = torch.autograd.grad(energy,[cart,disp_cell])
            forces = torch.neg(forces).detach().cpu().numpy()
            self.results['forces'] = forces
            virial = virial.detach().cpu().numpy()
            self.results['stress'] = virial/self.atoms.get_volume()
        elif "forces" in properties and "stress" not in properties:
            forces = torch.autograd.grad(energy,cart)[0]
            forces = torch.neg(forces).detach().cpu().numpy()
            self.results['forces'] = forces
        elif "stress" in properties and "forces" not in properties:
            virial = torch.autograd.grad(energy,disp_cell)[0]
            virial = virial.detach().cpu().numpy()
            self.results['stress'] = virial/self.atoms.get_volume()
