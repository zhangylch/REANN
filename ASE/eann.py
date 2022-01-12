#----------------eann interface is for EANN package-------------------------------

import numpy as np
import os
import torch
import re
#from gpu_sel import *
from ase.data import chemical_symbols, atomic_numbers
#from ase.units import Bohr
#from ase.neighborlist import NeighborList
from ase.calculators.calculator import (Calculator, all_changes,
                                        PropertyNotImplementedError)

class EANN(Calculator):

    implemented_properties = ['energy', 'forces']

    nolabel = True

    #default_parameters = {'asap_cutoff': False}

    def __init__(self,atomtype, device='cpu',period=[1,1,1],nn = 'EANN_PES_DOUBLE.pt',**kwargs):
        Calculator.__init__(self, **kwargs)
        self.device= device
        self.atomtype = atomtype
        self.period = period
        pes=torch.jit.load(nn)
        pes.to(device).to(torch.double)
        pes.eval()
        # pes=torch.jit.optimize_for_inference(pes)
        self.pes = pes

    #def initialize(self,atomtype,divice='cpu',pes='EANN_PES_DOUBLE.pt'):
    #    if divice  == None:
    #        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #    if atomtype == None:
    #        print('there in no atomtype')
        
    def transfer2eann(self,atoms):
        species = self.atoms.symbols
        cell = np.array([list(arr) for arr in list(self.atoms.cell)])
        cart = self.atoms.positions
        species = str(species)
        #number = re.sub('\D'," ", species).split()
        #number_atom = [int(num) for num in number]
        #spe = re.sub('\d+\.?\d*' ,' ',species).split()
        spe1=re.sub( r"([A-Z])", r" \1", species).split()
        spe = []  
        number_atom = []
        for sp in spe1:
            s = re.sub('\d+\.?\d*' ,' ',sp).split()
            num_atom = re.sub('\D'," ", sp).split()
            if len(num_atom) == 0:
                number_atom.append(1)
            else:
                number_atom.append(int(num_atom[0]))
            spe.append(s[0])
        #print(number_atom,spe) 
        return cell,cart,number_atom,spe
    
    def eann_all_str(self,number_atom,spe,atomtype):
        species = []
        atomtype_number = [i for i in range(len(atomtype))]
        spe_number = []
        for i in range(len(spe)):
            for j in range(len(atomtype)):
                if spe[i] == atomtype[j]:
                    spe_number.append(atomtype_number[j])
        number_sum = np.sum(np.array(number_atom))
        
        for i in range(number_sum):
            # a = 0
            number_a1 = 0
            for i1 in range(len(number_atom)):
                number_a1 += number_atom[i1]
                if i - number_a1 < 0:
                    species.append(spe_number[i1])
                    break
        return species

    
    def calculate(self,atoms=None, properties=['energy'],
                  system_changes=all_changes):
        pes = self.pes
        period=self.period
        atomtype = self.atomtype
        device=self.device
        
        Calculator.calculate(self, atoms, properties, system_changes)
        cell,cart,number_atom,spe = self.transfer2eann(self.atoms)
        species = self.eann_all_str(number_atom,spe,atomtype)
        period_table=torch.tensor(period,dtype=torch.float,device=device)
        species=torch.from_numpy(np.array(species)).to(device)  # from numpy array to torch tensor
        cart=torch.from_numpy(np.array(cart)).to(device).to(torch.double)  # also float32/double
        tcell=torch.from_numpy(cell).to(device).to(torch.double)  # also float32/double
        #pes=torch.jit.load(nn)
        #pes.to(device).to(torch.double)
        #pes.eval()
        energy=pes(period_table,cart,tcell,species)[0]
        force=pes(period_table,cart,tcell,species)[1]
        energy = float(energy.detach().numpy())
        force = force.detach().numpy()
        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = force

