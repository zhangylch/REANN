#----------------reann interface is for REANN package-------------------------------


import numpy as np
import os
import torch
import re
#from gpu_sel import *
from ase.data import chemical_symbols, atomic_numbers
from ase.units import Bohr
from ase.neighborlist import NeighborList
from ase.calculators.calculator import (Calculator, all_changes,
                                        PropertyNotImplementedError)

atom2mass={ 'H':1.008,     'He':4.003,   'Li':6.941,    'Be':9.012,   'B':10.811,     'C':12.017,     'N':14.007,     'O':15.999,
             'F':18.998,     'Ne':20.180,  'Na':22.990,   'Mg':24.305,  'Al':26.982,   'Si':28.086,   'P':30.974,    'S':32.065,
             'Cl':35.453,   'Ar':39.948,  'K':39.098 ,    'Ca':40.078,  'Sc':44.956,   'Ti':47.867,   'V':50.942,    'Cr':51.996,
             'Mn':54.938,   'Fe':55.845,  'Co':58.933,   'Ni':58.693,  'Cu':63.546,   'Zn':65.409,   'Ga':69.723,   'Ge':72.64,
             'As':74.922,   'Se':78.96,  'Br':79.904,   'Kr':83.798,  'Rb':85.467,   'Sr':87.62,   'Y':88.906,    'Zr':91.224,
             'Nb':92.907,   'Mo':95.94,  'Tc':97.907,   'Ru':101.07,  'Rh':102.905,   'Pd':106.42,   'Ag':107.868,   'Cd':112.411,
             'In':114.818,   'Sn':118.710,  'Sb':121.760,   'Te':127.60,  'I':126.904,    'Xe':131.293,   'Cs':132.905,   'Ba':137.327,
             'La':138.905,   'Ce':140.116,  'Pr':140.908,   'Nd':144.242,  'Pm':145,   'Sm':150.36,   'Eu':151.964,   'Gd':157.25,
             'Tb':158.925,   'Dy':162.500,  'Ho':164.930,   'Er':164.930,  'Tm':168.934,   'Yb':173.04,   'Lu':174.967,   'Hf':178.49,
             'Ta':180.948,   'W':183.84,   'Re':186.207,   'Os':190.23,  'Ir':192.217,   'Pt':195.084,   'Au':196.967,   'Hg':200.59,
             'Tl':204.383,   'Pb':207.2,  'Bi':208.980,   'Po':208.982,  'At':209.987,   'Rn':222.018,   'Fr':223,   'Ra':226,
             'Ac':227,   'Th':232.038,  'Pa':231.036,   'U':238.029,   'Np':237,   'Pu':244,   'Am':243,   'Cm':247,
             'Bk':247,   'Cf':251,  'Es':252,   'Fm':257, 'Md':258,  'No':259,  'Lr':262,  'Rf':261,
             'Db':262,  'Sg':266, 'Bh':264,  'Hs':277, 'Mt':268,  'Ds':281,  'Rg':272,  'Cn':285,
             'Uut':284, 'Fl':289, 'Uup':288, 'Lv':293, 'Uus':291, 'UUo':294}

class REANN(Calculator):

    implemented_properties = ['energy', 'forces']

    nolabel = True

    default_parameters = {'asap_cutoff': False}

    def __init__(self, atomtype,period=[1,1,1],device='cpu',nn = 'REANN_PES_DOUBLE.pt',**kwargs):
        Calculator.__init__(self, **kwargs)
        self.device= device
        self.atomtype = atomtype
        self.period = period
        pes=torch.jit.load(nn)
        pes.to(device).to(torch.double)
        pes.eval()
        pes=torch.jit.optimize_for_inference(pes)
        self.pes = pes
        
    
    # def mass_atom(self,atomtype):
    #     mass = []
    #     for atom in atomtype:
    #         mass.append(atom2mass[atom])
        
    #     return mass
        
    def transfer2eann(self,atoms):
        species = self.atoms.symbols
        #cell = self.atoms.cell
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
            #print(s,num_atom)


        return cell,cart,number_atom,spe
    
    def eann_all_str(self,number_atom,spe,atomtype):
        species = []
        atomtype_number = [i for i in range(len(atomtype))]
        spe_number = []
        mass = []
        for i in range(len(spe)):
            for j in range(len(atomtype)):
                if spe[i] == atomtype[j]:
                    spe_number.append(atomtype_number[j])
        number_sum = np.sum(np.array(number_atom))
        
        for i in range(number_sum):
            number_a1 = 0
            for i1 in range(len(number_atom)):
                number_a1 += number_atom[i1]
                if i - number_a1 < 0:
                    species.append(spe_number[i1])
                    mass.append(atom2mass[atomtype[atomtype_number.index(spe_number[i1])]])
                    break
        return species,mass

    
    def calculate(self,atoms=None, properties=['energy'],
                  system_changes=all_changes):
        #nn = self.nn
        period=self.period
        atomtype = self.atomtype
        device=self.device
        pes=self.pes
        Calculator.calculate(self, atoms, properties, system_changes)
        cell,cart,number_atom,spe = self.transfer2eann(self.atoms)
        species,mass = self.eann_all_str(number_atom,spe,atomtype)
        period_table=torch.tensor(period,dtype=torch.float,device=device)
        species=torch.from_numpy(np.array(species)).to(device)  # from numpy array to torch tensor
        cart=torch.from_numpy(np.array(cart)).to(device).to(torch.double)  # also float32/double
        tcell=torch.from_numpy(cell).to(device).to(torch.double)  # also float32/double
        mass=torch.from_numpy(np.array(mass)).to(device).to(torch.double)  # also float32/double
        #pes=torch.jit.load(nn)
        #pes.to(device).to(torch.double)
        #pes.eval()
        #print(period_table,cart,tcell,species,mass)
        energy=pes(period_table,cart,tcell,species,mass)[0]
        force=pes(period_table,cart,tcell,species,mass)[1]
        energy = float(energy.detach().numpy())
        force = force.detach().numpy()
        self.results['energy'] = energy
        self.results['forces'] = force

