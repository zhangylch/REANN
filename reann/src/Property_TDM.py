import numpy as np
import torch 
import opt_einsum as oe
from torch.autograd.functional import jacobian
from src.MODEL import *
#============================calculate the energy===================================
class Property(torch.nn.Module):
    def __init__(self,density,nnmodlist):
        super(Property,self).__init__()
        self.density=density
        self.nnmod=nnmodlist[0]
        if len(nnmodlist) > 1:
            self.nnmod1=nnmodlist[1]
            self.nnmod2=nnmodlist[2]

    def forward(self,cart,numatoms,species,atom_index,shifts,create_graph=None):
        species=species.view(-1)
        density = self.density(cart,numatoms,species,atom_index,shifts)
        output=self.nnmod(density,species).view(numatoms.shape[0],-1,3)
        varene=torch.sum(output[:,:,2],dim=1)
        dipole=oe.contract("ijn,ijk -> nik",output[:,:,0:2],cart,backend="torch")
        tdm=dipole[0]+dipole[1]+varene[:,None]*torch.cross(dipole[0],dipole[1],dim=1)
        return tdm,

