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

    def forward(self,cart,numatoms,species,atom_index,shifts,dos_ene,create_graph=None):
        species=species.view(-1)
        density = self.density(cart,numatoms,species,atom_index,shifts)
        output=torch.sum(self.nnmod(density,species).view(numatoms.shape[0],-1),dim=1)+dos_ene
        input_dos=torch.cat((density.view(cart.shape[0],cart.shape[1],-1),output.view(cart.shape[0],1,1).expand(-1,cart.shape[1],-1)),dim=2)
        dos=torch.sum(self.nnmod1(input_dos.view(density.shape[0],-1),species).view(cart.shape[0],-1),dim=1).view(-1,1)
        return dos/(numatoms.view(-1,1).to(cart.dtype)),
