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
        output=self.nnmod(density,species).view(-1)
        tmp_index=torch.arange(numatoms.shape[0],device=cart.device)*cart.shape[1]
        self_mol_index=tmp_index.view(-1,1).expand(-1,atom_index.shape[2]).reshape(1,-1)
        cart_=cart.flatten(0,1)
        totnatom=cart_.shape[0]
        padding_mask=torch.nonzero((shifts.view(-1,3)>-1e10).all(1)).view(-1)
        # get the index for the distance less than cutoff (the dimension is reduntant)
        atom_index12=(atom_index.view(2,-1)+self_mol_index)[:,padding_mask]
        selected_cart = cart_.index_select(0, atom_index12.view(-1)).view(2, -1, 3)
        shift_values=shifts.view(-1,3).index_select(0,padding_mask)
        dist_vec = selected_cart[0] - selected_cart[1] + shift_values
        tot_vec = torch.zeros((species.shape[0],3),dtype=cart.dtype,device=cart.device)
        tot_vec = torch.index_add(tot_vec,0,atom_index12[0],dist_vec)
        dipole=torch.sum(oe.contract("i,ij -> ij",output,tot_vec,backend="torch").view(cart.shape[0],-1,3),dim=1)
        return dipole,

