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

    def forward(self,cart,numatoms,species,atom_index,shifts,create_graph=True):
        cart.requires_grad=True
        species=species.view(-1)
        density = self.density(cart,numatoms,species,atom_index,shifts)
        output=self.nnmod(density,species).view(numatoms.shape[0],-1)
        varene=torch.sum(output,dim=1)
        grad_outputs=torch.ones(numatoms.shape[0],device=cart.device)
        jab1=torch.autograd.grad(varene,cart,grad_outputs=grad_outputs,\
        create_graph=True,only_inputs=True,allow_unused=True)[0]
        output=self.nnmod1(density,species).view(numatoms.shape[0],-1)
        varene=torch.sum(output,dim=1)
        jab2=torch.autograd.grad(varene,cart,grad_outputs=grad_outputs,\
        create_graph=create_graph,only_inputs=True,allow_unused=True)[0]
        output=self.nnmod2(density,species).view(numatoms.shape[0],-1)
        varene=torch.sum(output,dim=1)
        jab1=jab1+cart
        polar=oe.contract("ijk,ikm -> ijm",jab1.permute(0,2,1).contiguous(),jab2,backend="torch")
        polar=polar+polar.permute(0,2,1)
        polar[:,0,0]=polar[:,0,0]+varene
        polar[:,1,1]=polar[:,1,1]+varene
        polar[:,2,2]=polar[:,2,2]+varene
        return polar.reshape(-1,9),
