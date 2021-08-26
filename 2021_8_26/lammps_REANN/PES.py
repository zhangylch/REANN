import torch
import numpy as np
import os
from lammps_REANN.density import *
from src.MODEL import *

class PES(torch.nn.Module):
    def __init__(self,nlinked=1):
        super(PES, self).__init__()
        #========================set the global variable for using the exec=================
        global nblock, nl, dropout_p, activate, table_norm
        global oc_loop,oc_nblock, oc_nl, oc_dropout_p, oc_activate,oc_table_norm
        global nwave, cutoff, nipsin, atomtype
        # global parameters for input_nn
        nblock = 1                    # nblock>=2  resduial NN block will be employed nblock=1: simple feedforward nn
        nl=[256,128,64,32]                # NN structure
        dropout_p=[0.0,0.0,0.0]       # dropout probability for each hidden layer
        activate = 'Softplus'         # activate function: "Gelu", "tanh", "Softplus" are supported
        table_norm= False
        oc_loop = 0
        oc_nl = [64,32]          # neural network architecture   
        oc_nblock = 1
        oc_dropout_p=[0.0,0.0,0.0,0.0]
        #=====================act fun===========================
        oc_activate = 'Softplus'          # default "Softplus", optional "Gelu", "tanh"
        #========================queue_size sequence for laod data into gpu
        oc_table_norm=False
        
        #======================read input_nn==================================
        with open('para/input_nn','r') as f1:
           while True:
              tmp=f1.readline()
              if not tmp: break
              string=tmp.strip()
              if len(string)!=0:
                  if string[0]=='#':
                     pass
                  else:
                     m=string.split('#')
                     exec(m[0],globals())
        # define the outputneuron of NN
        outputneuron=1
        #======================read input_nn=============================================
        nipsin=[0,1,2]
        cutoff=4.0
        nwave=12
        with open('para/input_density','r') as f1:
           while True:
              tmp=f1.readline()
              if not tmp: break
              string=tmp.strip()
              if len(string)!=0:
                  if string[0]=='#':
                     pass
                  else:
                     m=string.split('#')
                     exec(m[0],globals())
        
        dropout_p=np.array(dropout_p)
        oc_dropout_p=np.array(oc_dropout_p)
        maxnumtype=len(atomtype)
        #========================use for read rs/inta or generate rs/inta================
        if 'rs' in globals().keys():
           rs=torch.from_numpy(np.array(rs))
           inta=torch.from_numpy(np.array(inta))
           nwave=rs.shape[1]
        else:
           inta=torch.ones((maxnumtype,nwave))
           rs=torch.stack([torch.linspace(0,cutoff,nwave) for itype in range(maxnumtype)],dim=0)
        #======================for orbital================================
        ipsin=len(nipsin)
        nipsin=torch.Tensor(nipsin)
        norbit=nwave*ipsin
        #========================nn structure========================
        nl.insert(0,int(norbit))
        oc_nl.insert(0,int(norbit))
        #================read the periodic boundary condition, element and mass=========
        self.cutoff=cutoff
        ocmod_list=[]
        for ioc_loop in range(oc_loop):
            ocmod_list.append(NNMod(maxnumtype,norbit,atomtype,oc_nblock,list(oc_nl),\
            oc_activate,oc_dropout_p,table_norm=oc_table_norm))
        self.density=GetDensity(rs,inta,cutoff,nipsin,ocmod_list)
        self.nnmod=NNMod(maxnumtype,outputneuron,atomtype,nblock,list(nl),activate,dropout_p,table_norm=table_norm)
        #================================================nn module==================================================
     
    def forward(self,cart,atom_index,local_species,neigh_list):
        atom_index = atom_index.t().contiguous()
        cart.requires_grad_(True)
        density=self.density(cart,atom_index,local_species,neigh_list)
        output = self.nnmod(density,local_species)+self.nnmod.initpot
        varene = torch.sum(output)
        grad = torch.autograd.grad([varene,],[cart,])[0]
        if grad is not None:
            return varene.detach(),-grad.detach().view(-1),output.detach()

