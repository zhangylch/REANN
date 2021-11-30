import torch
import numpy as np
import os
from inference.density import *
from src.MODEL import *
from inference.get_neigh import *

class PES(torch.nn.Module):
    def __init__(self,nlinked=1):
        super(PES, self).__init__()
        #========================set the global variable for using the exec=================
        global nblock, nl, dropout_p, table_norm, activate, norbit
        global oc_loop,oc_nblock, oc_nl, oc_dropout_p, oc_table_norm, oc_activate
        global nwave, neigh_atoms, cutoff, nipsin, atomtype
        # global parameters for input_nn
        nblock = 1                    # nblock>=2  resduial NN block will be employed nblock=1: simple feedforward nn
        nl=[128,128]                # NN structure
        dropout_p=[0.0,0.0]       # dropout probability for each hidden layer
        activate = 'Relu_like'
        table_norm= True
        oc_loop = 1
        oc_nl = [128,128]          # neural network architecture   
        oc_nblock = 1
        oc_dropout_p=[0.0,0.0]
        oc_activate = 'Relu_like'
        #========================queue_size sequence for laod data into gpu
        oc_table_norm=True
        norbit=None
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
        nipsin=2
        cutoff=4.5
        nwave=7
        neigh_atoms=150
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

        if activate=='Tanh_like':
            from src.activate import Tanh_like as actfun
        else:
            from src.activate import Relu_like as actfun

        if oc_activate=='Tanh_like':
            from src.activate import Tanh_like as oc_actfun
        else:
            from src.activate import Relu_like as oc_actfun
       
        dropout_p=np.array(dropout_p)
        oc_dropout_p=np.array(oc_dropout_p)
        maxnumtype=len(atomtype)
        #========================use for read rs/inta or generate rs/inta================
        self.outputneuron=outputneuron
        if 'rs' in globals().keys():
           rs=torch.from_numpy(np.array(rs))
           inta=torch.from_numpy(np.array(inta))
           nwave=rs.shape[1]
        else:
           inta=torch.ones((maxnumtype,nwave))
           rs=torch.stack([torch.linspace(0,cutoff,nwave) for itype in range(maxnumtype)],dim=0)
        #======================for orbital================================
        nipsin+=1
        if not norbit:
            norbit=int(nwave*(nwave+1)/2*nipsin)
        #========================nn structure========================
        nl.insert(0,int(norbit))
        oc_nl.insert(0,int(norbit))
        #================read the periodic boundary condition, element and mass=========
        self.cutoff=cutoff
        ocmod_list=[]
        for ioc_loop in range(oc_loop):
            ocmod_list.append(NNMod(maxnumtype,nwave,atomtype,oc_nblock,list(oc_nl),\
            oc_dropout_p,oc_actfun,table_norm=oc_table_norm))
        self.density=GetDensity(rs,inta,cutoff,nipsin,norbit,ocmod_list)
        self.nnmod=NNMod(maxnumtype,outputneuron,atomtype,nblock,list(nl),dropout_p,actfun,table_norm=table_norm)
        self.nnmod1=NNMod(maxnumtype,outputneuron,atomtype,nblock,list(nl),dropout_p,actfun,table_norm=table_norm)
        self.nnmod2=NNMod(maxnumtype,outputneuron,atomtype,nblock,list(nl),dropout_p,actfun,table_norm=table_norm)
        #================================================nn module==================================================
        self.neigh_list=Neigh_List(cutoff,nlinked)
     
    def forward(self,period_table,cart,cell,species,mass):
        cart=cart.detach().clone()
        neigh_list, shifts=self.neigh_list(period_table,cart,cell,mass)
        cart.requires_grad_(True)
        density=self.density(cart,neigh_list,shifts,species)
        output = self.nnmod(density,species)
        varene=torch.sum(output)
        jab1=torch.autograd.grad([varene,],[cart,],create_graph=True)[0]
        output = self.nnmod1(density,species)
        varene=torch.sum(output)
        jab2=torch.autograd.grad([varene,],[cart,])[0]
        output=self.nnmod2(density,species)
        varene=torch.sum(output)
        if (jab1 is not None) and (jab2 is not None):
            jab1=jab1+cart
            polar=torch.einsum("ij,ik -> jk",jab1,jab2)
            polar=polar+polar.permute(1,0)
            polar[0,0]=polar[0,0]+varene
            polar[1,1]=polar[1,1]+varene
            polar[2,2]=polar[2,2]+varene
            return polar.detach()

