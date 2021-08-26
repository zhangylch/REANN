import torch
import numpy as np
import os
from pes.density import *
from pes.MODEL import *
from pes.get_neigh import *

class PES(torch.nn.Module):
    def __init__(self,nlinked=1):
        super(PES, self).__init__()
        #========================set the global variable for using the exec=================
        global nblock, nl, dropout_p, activate, table_bn
        global oc_loop,oc_nblock, oc_nl, oc_dropout_p, oc_activate, oc_table_bn
        global nwave, neigh_atoms, cutoff, nipsin, atomtype
        # global parameters for input_nn
        nblock = 1                    # nblock>=2  resduial NN block will be employed nblock=1: simple feedforward nn
        nl=[256,128,64,32]                # NN structure
        dropout_p=[0.0,0.0,0.0]       # dropout probability for each hidden layer
        activate = 'Softplus'         # activate function: "Gelu", "tanh", "Softplus" are supported
        table_bn= False
        oc_loop = 1
        oc_nl = [64,32]          # neural network architecture   
        oc_nblock = 1
        oc_dropout_p=[0.0,0.0,0.0,0.0]
        #=====================act fun===========================
        oc_activate = 'Softplus'          # default "Softplus", optional "Gelu", "tanh"
        #========================queue_size sequence for laod data into gpu
        oc_table_bn=False
        
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
        if start_table<=2:
           outputneuron=1
        elif start_table==3:
           outputneuron=3
        elif start_table==4:
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
        maxnumtype=len(atomtype)
        #========================use for read rs/inta or generate rs/inta================
        self.outputneuron=outputneuron
        self.table=start_table
        if 'rs' in globals().keys():
           rs=torch.from_numpy(np.array(rs))
           inta=torch.from_numpy(np.array(inta))
           nwave=rs.shape[1]
        else:
           inta=torch.ones((maxnumtype,nwave))
           rs=torch.stack([torch.linspace(0,cutoff,nwave) for itype in range(maxnumtype)],dim=0)
        #======================for orbital================================
        ipsin=len(nipsin)
        norbit=nwave*ipsin
        #========================nn structure========================
        nl.insert(0,int(norbit))
        oc_nl.insert(0,int(norbit))
        #================read the periodic boundary condition, element and mass=========
        self.cutoff=cutoff
        ocmod_list=[]
        for ioc_loop in range(oc_loop):
            ocmod_list.append(OCMod(maxnumtype,norbit,atomtype,oc_nblock,list(oc_nl),\
            oc_activate,oc_dropout_p,table_bn=oc_table_bn))
        self.density=GetDensity(rs,inta,cutoff,nipsin,maxnumtype,nwave,ocmod_list)
        self.nnmod=NNMod(maxnumtype,outputneuron,atomtype,nblock,list(nl),activate,dropout_p,table_bn=table_bn)
        if start_table==4:
            self.nnmod1=NNMod(maxnumtype,outputneuron,atomtype,nblock,list(nl),activate,dropout_p,table_bn=table_bn)
            self.nnmod2=NNMod(maxnumtype,outputneuron,atomtype,nblock,list(nl),activate,dropout_p,table_bn=table_bn)
        #==============================nn module=================================
        self.neigh_list=Neigh_List(cutoff,nlinked)
     
    def forward(self,period_table,cart,cell,species):
        cart=cart.detach().clone()
        neigh_list, shifts=self.neigh_list(period_table,cart,cell)
        if self.table==0:
            return self.Energy(cart,neigh_list,shifts,species)
        elif self.table==1:
            return self.Force(cart,neigh_list,shifts,species)
        elif self.table==2:
            return self.DM(cart,neigh_list,shifts,species)
        elif self.table==3:
            return self.TDM(cart,neigh_list,shifts,species)
        elif self.table==4:
            return self.POL(cart,neigh_list,shifts,species)

    def Energy(self,cart,neigh_list,shifts,species):
        density=self.density(cart,neigh_list,shifts,species)
        output = self.nnmod(density,species)
        varene = torch.sum(output)
        return varene,

    def Force(self,cart,neigh_list,shifts,species):
        cart.requires_grad_(True)
        density=self.density(cart,neigh_list,shifts,species)
        output = self.nnmod(density,species)
        varene = torch.sum(output)
        grad = torch.autograd.grad([varene,],[cart,])[0]
        if grad is not None:
            return torch.cat([varene,-grad.view(-1)]),

    def DM(self,cart,neigh_list,shifts,species):
        density=self.density(cart,neigh_list,shifts,species)
        output = self.nnmod(density,species)
        dipole=torch.einsum("i,ij -> j",output,cart)
        return dipole,

    def TDM(self,cart,neigh_list,shifts,species):
        density=self.density(cart,neigh_list,shifts,species)
        output = self.nnmod(density,species).view(-1,self.outputneuron)
        dipole=torch.einsum("ij,ik -> jk",output[:,0:2],cart)
        varene=torch.sum(output[:,2])
        tdm=dipole[0]+dipole[1]+varene*torch.cross(dipole[0],dipole[1])
        return tdm,
 
    def POL(self,cart,neigh_list,shifts,species):
        cart.requires_grad_(True)
        density=self.density(cart,neigh_list,shifts,species)
        output = self.nnmod(density,species)
        varene=torch.sum(output)
        jab1=torch.autograd.grad([varene,],[cart,])[0]
        output = self.nnmod1(density,species)
        varene=torch.sum(output)
        jab2=torch.autograd.grad([varene,],[cart,])[0]
        output=self.nnmod2(density,species)
        varene=torch.sum(output)
        if (jab1 is not None) and (jab2 is not None):
            jab1=jab1+cart
            polar=torch.einsum("ij,ij -> ii",jab1,jab2)
            polar=polar+polar.permute(1,0)
            polar[0,0]=polar[0,0]+varene
            polar[1,1]=polar[1,1]+varene
            polar[2,2]=polar[2,2]+varene
            return polar,
