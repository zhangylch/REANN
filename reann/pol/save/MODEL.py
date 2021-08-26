import torch
from collections import OrderedDict
from torch import nn
from torch.nn import Linear,Dropout,BatchNorm1d,Sequential
from torch.nn import Softplus,GELU,Tanh
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, nl, actfun, dropout_p, table_bn=True):
        super(ResBlock, self).__init__()
        # activation function used for the nn module
        nhid=len(nl)-1
        sumdrop=np.sum(dropout_p)
        modules=[]
        for i in range(1,nhid):
            if table_bn: modules.append(BatchNorm1d(nl[i]))
            modules.append(actfun)
            if sumdrop>=0.0001: modules.append(Dropout(p=dropout_p[i-1]))
            linear=Linear(nl[i],nl[i+1],bias=((not(i==nhid-1)) and (not table_bn)))
            torch.nn.init.xavier_normal_(linear.weight, \
            gain=nn.init.calculate_gain('relu'))
            modules.append(linear)
        self.resblock=Sequential(*modules)

    def forward(self, x):
        out = self.resblock(x)
        return out + x


class OCMod(torch.nn.Module):
   def __init__(self,maxnumtype,outputneuron,atomtype,nblock,nl,activate,dropout_p,initpot=None,enable_force=True,table_bn=True):
      """
      maxnumtype: is the maximal element
      nl: is the neural network structure;
      actfun: activation function 
      outputneuron: the number of output neuron of neural network
      atomtype: elements in all systems
      initpot: initpot used for initialize the biases of the last layers
      """
      super(OCMod,self).__init__()
      # activation function used for the nn module
      if activate=='Softplus':
         actfun=Softplus()
      elif activate=='Gelu':
         actfun=GELU()
      elif activate=='tanh':
         actfun=Tanh()
      self.enable_force=enable_force
      # create the structure of the nn     
      sumdrop=np.sum(dropout_p)
      self.outputneuron=outputneuron
      self.elemental_nets=OrderedDict()
      with torch.no_grad():
          if nblock==1:
              nl.append(outputneuron)
              nhid=len(nl)-2
              for ele in atomtype:
                  modules=[]
                  for i in range(nhid):
                      linear=Linear(nl[i],nl[i+1],bias=(not table_bn))
                      torch.nn.init.xavier_normal_(linear.weight, \
                      gain=nn.init.calculate_gain('relu'))
                      #xavier initialization (normal distribution)
                      modules.append(linear)
                      if table_bn: modules.append(BatchNorm1d(nl[i+1]))
                      modules.append(actfun)
                      if sumdrop>0.0001: modules.append(Dropout(p=dropout_p[i]))
                  linear=Linear(nl[nhid],nl[nhid+1])
                  torch.nn.init.xavier_normal_(linear.weight, gain=nn.init.calculate_gain('relu'))
                  if not not initpot: linear.bias[:]=initpot
                  modules.append(linear)
                  self.elemental_nets[ele] = Sequential(*modules)
          else:
              nl.append(nl[1])
              nhid=len(nl)-1
              for ele in atomtype:
                  modules=[]
                  linear=Linear(nl[0],nl[1],bias=(not table_bn))
                  torch.nn.init.xavier_normal_(linear.weight, \
                  gain=nn.init.calculate_gain('relu'))
                  # xavier initialization (normal distribution)
                  modules.append(linear)
                  for iblock in range(nblock):
                      modules.append( * [ResBlock(nl,actfun,dropout_p,table_bn=table_bn)])
                  modules.append(actfun)
                  linear=Linear(nl[nhid],self.outputneuron)
                  torch.nn.init.xavier_normal_(linear.weight, gain=nn.init.calculate_gain('relu'))
                  if not not initpot: linear.bias[:]=initpot
                  modules.append(linear)
                  self.elemental_nets[ele] = Sequential(*modules)
      self.elemental_nets=nn.ModuleDict(self.elemental_nets)
      self.outputneuron=outputneuron

   def forward(self,oc_density,species):    
      # elements: dtype: LongTensor store the index of elements of each center atom
      output=torch.empty((oc_density.shape[0],self.outputneuron), dtype=oc_density.dtype, device=oc_density.device)
      for itype, (_, m) in enumerate(self.elemental_nets.items()):
          mask = (species == itype)
          ele_index = torch.nonzero(mask).view(-1)
          if ele_index.shape[0] > 0:
              ele_den = oc_density[ele_index]
              output[ele_index]=m(ele_den)
      return output

class NNMod(torch.nn.Module):
   def __init__(self,maxnumtype,outputneuron,atomtype,nblock,nl,activate,dropout_p,initpot=None,enable_force=True,table_bn=True):
      """
      maxnumtype: is the maximal element
      nl: is the neural network structure;
      actfun: activation function 
      outputneuron: the number of output neuron of neural network
      atomtype: elements in all systems
      initpot: initpot used for initialize the biases of the last layers
      """
      super(NNMod,self).__init__()
      # activation function used for the nn module
      if activate=='Softplus':
         actfun=Softplus()
      elif activate=='Gelu':
         actfun=GELU()
      elif activate=='tanh':
         actfun=Tanh()
      self.enable_force=enable_force
      # create the structure of the nn     
      sumdrop=np.sum(dropout_p)
      self.outputneuron=outputneuron
      self.elemental_nets=OrderedDict()
      with torch.no_grad():
          if nblock==1:
              nl.append(outputneuron)
              nhid=len(nl)-2
              for ele in atomtype:
                  modules=[]
                  for i in range(nhid):
                      linear=Linear(nl[i],nl[i+1],bias=(not table_bn))
                      torch.nn.init.xavier_normal_(linear.weight, \
                      gain=nn.init.calculate_gain('relu'))
                      #xavier initialization (normal distribution)
                      modules.append(linear)
                      if table_bn: modules.append(BatchNorm1d(nl[i+1]))
                      modules.append(actfun)
                      if sumdrop>0.0001: modules.append(Dropout(p=dropout_p[i]))
                  linear=Linear(nl[nhid],nl[nhid+1])
                  torch.nn.init.xavier_normal_(linear.weight, gain=nn.init.calculate_gain('relu'))
                  if not not initpot: linear.bias[:]=initpot
                  modules.append(linear)
                  self.elemental_nets[ele] = Sequential(*modules)
          else:
              nl.append(nl[1])
              nhid=len(nl)-1
              for ele in atomtype:
                  modules=[]
                  linear=Linear(nl[0],nl[1],bias=(not table_bn))
                  torch.nn.init.xavier_normal_(linear.weight, \
                  gain=nn.init.calculate_gain('relu'))
                  # xavier initialization (normal distribution)
                  modules.append(linear)
                  for iblock in range(nblock):
                      modules.append( * [ResBlock(nl,actfun,dropout_p,table_bn=table_bn)])
                  modules.append(actfun)
                  linear=Linear(nl[nhid],self.outputneuron)
                  torch.nn.init.xavier_normal_(linear.weight, gain=nn.init.calculate_gain('relu'))
                  if not not initpot: linear.bias[:]=initpot
                  modules.append(linear)
                  self.elemental_nets[ele] = Sequential(*modules)
      self.elemental_nets=nn.ModuleDict(self.elemental_nets)
      self.outputneuron=outputneuron

   def forward(self,density, species):    
      # elements: dtype: LongTensor store the index of elements of each center atom
      output=torch.empty((density.shape[0],self.outputneuron), dtype=density.dtype, device=density.device)
      for itype, (_, m) in enumerate(self.elemental_nets.items()):
          mask = (species == itype)
          ele_index = torch.nonzero(mask).view(-1)
          if ele_index.shape[0] > 0:
              ele_den = density[ele_index]
              output[ele_index]=m(ele_den)
      return output
