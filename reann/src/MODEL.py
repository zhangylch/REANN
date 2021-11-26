import torch
from collections import OrderedDict
from torch import nn
from torch.nn import Linear,Dropout,BatchNorm1d,Sequential,LayerNorm
from torch.nn import Softplus,GELU,Tanh,SiLU
from torch.nn.init import xavier_uniform_,zeros_,constant_
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, nl, dropout_p, actfun, table_norm=True):
        super(ResBlock, self).__init__()
        # activation function used for the nn module
        nhid=len(nl)-1
        sumdrop=np.sum(dropout_p)
        modules=[]
        for i in range(1,nhid):
            modules.append(actfun(nl[i-1],nl[i]))
            if table_norm: modules.append(LayerNorm(nl[i]))
            if sumdrop>=0.0001: modules.append(Dropout(p=dropout_p[i-1]))
            #bias = not(i==nhid-1)
            linear=Linear(nl[i],nl[i+1])
            if i==nhid-1: 
                zeros_(linear.weight)
            else:
                xavier_uniform_(linear.weight)
            zeros_(linear.bias)
            modules.append(linear)
        self.resblock=Sequential(*modules)

    def forward(self, x):
        return self.resblock(x) + x

#==================for get the atomic energy=======================================
class NNMod(torch.nn.Module):
   def __init__(self,maxnumtype,outputneuron,atomtype,nblock,nl,dropout_p,actfun,initpot=0.0,table_norm=True):
      """
      maxnumtype: is the maximal element
      nl: is the neural network structure;
      outputneuron: the number of output neuron of neural network
      atomtype: elements in all systems
      """
      super(NNMod,self).__init__()
      self.register_buffer("initpot",torch.Tensor([initpot]))
      # create the structure of the nn     
      self.outputneuron=outputneuron
      elemental_nets=OrderedDict()
      sumdrop=np.sum(dropout_p)
      with torch.no_grad():
          nl.append(nl[1])
          nhid=len(nl)-1
          for ele in atomtype:
              modules=[]
              linear=Linear(nl[0],nl[1])
              xavier_uniform_(linear.weight)
              modules.append(linear)
              for iblock in range(nblock):
                  modules.append( * [ResBlock(nl,dropout_p,actfun,table_norm=table_norm)])
              modules.append(actfun(nl[nhid-1],nl[nhid]))
              linear=Linear(nl[nhid],self.outputneuron)
              zeros_(linear.weight)
              if abs(initpot)>1e-6: zeros_(linear.bias)
              modules.append(linear)
              elemental_nets[ele] = Sequential(*modules)
      self.elemental_nets=nn.ModuleDict(elemental_nets)

#   @pysnooper.snoop('out',depth=2)   for debug
   def forward(self,density,species):    
      # elements: dtype: LongTensor store the index of elements of each center atom
      output = torch.zeros((density.shape[0],self.outputneuron), dtype=density.dtype, device=density.device)
      for itype, (_, m) in enumerate(self.elemental_nets.items()):
          mask = (species == itype)
          ele_index = torch.nonzero(mask).view(-1)
          if ele_index.shape[0] > 0:
              ele_den = density[ele_index]
              output[ele_index] = m(ele_den)
      return output
