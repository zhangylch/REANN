import torch 
from torch import nn

class Relu_like(nn.Module):
    def __init__(self,neuron1,neuron):
        super (Relu_like,self).__init__()
        self.alpha=nn.parameter.Parameter(torch.ones(1,neuron))
        self.beta=nn.parameter.Parameter(torch.ones(1,neuron)/float(neuron1))
        self.silu=nn.SiLU()

    def forward(self,x):
        return self.alpha*self.silu(x*self.beta)

class Tanh_like(nn.Module):
    def __init__(self,neuron1,neuron):
        super (Tanh_like,self).__init__()
        self.alpha=nn.parameter.Parameter(torch.ones(1,neuron)/torch.sqrt(torch.tensor([float(neuron1)])))
        self.beta=nn.parameter.Parameter(torch.ones(1,neuron)/float(neuron1))

    def forward(self,x):
        return self.alpha*x/torch.sqrt(1.0+torch.square(x*self.beta))
