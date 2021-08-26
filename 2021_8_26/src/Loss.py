import torch 
import torch.nn as nn

class Loss(nn.Module):
   def __init__(self):
      super(Loss, self).__init__()
      self.loss_fn=nn.MSELoss(reduction="sum")

   def forward(self,var,ab): 
      return  torch.cat([self.loss_fn(ivar,iab).view(-1) for ivar, iab in zip(var,ab)])
