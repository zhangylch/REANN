import torch

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.detach().clone()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = (1.0 - self.decay) * param + self.decay * self.shadow[name]
    
    def apply_shadow(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.backup[name] = param.detach().clone()
                    param.copy_(self.shadow[name])
    
    def restore(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.copy_(self.backup[name])
            self.backup = {}
