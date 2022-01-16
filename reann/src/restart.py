import torch
class Restart():
    def __init__(self,optimizer):
        self.optim=optimizer

    def __call__(self,model,checkfile):
        self.forward(model,checkfile)
    
    def forward(self,model,checkfile):
        if torch.cuda.is_available():
            device="cuda"
        else:
            device="cpu"
        checkpoint = torch.load(checkfile,map_location=torch.device(device))
        model.load_state_dict(checkpoint['reannparam'])
        self.optim.load_state_dict(checkpoint['optimizer'])
