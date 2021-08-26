import torch

def normdensity(norbit,getdensity,data_train,device):
    maxdensity=torch.ones(norbit,device=device)*1e-12
    for data in data_train:
        abProp,cart,numatoms,species,atom_index,shifts=data
        species=species.view(-1)
        density = getdensity(cart,numatoms,species,atom_index,shifts).detach()
        maxdensity=torch.maximum(maxdensity,torch.max(density,dim=0)[0])
        # empty the cache in gpu
        torch.cuda.empty_cache()
    with torch.no_grad():
        getdensity.params.copy_(getdensity.params/torch.sqrt(maxdensity.view(1,-1)))
