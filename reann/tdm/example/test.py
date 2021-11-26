import numpy as np
import torch
from gpu_sel import *
# used for select a unoccupied GPU
gpu_sel()
# gpu/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# same as the atomtype in the file input_density
atomtype=['O', 'H', 'C', 'N']
#load the serilizable model
pes=torch.jit.load("EANN_TDM_FLOAT.pt")
# FLOAT: torch.float32; DOUBLE:torch.float32 for using float/float32 in inference
pes.to(device).to(torch.float32)
# set the eval mode
pes=torch.jit.optimize_for_inference(pes.eval())
# save the lattic parameters
cell=np.zeros((3,3),dtype=np.float32)
period_table=torch.tensor([0,0,0],dtype=torch.float32,device=device)   # same as the pbc in the periodic boundary condition
npoint=0
rmse=torch.zeros(2,dtype=torch.float32,device=device)
with open("/home/home/zyl/pytorch/2021_3_6/data/tdm/NMA-pi-pi/train/configuration",'r') as f1:
    while True:
        string=f1.readline()
        if not string: break
        string=f1.readline()
        cell[0]=np.array(list(map(float,string.split())))
        string=f1.readline()
        cell[1]=np.array(list(map(float,string.split())))
        string=f1.readline()
        cell[2]=np.array(list(map(float,string.split())))
        string=f1.readline()
        species=[]
        cart=[]
        abforce=[]
        mass=[]
        while True:
            string=f1.readline()
            if "abprop" in string: break
            tmp=string.split()
            tmp1=list(map(float,tmp[2:5]))
            mass.append(float(tmp[1]))
            cart.append(tmp1[0:3])
            species.append(atomtype.index(tmp[0]))
        abene=list(map(float,string.split()[1:4]))
        mass=torch.Tensor(mass).to(torch.float32).to(device)
        abene=torch.from_numpy(np.array([abene])).to(device)
        species=torch.from_numpy(np.array(species)).to(device)  # from numpy array to torch tensor
        cart=torch.from_numpy(np.array(cart)).to(device).to(torch.float32)  # also float32/double
        tcell=torch.from_numpy(cell).to(device).to(torch.float32)  # also float32/double
        dipole=pes(period_table,cart,tcell,species,mass)
        print("hello")
        print((dipole-abene).cpu().numpy())
        print(dipole.cpu().numpy())
        print(abene.cpu().numpy())
        rmse[0]+=torch.sum(torch.square(dipole-abene))
        npoint+=1
rmse=rmse.detach().cpu().numpy()
print(np.sqrt(rmse[0]/npoint))
