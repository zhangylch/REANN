import numpy as np
import torch
from gpu_sel import *
# used for select a unoccupied GPU
gpu_sel()
# gpu/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# same as the atomtype in the file input_density
atomtype=["C","O","H"]
#load the serilizable model
pes=torch.jit.load("EANN_DM_DOUBLE.pt")
for name,m in pes.named_parameters():
    print(name)
    print(m)
# FLOAT: torch.float32; DOUBLE:torch.double for using float/double in inference
pes.to(device).to(torch.double)
# set the eval mode
pes.eval()
pes=torch.jit.optimize_for_inference(pes)
# save the lattic parameters
cell=np.zeros((3,3),dtype=np.float64)
period_table=torch.tensor([0,0,0],dtype=torch.double,device=device)   # same as the pbc in the periodic boundary condition
npoint=0
rmse=torch.zeros(1,dtype=torch.double,device=device)
with open("/home/home/zyl/pytorch/2021_9_5/data/dipole/ethanol/test/configuration",'r') as f1:
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
        mass=[]
        while True:
            string=f1.readline()
            if "abprop" in string: break
            tmp=string.split()
            tmp1=list(map(float,tmp[2:8]))
            cart.append(tmp1[0:3])
            mass.append(float(tmp[1]))
            species.append(atomtype.index(tmp[0]))
        label=list(map(float,string.split()[1:4]))
        label=torch.from_numpy(np.array([label])).to(device)
        species=torch.from_numpy(np.array(species)).to(device)  # from numpy array to torch tensor
        cart=torch.from_numpy(np.array(cart)).to(device).to(torch.double)  # also float32/double
        tcell=torch.from_numpy(cell).to(device).to(torch.double)  # also float32/double
        mass=torch.from_numpy(np.array(mass)).to(device).to(torch.double)  # also float32/double
        predict=pes(period_table,cart,tcell,species,mass)
        print(predict.cpu().numpy()[0:3])
        print(label.cpu().numpy()[0:3])
        rmse[0]+=torch.sum(torch.abs(predict-label))
        npoint+=1
rmse=rmse.detach().cpu().numpy()
print(rmse[0]/npoint/3)
