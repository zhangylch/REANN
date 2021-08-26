import numpy as np
import torch
from gpu_sel import *
# used for select a unoccupied GPU
gpu_sel()
# gpu/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# same as the atomtype in the file input_density
atomtype=["C","H"]
#load the serilizable model
pes=torch.jit.load("EANN_PES_DOUBLE.pt")
# FLOAT: torch.float32; DOUBLE:torch.double for using float/double in inference
pes.to(device).to(torch.double)
# set the eval mode
pes.eval()
# save the lattic parameters
cell=np.zeros((3,3),dtype=np.float64)
period_table=torch.tensor([0,0,0],dtype=torch.double,device=device)   # same as the pbc in the periodic boundary condition
npoint=0
npoint1=0
rmse1=torch.zeros(2,dtype=torch.double,device=device)
rmse=torch.zeros(2,dtype=torch.double,device=device)
with open("/share/home/bjiangch/group-zyl/zyl/pytorch/2021_05_19/addpoint/10-1/save/1e3/1000/2/test/configuration",'r') as f1:
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
        while True:
            string=f1.readline()
            if "abprop" in string: break
            tmp=string.split()
            tmp1=list(map(float,tmp[2:8]))
            cart.append(tmp1[0:3])
            abforce.append(tmp1[3:6])
            species.append(atomtype.index(tmp[0]))
        abene=float(string.split()[1])
        abene=torch.from_numpy(np.array([abene])).to(device)
        species=torch.from_numpy(np.array(species)).to(device)  # from numpy array to torch tensor
        cart=torch.from_numpy(np.array(cart)).to(device).to(torch.double)  # also float32/double
        abforce=torch.from_numpy(np.array(abforce)).to(device).to(torch.double)  # also float32/double
        tcell=torch.from_numpy(cell).to(device).to(torch.double)  # also float32/double
        energy,force=pes(period_table,cart,tcell,species)
        energy=energy.detach()
        force=force.detach()
        print((energy-abene).cpu().numpy()[0])
        if torch.abs(energy-abene)<3.0:
           rmse[0]+=torch.sum(torch.square(energy-abene))
           rmse[1]+=torch.sum(torch.square(force-abforce))
           rmse1[0]+=torch.sum(torch.square(energy-abene))
           rmse1[1]+=torch.sum(torch.square(force-abforce))
           npoint1+=1
           npoint+=1
        else:
           rmse1[0]+=torch.sum(torch.square(energy-abene))
           rmse1[1]+=torch.sum(torch.square(force-abforce))
           print('hello1',npoint1)
           npoint1+=1
rmse=rmse.detach().cpu().numpy()
rmse1=rmse1.detach().cpu().numpy()
print("hello")
print(np.sqrt(rmse[0]/npoint))
print(np.sqrt(rmse[1]/npoint/15))
print(np.sqrt(rmse1[0]/npoint1))
print(np.sqrt(rmse1[1]/npoint1/15))
print(npoint,npoint1)
