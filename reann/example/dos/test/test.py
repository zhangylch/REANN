# This is an example script to show how to obtain the energy and force by invoking the potential saved by the training .
# Typically, you can read the structure,mass, lattice parameters(cell) and give the correct periodic boundary condition (pbc) and t    he index of each atom. All the information are required to store in the tensor of torch. Then, you just pass these information to t    he calss "pes" that will output the energy and force.

import numpy as np
import torch
from gpu_sel import *
from write_format import *
# used for select a unoccupied GPU
gpu_sel()
# gpu/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# same as the atomtype in the file input_density
atomtype=["Si"]
#load the serilizable model
pes=torch.jit.load("../REANN_PES_DOUBLE.pt")
# FLOAT: torch.float32; DOUBLE:torch.double for using float/double in inference
pes.to(device).to(torch.double)
# set the eval mode
pes.eval()
pes=torch.jit.optimize_for_inference(pes)
# save the lattic parameters
cell=np.zeros((3,3),dtype=np.float64)
period_table=torch.tensor([1,1,1],dtype=torch.double,device=device)   # same as the pbc in the periodic boundary condition
npoint=torch.zeros(2,dtype=torch.long,device=device)
with open("/data/home/scv2201/run/zyl/data/dos/train/configuration",'r') as f1:
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
        element=[]
        while True:
            string=f1.readline()
            if "abprop" in string: break
            tmp=string.split()
            element.append(tmp[0])
            tmp1=list(map(float,tmp[2:8]))
            cart.append(tmp1[0:3])
            abforce.append(tmp1[3:6])
            mass.append(float(tmp[1]))
            species.append(atomtype.index(tmp[0]))
        abene=float(string.split()[1])
        abene=torch.from_numpy(np.array([abene])).to(device)
        species=torch.from_numpy(np.array(species)).to(device)  # from numpy array to torch tensor
        cart=torch.from_numpy(np.array(cart)).to(device).to(torch.double)  # also float32/double
        mass=torch.from_numpy(np.array(mass)).to(device).to(torch.double)  # also float32/double
        abforce=torch.from_numpy(np.array(abforce)).to(device).to(torch.double)  # also float32/double
        tcell=torch.from_numpy(cell).to(device).to(torch.double)  # also float32/double
        energy,force=pes(period_table,cart,tcell,species,mass)
        energy=energy.detach()
        force=force.detach()
        rmse[0]+=torch.square(energy-abene)[0]
        npoint[0]+=1
        rmse[1]+=torch.sum(torch.square(force-abforce))
        npoint[1]+=force.view(-1).shape[0]
        #if abs(energy.cpu().numpy()-abene.cpu().numpy()) < 0.8 and torch.sqrt(torch.mean(torch.square(force-abforce))) < 0.5 :
        #   write_format(f2,npoint,period_table.cpu().numpy(),element,mass.cpu().numpy(),cart.cpu().numpy(),abene.cpu().numpy(),force=abforce.cpu().numpy(),cell=tcell.cpu().numpy())
        #   npoint+=1
        #else:
        #   print((energy-abene).cpu().numpy()[0])
f2.close()
print(torch.sqrt(rmse/npoint))
