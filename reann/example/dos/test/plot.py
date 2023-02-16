# This is an example script to show how to obtain the energy and force by invoking the potential saved by the training .
# Typically, you can read the structure,mass, lattice parameters(cell) and give the correct periodic boundary condition (pbc) and t    he index of each atom. All the information are required to store in the tensor of torch. Then, you just pass these information to t    he calss "pes" that will output the energy and force.
import re
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
ldos=np.load("ldos.npy")
xdos=np.load("xdos.npy")
x_dos=torch.from_numpy(xdos).to(device).to(torch.double)
pattern=re.compile(r"(?<={}=)\"(.+?)\"".format("Lattice"))
num=0
npoint=0
with open("/data/home/scv2201/run/zyl/data/dos/training_dataset.xyz",'r') as f1:
    while True:
        string=f1.readline()
        if not string or num>10: break
        numatom=int(string)
        string=f1.readline()
        tmp=re.findall(pattern,string)
        cell=np.array(list(map(float,tmp[0].split()))).reshape(3,3)
        element=[]
        mass=[]
        cart=[]
        species=[]
        for i in range(numatom):
            string=f1.readline()
            tmp=string.split()
            mass.append(28.085)
            cart.append(list(map(float,tmp[1:4])))
            species.append(atomtype.index(tmp[0]))
        species=torch.from_numpy(np.array(species)).to(device)  # from numpy array to torch tensor
        cart=torch.from_numpy(np.array(cart)).to(device).to(torch.double)  # also float32/double
        mass=torch.from_numpy(np.array(mass)).to(device).to(torch.double)  # also float32/double
        tcell=torch.from_numpy(cell).to(device).to(torch.double)  # also float32/double
        dos=pes(period_table,cart,tcell,species,mass,x_dos).cpu().detach().numpy()
        filename="file_"+str(num)
        with open(filename,'w') as f2:
            for i in range(x_dos.shape[0]):
                f2.write("{}  {}  {} \n".format(x_dos[i],dos[i],ldos[num][i]))
        num+=1
