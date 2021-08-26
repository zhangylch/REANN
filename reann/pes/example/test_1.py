import numpy as np
import torch
from gpu_sel import *
import sys
import os
error=float(sys.argv[1])
# used for select a unoccupied GPU
gpu_sel()
# gpu/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# same as the atomtype in the file input_density
atomtype=["C","H"]
mass=[12.001, 1.008]
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
f2=open("badpoint","w")
f3=open("configuration_1","w")
with open("configuration",'r') as f1:
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
        numatom=species.shape[0]
        if torch.abs(energy-abene)<error:
           rmse[0]+=torch.sum(torch.square(energy-abene))
           rmse[1]+=torch.sum(torch.square(force-abforce))
           rmse1[0]+=torch.sum(torch.square(energy-abene))
           rmse1[1]+=torch.sum(torch.square(force-abforce))
           npoint+=1
           npoint1+=1
           if torch.abs(energy-abene)>0.1:
               cart=cart.cpu().detach().numpy()
               species=species.cpu().numpy()
               abforce=abforce.cpu().numpy()
               abene=abene.cpu().numpy()
               f3.write("point \n")
               f3.write("100.0  0.0   0.0 \n")
               f3.write("  0.0 100.0  0.0 \n")
               f3.write("  0.0  0.0  100.0 \n")
               f3.write("pbc  0  0  0 \n")
               for natom in range(numatom):
                   f3.write("{}  {}  ".format(atomtype[species[natom]],mass[species[natom]]))
                   for idim in range(3):
                       f3.write("{}  ".format(cart[natom,idim]))
                   for idim in range(3):
                       f3.write("{}  ".format(abforce[natom,idim]))
                   f3.write("\n")
               f3.write("abprop: {}  \n".format(abene[0]))
               f3.flush()
        else:
           rmse1[0]+=torch.sum(torch.square(energy-abene))
           rmse1[1]+=torch.sum(torch.square(force-abforce))
           npoint1+=1
           cart=cart.cpu().detach().numpy()
           species=species.cpu().numpy()
           abforce=abforce.cpu().numpy()
           abene=abene.cpu().numpy()
           f2.write("point \n")
           f2.write("100.0  0.0   0.0 \n")
           f2.write("  0.0 100.0  0.0 \n")
           f2.write("  0.0  0.0  100.0 \n")
           f2.write("pbc  0  0  0 \n")
           for natom in range(numatom):
               f2.write("{}  {}  ".format(atomtype[species[natom]],mass[species[natom]]))
               for idim in range(3):
                   f2.write("{}  ".format(cart[natom,idim]))
               for idim in range(3):
                   f2.write("{}  ".format(abforce[natom,idim]))
               f2.write("\n")
           f2.write("abprop: {}  \n".format(abene[0]))
           f2.flush()
rmse=rmse.detach().cpu().numpy()
rmse1=rmse1.detach().cpu().numpy()
print(np.sqrt(rmse[0]/npoint))
print(np.sqrt(rmse[1]/npoint/15))
print(np.sqrt(rmse1[1]/npoint1/15))
print(np.sqrt(rmse1[0]/npoint1))
print(npoint==npoint1)
f2.close()
f3.close()
os.system("mv configuration_1 configuration")
