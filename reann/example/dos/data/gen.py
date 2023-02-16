import re
import numpy as np
from write_format import *
pattern=re.compile(r"(?<={}=)\"(.+?)\"".format("Lattice"))
ldos=np.load("ldos.npy")
xdos=np.load("xdos.npy")
num=0
prob=2
npoint=0
fileobj=open("configuration",'w')
with open("training_dataset.xyz",'r') as f1:
    while True:
        string=f1.readline()
        if not string: break
        numatom=int(string)
        string=f1.readline()
        tmp=re.findall(pattern,string)
        cell=np.array(list(map(float,tmp[0].split()))).reshape(3,3)
        element=[]
        mass=[]
        cart=[]
        for i in range(numatom):
            string=f1.readline()
            tmp=string.split()
            element.append(tmp[0])
            mass.append(28.085)
            cart.append(list(map(float,tmp[1:4])))
        ydos=ldos[num]
        num+=1
        if np.random.uniform()<prob:
            for i,ix in enumerate(xdos):
                write_format(fileobj,npoint,np.array([1,1,1]),element,np.array(mass),np.array(cart),ydos[i:i+1],dos_ene=ix,cell=cell)
            npoint+=1
fileobj.close()
