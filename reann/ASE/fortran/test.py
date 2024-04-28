import getneigh as getneigh
import train_1.getneigh as getneigh1
import numpy as np
import sys
print(sys.path)

cutoff=5.0
nwave=2
max_l=10
numatom=8
maxneigh=80
cart=np.random.rand(3,numatom)*2
cell=np.zeros((3,3))
cell[0,0]=100.0
cell[1,1]=100.0
cell[2,2]=100.0
atomindex=np.ones((2,maxneigh),dtype=np.int32)
shifts=np.ones((3,maxneigh))
in_dier=cutoff/2.0
getneigh.init_neigh(cutoff,in_dier,cell)
getneigh1.init_neigh(cutoff,in_dier,cell)

cart,atomindex,shifts,scutnum=getneigh.get_neigh(cart,maxneigh)
getneigh.deallocate_all()
