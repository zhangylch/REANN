import os
import gc
import torch
import numpy as np
from src.read_data import *
from src.get_info_of_rank import *
from src.gpu_sel import *
# used for DDP
import torch.distributed as dist


# open a file for output information in iterations
fout=open('nn.err','w')
fout.write("REANN Package used for fitting energy and tensorial Property\n")

# global parameters for input_nn
start_table=0                  # 0 for energy 1 for force 2 for dipole 3 for transition dipole moment 4 for polarizability
table_coor=0                   # 0: cartestion coordinates used 1: fraction coordinates used
table_init=0                   # 1: a pretrained or restart  
nblock = 1                     # nblock>=2  resduial NN block will be employed nblock=1: simple feedforward nn
ratio=0.9                      # ratio for vaildation
#==========================================================
Epoch=10000                  # total numbers of epochs for fitting 
patience_epoch=100              # patience epoch  Number of epochs with no improvement after which learning rate will be reduced. 
decay_factor=0.5               # Factor by which the learning rate will be reduced. new_lr = lr * factor.      
print_epoch=1                 # number of epoch to calculate and print the error
# adam parameter                 
start_lr=0.001                  # initial learning rate
end_lr=1e-5                    # final learning rate
#==========================================================
# regularization coefficence
re_ceff=0.0                 # L2 normalization cofficient
batchsize_train=32                  # batch size 
batchsize_test=256                  # batch size 
e_ceff=0.1                    # weight of energy
init_f = 10                 # initial force weight in loss function
final_f = 5e-1                # final force weight in loss function
nl=[128,128]                  # NN structure
dropout_p=[0.0,0.0]           # dropout probability for each hidden layer
activate = 'Relu_like'          # default "Tanh_like", optional "Relu_like"
queue_size=10
table_norm= True
find_unused = False
#===========param for orbital coefficient ===============================================
oc_loop = 1
oc_nl = [128,128]          # neural network architecture   
oc_nblock = 1
oc_dropout_p=[0.0,0.0]
#=====================act fun===========================
oc_activate = 'Relu_like'          # default "Tanh_like", optional "Relu_like"
#========================queue_size sequence for laod data into gpu
oc_table_norm=True
DDP_backend="nccl"
# floder to save the data
floder="./"
dtype='float32'   #float32/float64
norbit=None
#======================read input_nn=================================================================
with open('para/input_nn','r') as f1:
   while True:
      tmp=f1.readline()
      if not tmp: break
      string=tmp.strip()
      if len(string)!=0:
          if string[0]=='#':
              pass
          else:
              m=string.split('#')
              exec(m[0])

if dtype=='float64':
    torch_dtype=torch.float64
    np_dtype=np.float64
else:
    torch_dtype=torch.float32
    np_dtype=np.float32

# set the default type as double
torch.set_default_dtype(torch_dtype)

#======================read input_density=============================================
# defalut values in input_density
nipsin=2
cutoff=5.0
nwave=6
with open('para/input_density','r') as f1:
   while True:
      tmp=f1.readline()
      if not tmp: break
      string=tmp.strip()
      if len(string)!=0:
          if string[0]=='#':
             pass
          else:
             m=string.split('#')
             exec(m[0])
# increase the nipsin
nipsin+=1

#================ end of read parameter from input file================================

# define the outputneuron of NN
if start_table<=2:
   outputneuron=1
elif start_table==3:
   outputneuron=3
elif start_table==4:
   outputneuron=1

#========================use for read rs/inta or generate rs/inta================
maxnumtype=len(atomtype)
if 'rs' in locals().keys():
   rs=torch.from_numpy(np.array(rs,dtype=np_dtype))
   inta=torch.from_numpy(np.array(inta,dtype=np_dtype))
   nwave=rs.shape[1]
else:
   inta=-(torch.rand(maxnumtype,nwave)+0.2)
   rs=torch.rand(maxnumtype,nwave)*cutoff

if not norbit:
    norbit=int((nwave+1)*nwave/2*(nipsin))
nl.insert(0,norbit)
oc_nl.insert(0,norbit)

#=============================================================================
floder_train=floder+"train/"
floder_test=floder+"test/"
# obtain the number of system
floderlist=[floder_train,floder_test]
# read the configurations and physical properties
if start_table==0 or start_table==1:
    numpoint,atom,mass,numatoms,scalmatrix,period_table,coor,pot,force=  \
    Read_data(floderlist,1,start_table=start_table)
elif start_table==2 or start_table==3:
    numpoint,atom,mass,numatoms,scalmatrix,period_table,coor,dip,force=  \
    Read_data(floderlist,3)
else:
    numpoint,atom,mass,numatoms,scalmatrix,period_table,coor,pol,force=  \
    Read_data(floderlist,9)

#============================convert form the list to torch.tensor=========================
numpoint=np.array(numpoint,dtype=np.int64)
numatoms=np.array(numatoms,dtype=np.int64)
# here the double is used to scal the potential with a high accuracy
initpot=0.0
if start_table<=1:
    pot=np.array(pot,dtype=np.float64).reshape(-1)
    initpot=np.sum(pot)/np.sum(numatoms)
    pot=pot-initpot*numatoms
# get the total number configuration for train/test
ntotpoint=0
for ipoint in numpoint:
    ntotpoint+=ipoint

#define golbal var
if numpoint[1]==0: 
    numpoint[0]=int(ntotpoint*ratio)
    numpoint[1]=ntotpoint-numpoint[0]

ntrain_vec=0
for ipoint in range(numpoint[0]):
    ntrain_vec+=numatoms[ipoint]*3

ntest_vec=0
for ipoint in range(numpoint[0],ntotpoint):
    ntest_vec+=numatoms[ipoint]*3


# parallel process the variable  
#=====================environment for select the GPU in free=================================================
local_rank = int(os.environ.get("LOCAL_RANK"))
local_size = int(os.environ.get("LOCAL_WORLD_SIZE"))
if local_size==1 and local_rank==0: gpu_sel()
world_size = int(os.environ.get("WORLD_SIZE"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu",local_rank)
dist.init_process_group(backend=DDP_backend)
a=torch.empty(100000,device=device)  # used for apply some memory to prevent two process on the smae gpu
if batchsize_train<world_size or batchsize_test<world_size:
    raise RuntimeError("The batchsize used for training or test dataset are smaller than the number of processes, please decrease the number of processes.")
# device the batchsize to each rank
batchsize_train=int(batchsize_train/world_size)
batchsize_test=int(batchsize_test/world_size)
#=======get the minimal data in each process for fixing the bug of different step for each process
min_data_len_train=numpoint[0]-int(np.ceil(numpoint[0]/world_size))*(world_size-1)
min_data_len_test=numpoint[1]-int(np.ceil(numpoint[1]/world_size))*(world_size-1)
if min_data_len_train<=0 or min_data_len_test<=0:
    raise RuntimeError("The size of training or test dataset are smaller than the number of processes, please decrease the number of processes.")
# devide the work on each rank
# get the shifts and atom_index of each neighbor for train
rank=dist.get_rank()
rank_begin=int(np.ceil(numpoint[0]/world_size))*rank
rank_end=min(int(np.ceil(numpoint[0]/world_size))*(rank+1),numpoint[0])
range_train=[rank_begin,rank_end]
com_coor_train,force_train,numatoms_train,species_train,atom_index_train,shifts_train=\
get_info_of_rank(range_train,atom,atomtype,mass,numatoms,scalmatrix,period_table,coor,force,\
start_table,table_coor,neigh_atoms,batchsize_train,cutoff,device,np_dtype)

# get the shifts and atom_index of each neighbor for test
rank_begin=int(np.ceil(numpoint[1]/world_size))*rank
rank_end=min(int(np.ceil(numpoint[1]/world_size))*(rank+1),numpoint[1])
range_test=[numpoint[0]+rank_begin,numpoint[0]+rank_end]
com_coor_test,force_test,numatoms_test,species_test,atom_index_test,shifts_test=\
get_info_of_rank(range_test,atom,atomtype,mass,numatoms,scalmatrix,period_table,coor,force,\
start_table,table_coor,neigh_atoms,batchsize_test,cutoff,device,np_dtype)

# nprop is the number of properties used for training in the same NN  if start_table==1: nprop=2 else nprop=1
nprop=1
if start_table==1: 
    pot_train=torch.from_numpy(np.array(pot[range_train[0]:range_train[1]],dtype=np_dtype))
    pot_test=torch.from_numpy(np.array(pot[range_test[0]:range_test[1]],dtype=np_dtype))
    abpropset_train=(pot_train,force_train)
    abpropset_test=(pot_test,force_test)
    nprop=2
    test_nele=torch.empty(nprop)
    train_nele=torch.empty(nprop)
    train_nele[0]=numpoint[0]
    train_nele[1]=ntrain_vec
    test_nele[0]=numpoint[1] 
    test_nele[1]=ntest_vec
   
if start_table==0: 
    pot_train=torch.from_numpy(np.array(pot[range_train[0]:range_train[1]],dtype=np_dtype))
    pot_test=torch.from_numpy(np.array(pot[range_test[0]:range_test[1]],dtype=np_dtype))
    abpropset_train=(pot_train,)
    abpropset_test=(pot_test,)
    test_nele=torch.empty(nprop)
    train_nele=torch.empty(nprop)
    train_nele[0]=numpoint[0] 
    test_nele[0]=numpoint[1] 

if start_table==2 or start_table==3: 
    dip_train=torch.from_numpy(np.array(dip[range_train[0]:range_train[1]],dtype=np_dtype))
    dip_test=torch.from_numpy(np.array(dip[range_test[0]:range_test[1]],dtype=np_dtype))
    abpropset_train=(dip_train,)
    abpropset_test=(dip_test,)
    test_nele=torch.empty(nprop)
    train_nele=torch.empty(nprop)
    train_nele[0]=numpoint[0]*3
    test_nele[0]=numpoint[1]*3

if start_table==4: 
    pol_train=torch.from_numpy(np.array(pol[range_train[0]:range_train[1]],dtype=np_dtype))
    pol_test=torch.from_numpy(np.array(pol[range_test[0]:range_test[1]],dtype=np_dtype))
    abpropset_train=(pol_train,)
    abpropset_test=(pol_test,)
    test_nele=torch.empty(nprop)
    train_nele=torch.empty(nprop)
    train_nele[0]=numpoint[0]*9
    test_nele[0]=numpoint[1]*9 

# delete the original coordiante
del coor,mass,numatoms,atom,scalmatrix,period_table
if start_table==0: del pot
if start_table==1: del pot,force
if start_table==2 and start_table==3: del dip
if start_table==4: del pol
gc.collect()
    
#======================================================
# random list of index
prop_ceff=torch.ones(2,device=device)
prop_ceff[0]=e_ceff
prop_ceff[1]=init_f
patience_epoch=patience_epoch/print_epoch

# dropout_p for each hidden layer
dropout_p=np.array(dropout_p,dtype=np_dtype)
oc_dropout_p=np.array(oc_dropout_p,dtype=np_dtype)
