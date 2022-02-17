#! /usr/bin/env python3
import time
from src.read import *
from src.dataloader import *
from src.optimize import *
from src.density import *
from src.MODEL import *
from src.EMA import *
from src.restart import *
from torch.nn.parallel import DistributedDataParallel as DDP
if activate=='Tanh_like':
    from src.activate import Tanh_like as actfun
else:
    from src.activate import Relu_like as actfun

if oc_activate=='Tanh_like':
    from src.activate import Tanh_like as oc_actfun
else:
    from src.activate import Relu_like as oc_actfun

if start_table==0:
    from src.Property_energy import *
elif start_table==1:
    from src.Property_force import *
elif start_table==2:
    from src.Property_DM import *
elif start_table==3:
    from src.Property_TDM import *
elif start_table==4:
    from src.Property_POL import *
from src.cpu_gpu import *
from src.Loss import *
PES_Lammps=None
if start_table<=1:
    import pes.script_PES as PES_Normal
    if oc_loop==0:
        import lammps.script_PES as PES_Lammps
    else:
        import lammps_REANN.script_PES as PES_Lammps
elif start_table==2:
    import dm.script_PES as PES_Normal
elif start_table==3:
    import tdm.script_PES as PES_Normal
elif start_table==4:
    import pol.script_PES as PES_Normal

#==============================train data loader===================================
dataloader_train=DataLoader(com_coor_train,abpropset_train,numatoms_train,\
species_train,atom_index_train,shifts_train,batchsize_train,min_data_len=min_data_len_train,shuffle=True)
#=================================test data loader=================================
dataloader_test=DataLoader(com_coor_test,abpropset_test,numatoms_test,\
species_test,atom_index_test,shifts_test,batchsize_test,min_data_len=min_data_len_test,shuffle=False)
# dataloader used for load the mini-batch data
if torch.cuda.is_available(): 
    data_train=CudaDataLoader(dataloader_train,device,queue_size=queue_size)
    data_test=CudaDataLoader(dataloader_test,device,queue_size=queue_size)
else:
    data_train=dataloader_train
    data_test=dataloader_test
#==============================oc nn module=================================
# outputneuron=nwave for each orbital have a different coefficients
ocmod_list=[]
for ioc_loop in range(oc_loop):
    ocmod=NNMod(maxnumtype,nwave,atomtype,oc_nblock,list(oc_nl),oc_dropout_p,oc_actfun,table_norm=oc_table_norm)
    ocmod_list.append(ocmod)
#=======================density======================================================
getdensity=GetDensity(rs,inta,cutoff,neigh_atoms,nipsin,norbit,ocmod_list)
#==============================nn module=================================
nnmod=NNMod(maxnumtype,outputneuron,atomtype,nblock,list(nl),dropout_p,actfun,initpot=initpot,table_norm=table_norm)
nnmodlist=[nnmod]
if start_table == 4:
    nnmod1=NNMod(maxnumtype,outputneuron,atomtype,nblock,list(nl),dropout_p,actfun,table_norm=table_norm)
    nnmod2=NNMod(maxnumtype,outputneuron,atomtype,nblock,list(nl),dropout_p,actfun,table_norm=table_norm)
    nnmodlist.append(nnmod1)
    nnmodlist.append(nnmod2)
#=========================create the module=========================================
Prop_class=Property(getdensity,nnmodlist).to(device)  # to device must be included

##  used for syncbn to synchronizate the mean and variabce of bn 
#Prop_class=torch.nn.SyncBatchNorm.convert_sync_batchnorm(Prop_class).to(device)
if world_size>1:
    if torch.cuda.is_available():
        Prop_class = DDP(Prop_class, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused)
    else:
        Prop_class = DDP(Prop_class, find_unused_parameters=find_unused)

#define the loss function
loss_fn=Loss()

#define optimizer
optim=torch.optim.AdamW(Prop_class.parameters(), lr=start_lr, weight_decay=re_ceff)

# learning rate scheduler 
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,factor=decay_factor,patience=patience_epoch,min_lr=end_lr)

#define the restart
restart=Restart(optim)

# load the model from EANN.pth
if table_init==1:
    restart(Prop_class,"REANN.pth")
    nnmod.initpot[0]=initpot
    if optim.param_groups[0]["lr"]>start_lr: optim.param_groups[0]["lr"]=start_lr  #for restart with a learning rate 
    if optim.param_groups[0]["lr"]<end_lr: optim.param_groups[0]["lr"]=start_lr  #for restart with a learning rate 
    lr=optim.param_groups[0]["lr"]
    f_ceff=init_f+(final_f-init_f)*(lr-start_lr)/(end_lr-start_lr+1e-8)
    prop_ceff[1]=f_ceff


ema = EMA(Prop_class, 0.999)
#==========================================================
if dist.get_rank()==0:
    fout.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
    fout.flush()
    for name, m in Prop_class.named_parameters():
        print(name)
#==========================================================
Optimize(fout,prop_ceff,nprop,train_nele,test_nele,init_f,final_f,decay_factor,start_lr,end_lr,print_epoch,Epoch,\
data_train,data_test,Prop_class,loss_fn,optim,scheduler,ema,restart,PES_Normal,device,PES_Lammps=PES_Lammps)
if dist.get_rank()==0:
    fout.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
    fout.write("terminated normal\n")
    fout.close()
