import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist


def Optimize(fout,prop_ceff,nprop,train_nele,test_nele,init_f,final_f,decay_factor,start_lr,end_lr,print_epoch,Epoch,\
data_train,data_test,Prop_class,loss_fn,optim,scheduler,ema,restart,PES_Normal,device,PES_Lammps=None): 

    rank=dist.get_rank()
    best_loss=1e30*torch.ones(1,device=device)    

    for iepoch in range(Epoch): 
        # set the model to train
       Prop_class.train()
       lossprop=torch.zeros(nprop,device=device)        
       for data in data_train:
          abProp,cart,numatoms,species,atom_index,shifts=data
          loss=loss_fn(Prop_class(cart,numatoms,species,atom_index,shifts),abProp)
          lossprop+=loss.detach()
          loss=torch.sum(torch.mul(loss,prop_ceff[0:nprop]))
          # clear the gradients of param
          #optim.zero_grad()
          optim.zero_grad(set_to_none=True)
          #print(torch.cuda.memory_allocated)
          # obtain the gradients
          loss.backward()
          optim.step()   

          #doing the exponential moving average update the EMA parameters
          ema.update()
    
       #  print the error of vailadation and test each print_epoch
       if np.mod(iepoch,print_epoch)==0:
          # apply the EMA parameters to evaluate
          ema.apply_shadow()
          # set the model to eval for used in the model
          Prop_class.eval()
          # all_reduce the rmse form the training process 
          # here we dont need to recalculate the training error for saving the computation
          dist.all_reduce(lossprop,op=dist.ReduceOp.SUM)
          loss=torch.sum(lossprop)
          
          # get the current rank and print the error in rank 0
          if rank==0:
              lossprop=torch.sqrt(lossprop.detach().cpu()/train_nele)
              lr=optim.param_groups[0]["lr"]
              fout.write("{:5} {:4} {:15} {:5e}  {} ".format("Epoch=",iepoch,"learning rate",lr,"train error:"))
              for error in lossprop:
                  fout.write('{:10.5f} '.format(error))
          
          # calculate the test error
          lossprop=torch.zeros(nprop,device=device)
          for data in data_test:
             abProp,cart,numatoms,species,atom_index,shifts=data
             loss=loss_fn(Prop_class(cart,numatoms,species,atom_index,shifts,\
             create_graph=False),abProp)
             lossprop=lossprop+loss.detach()

          # all_reduce the rmse
          dist.all_reduce(lossprop,op=dist.ReduceOp.SUM)
          loss=torch.sum(torch.mul(loss,prop_ceff[0:nprop]))
          scheduler.step(loss)
          lr=optim.param_groups[0]["lr"]
          f_ceff=init_f+(final_f-init_f)*(lr-start_lr)/(end_lr-start_lr+1e-8)
          prop_ceff[1]=f_ceff
          #  save the best model
          if loss<best_loss[0]:
             best_loss[0]=loss
             if rank == 0:
                 state = {'reannparam': Prop_class.state_dict(), 'optimizer': optim.state_dict()}
                 torch.save(state, "./REANN.pth")
                 PES_Normal.jit_pes()
                 if PES_Lammps:
                     PES_Lammps.jit_pes()
          
          # restore the model for continue training
          ema.restore()
          # back to the best error
          if loss>25*best_loss[0] or loss.isnan():
              restart(Prop_class,"REANN.pth")
              optim.param_groups[0]["lr"]=optim.param_groups[0]["lr"]*decay_factor

          if rank==0:
              lossprop=torch.sqrt(lossprop.detach().cpu()/test_nele)
              fout.write('{} '.format("test error:"))
              for error in lossprop:
                 fout.write('{:10.5f} '.format(error))
              # if stop criterion
              fout.write("\n")
              fout.flush()
          if lr <=end_lr: break

