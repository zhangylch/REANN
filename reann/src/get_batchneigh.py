import os
import torch
import numpy as np
import src.get_neighbour as get_neighbour

def get_batch_neigh(com_coor,scalmatrix,species,period,neigh_atoms,batchsize,cutoff,device):
    ntotpoint=com_coor.shape[0]
    maxnumatom=com_coor.shape[1]
    shifts=torch.empty(ntotpoint,maxnumatom*neigh_atoms,3)
    atom_index=torch.empty((2,ntotpoint,maxnumatom*neigh_atoms),dtype=torch.long)
    tmpbatch=1
    maxneigh=0
    for ipoint in range(1,ntotpoint+1):
        if ipoint<ntotpoint and (scalmatrix[ipoint-1]==scalmatrix[ipoint]).all() and \
        (species[ipoint-1]==species[ipoint]).all() and (period[ipoint-1]==period[ipoint]).all \
        and tmpbatch<batchsize:
            tmpbatch+=1
        else:
            cart=com_coor[ipoint-tmpbatch:ipoint].to(device)
            cell=scalmatrix[ipoint-tmpbatch].to(device)
            species_=species[ipoint-tmpbatch:ipoint].to(device)
            pbc=period[ipoint-tmpbatch].to(device)
            tmpindex,tmpshifts,neigh=get_neighbour.neighbor_pairs\
            (pbc, cart, species_, cell, cutoff, neigh_atoms)
            atom_index[:,ipoint-tmpbatch:ipoint]=tmpindex.to("cpu")
            shifts[ipoint-tmpbatch:ipoint]=tmpshifts.to("cpu")
            maxneigh=max(maxneigh,neigh)
            torch.cuda.empty_cache()
            tmpbatch=1
    return shifts[:,0:maxneigh],atom_index[:,:,0:maxneigh]
