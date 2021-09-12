import torch
import numpy as np


class DataLoader():
    def __init__(self,image,label,numatoms,index_ele,atom_index,shifts,batchsize,shuffle=True):
        self.image=image
        self.label=label
        self.index_ele=index_ele
        self.numatoms=numatoms
        self.atom_index=atom_index
        self.shifts=shifts
        self.batchsize=batchsize
        self.dim=self.image.shape[0]
        self.end=self.dim        # neglect the last batch that less than the batchsize
        self.shuffle=shuffle               # to control shuffle the data
        if self.shuffle:
            self.shuffle_list=torch.randperm(self.dim)
        else:
            self.shuffle_list=torch.arange(self.dim)
        self.length=int(np.ceil(self.end/self.batchsize))
      
    def __iter__(self):
        self.ipoint = 0
        return self

    def __next__(self):
        if self.ipoint < self.end:
            index_batch=self.shuffle_list[self.ipoint:min(self.end,self.ipoint+self.batchsize)]
            coordinates=self.image.index_select(0,index_batch)
            abprop=(label.index_select(0,index_batch) for label in self.label)
            species=self.index_ele.index_select(0,index_batch)
            shifts=self.shifts.index_select(0,index_batch)
            numatoms=self.numatoms.index_select(0,index_batch)
            atom_index=self.atom_index.index_select(1,index_batch)
            self.ipoint+=self.batchsize
            return abprop,coordinates,numatoms,species,atom_index,shifts
        else:
            # if shuffle==True: shuffle the data 
            if self.shuffle:
                self.shuffle_list=torch.randperm(self.dim)
            raise StopIteration

