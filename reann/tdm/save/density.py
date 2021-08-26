import torch
from torch import nn
from torch import Tensor
from collections import OrderedDict
import numpy as np

class GetDensity(torch.nn.Module):
    def __init__(self,rs,inta,cutoff,nipsin,maxnumtype,nwave,ocmod_list):
        super(GetDensity,self).__init__()
        '''
        rs: tensor[ntype,nwave] float
        inta: tensor[ntype,nwave] float
        nipsin: np.array/list   int
        maxnumtype: int
        nwave: int
        cutoff: float
        '''
        maxnipsin=max(nipsin)
        self.register_buffer('nipsin', torch.tensor(nipsin,dtype=torch.long))
        self.register_buffer('cutoff', torch.Tensor([cutoff]))
        self.maxnumtype=maxnumtype
        self.nwave=nwave
        self.maxnipsin=maxnipsin
        npara=[]
        index_para=torch.tensor([],dtype=torch.long)
        self.ipsin=self.nipsin.shape[0]
        self.p_ori=0
        for j,i in enumerate(self.nipsin):
           npara.append(np.power(3,i))
           index_para=torch.cat((index_para,torch.ones(npara[i],dtype=torch.long)*j))
           self.p_ori+=int(npara[i])

        self.index_para=index_para
        # index_para: Type: longTensor,index_para was used to expand the dim of params 
        # in nn with para(l) 
        # will have the form index_para[0,|1,1,1|,2,2,2,2,2,2,2,2,2|...npara[l]..\...]
        self.rs=nn.parameter.Parameter(rs)
        self.inta=nn.parameter.Parameter(inta)
        self.params=nn.parameter.Parameter(2.0*torch.randn(maxnumtype,nwave)-1.0)
        ocmod=OrderedDict()
        for i, m in enumerate(ocmod_list):
            f_oc="memssage_"+str(i)
            ocmod[f_oc]= m
        self.ocmod = torch.nn.ModuleDict(ocmod)

    def gaussian(self,distances,species_):
        # Tensor: rs[nwave],inta[nwave] 
        # Tensor: distances[neighbour,1]
        # return: radial[neighbour,nwave]
        distances=distances.view(-1,1)
        radial=torch.empty((distances.shape[0],self.nwave),dtype=distances.dtype,\
        device=distances.device)
        for itype in range(self.maxnumtype):
            mask = (species_ == itype)
            ele_index = torch.nonzero(mask).view(-1)
            if ele_index.shape[0] > 0:
                part_radial=torch.exp(-self.inta[itype:itype+1]*torch.square \
                (distances.index_select(0,ele_index)-self.rs[itype:itype+1]))\
                *self.params[itype:itype+1]
                radial.masked_scatter_(mask.view(-1,1),part_radial)
        return radial
    
    def cutoff_cosine(self,distances):
        # assuming all elements in distances are smaller than cutoff
        # return cutoff_cosine[neighbour*nbatch]
        return torch.square(0.5 * torch.cos(distances * (np.pi / self.cutoff)) + 0.5)
    
    def angular(self,dist_vec):
        # return: angular[neighbour,npara[0]+npara[1]+...+npara[ipsin]]
        orbital=dist_vec
        totneighbour=dist_vec.shape[0]
        angular=torch.cat((torch.ones(totneighbour,1,device=dist_vec.device),dist_vec),dim=1)
        num=2
        for ipsin in range(1,self.maxnipsin):
            orbital=torch.einsum("ij,ik -> ijk",orbital,dist_vec).reshape(totneighbour,-1)
            if ipsin+1==self.nipsin[num]:
               angular=torch.cat((angular,orbital),dim=1)
               num+=1
        return angular  
    
    def forward(self,cart,neigh_list,shifts,species):
        """
        # input cart: coordinates (nall,3)
        # input atom_index12(2*maxneigh): store the index of neighbour atoms for each central atom
        # input shift_values: coordinates shift values (unit cell) (maxneigh,3)
        # Tensor: radial
        # angular: orbital form
        """
        numatom=cart.shape[0]
        neigh_species=species.index_select(0,neigh_list[1])
        selected_cart = cart.index_select(0, neigh_list.view(-1)).view(2, -1, 3)
        dist_vec = selected_cart[0] - selected_cart[1]-shifts
        distances = torch.linalg.norm(dist_vec,dim=-1)
        angular=self.angular(dist_vec)
        radial = torch.einsum("ij,i,ik -> ijk",angular,self.cutoff_cosine(distances),\
        self.gaussian(distances,neigh_species))
        orbital = torch.zeros((numatom,self.p_ori,self.nwave),dtype=cart.dtype,device=cart.device)
        orbital = torch.index_add(orbital,0,neigh_list[0],radial)
        part_den=torch.square(orbital)
        density=torch.zeros((numatom,self.ipsin,self.nwave),dtype=cart.dtype,device=cart.device)
        density=torch.index_add(density,1,self.index_para,part_den).view(numatom,-1)
        for ioc_loop, (_, m) in enumerate(self.ocmod.items()):
            output=m(density,species)
            density=self.orbital_coeff(numatom,radial,species,neigh_list,output)
        return density.view(numatom,-1)

    def orbital_coeff(self,numatom:int,orbital,species,neigh_list,output):
        expandpara=output.index_select(0,neigh_list[1])
        expandpara1=expandpara.view(-1,self.ipsin,self.nwave).index_select(1,self.index_para)
        #worbital=oe.contract("ijk, ijk -> ijk",expandpara1,orbital,backend="torch")
        worbital=torch.einsum("ijk, ijk -> ijk",expandpara1,orbital)
        sum_worbital=torch.zeros((numatom,self.p_ori,self.nwave),dtype=output.dtype,device=output.device)
        sum_worbital=torch.index_add(sum_worbital,0,neigh_list[0],worbital)
        part_den=torch.square(sum_worbital)
        oc_density=torch.zeros((numatom,self.ipsin,self.nwave),dtype=output.dtype,device=output.device)
        oc_density=torch.index_add(oc_density,1,self.index_para,part_den)
        return oc_density.view(numatom,-1)
