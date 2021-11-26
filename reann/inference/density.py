import torch
from torch import nn
from torch import Tensor
from collections import OrderedDict
import numpy as np


class GetDensity(torch.nn.Module):
    def __init__(self,rs,inta,cutoff,nipsin,norbit,ocmod_list):
        super(GetDensity,self).__init__()
        '''
        rs: tensor[ntype,nwave] float
        inta: tensor[ntype,nwave] float
        nipsin: np.array/list   int
        cutoff: float
        '''
        self.rs=nn.parameter.Parameter(rs)
        self.inta=nn.parameter.Parameter(inta)
        self.register_buffer('cutoff', torch.Tensor([cutoff]))
        self.register_buffer('nipsin', torch.tensor([nipsin]))
        npara=[1]
        index_para=torch.tensor([0],dtype=torch.long)
        for i in range(1,nipsin):
            npara.append(int(3**i))
            index_para=torch.cat((index_para,torch.ones((npara[i]),dtype=torch.long)*i))

        self.register_buffer('index_para',index_para)
        # index_para: Type: longTensor,index_para was used to expand the dim of params 
        # in nn with para(l) 
        # will have the form index_para[0,|1,1,1|,2,2,2,2,2,2,2,2,2|...npara[l]..\...]
        self.params=nn.parameter.Parameter(torch.ones_like(self.rs))
        self.hyper=nn.parameter.Parameter(torch.nn.init.orthogonal_(torch.ones(self.rs.shape[1],norbit)).\
        unsqueeze(0).unsqueeze(0).repeat(len(ocmod_list)+1,nipsin,1,1))
        ocmod=OrderedDict()
        for i, m in enumerate(ocmod_list):
            f_oc="memssage_"+str(i)
            ocmod[f_oc]= m
        self.ocmod= torch.nn.ModuleDict(ocmod)

    def gaussian(self,distances,species_):
        # Tensor: rs[nwave],inta[nwave] 
        # Tensor: distances[neighbour*numatom*nbatch,1]
        # return: radial[neighbour*numatom*nbatch,nwave]
        distances=distances.view(-1,1)
        radial=torch.empty((distances.shape[0],self.rs.shape[1]),dtype=distances.dtype,device=distances.device)
        for itype in range(self.rs.shape[0]):
            mask = (species_ == itype)
            ele_index = torch.nonzero(mask).view(-1)
            if ele_index.shape[0]>0:
                part_radial=torch.exp(self.inta[itype:itype+1]*torch.square \
                (distances.index_select(0,ele_index)-self.rs[itype:itype+1]))
                radial.masked_scatter_(mask.view(-1,1),part_radial)
        return radial
    
    def cutoff_cosine(self,distances):
        # assuming all elements in distances are smaller than cutoff
        # return cutoff_cosine[neighbour*numatom*nbatch]
        return torch.square(0.5 * torch.cos(distances * (np.pi / self.cutoff)) + 0.5)

    def angular(self,dist_vec,f_cut):
        # Tensor: dist_vec[neighbour*numatom*nbatch,3]
        # return: angular[neighbour*numatom*nbatch,npara[0]+npara[1]+...+npara[ipsin]]
        totneighbour=dist_vec.shape[0]
        dist_vec=dist_vec.permute(1,0).contiguous()
        orbital=f_cut.view(1,-1)
        angular=torch.empty(self.index_para.shape[0],totneighbour,dtype=f_cut.dtype,device=f_cut.device)
        angular[0]=f_cut
        num=1
        for ipsin in range(1,int(self.nipsin[0])):
            orbital=torch.einsum("ji,ki -> jki",orbital,dist_vec).reshape(-1,totneighbour)
            angular[num:num+orbital.shape[0]]=orbital
            num+=orbital.shape[0]
        return angular    
    
    def forward(self,cart,neigh_list,shifts,species):
        """
        # input cart: coordinates (nbatch*numatom,3)
        # input shifts: coordinates shift values (unit cell)
        # input numatoms: number of atoms for each configuration
        # atom_index: neighbour list indice
        # species: indice for element of each atom
        """ 
        numatom=cart.shape[0]
        neigh_species=species.index_select(0,neigh_list[1])
        selected_cart = cart.index_select(0, neigh_list.view(-1)).view(2, -1, 3)
        dist_vec = selected_cart[0] - selected_cart[1]-shifts
        distances = torch.linalg.norm(dist_vec,dim=-1)
        #dist_vec=dist_vec/distances.view(-1,1)
        orbital = torch.einsum("ji,ik -> ijk",self.angular(dist_vec,self.cutoff_cosine(distances)),self.gaussian(distances,neigh_species))
        orb_coeff=self.params.index_select(0,species)
        density=self.obtain_orb_coeff(0,numatom,orbital,neigh_list,orb_coeff)
        for ioc_loop, (_, m) in enumerate(self.ocmod.items()):
            orb_coeff += m(density,species)
            density = self.obtain_orb_coeff(ioc_loop+1,numatom,orbital,neigh_list,orb_coeff)
        return density

    def obtain_orb_coeff(self,iteration:int,numatom:int,orbital,neigh_list,orb_coeff):
        expandpara=orb_coeff.index_select(0,neigh_list[1])
        worbital=torch.einsum("ijk,ik ->ijk", orbital,expandpara)
        sum_worbital=torch.zeros((numatom,orbital.shape[1],self.rs.shape[1]),dtype=orb_coeff.dtype,device=orb_coeff.device)
        sum_worbital=torch.index_add(sum_worbital,0,neigh_list[0],worbital)
        expandpara=self.hyper[iteration].index_select(0,self.index_para)
        hyper_worbital=torch.einsum("ijk,jkm ->ijm",sum_worbital,expandpara)
        return torch.sum(torch.square(hyper_worbital),dim=1)

