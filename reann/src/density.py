import torch
from torch import nn
from torch import Tensor
from collections import OrderedDict
import numpy as np
import opt_einsum as oe

class GetDensity(torch.nn.Module):
    def __init__(self,rs,inta,cutoff,neigh_atoms,nipsin,norbit,ocmod_list):
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
            npara.append(np.power(3,i))
            index_para=torch.cat((index_para,torch.ones((npara[i]),dtype=torch.long)*i))

        self.register_buffer('index_para',index_para)
        self.params=nn.parameter.Parameter(torch.ones_like(self.rs)/float(neigh_atoms))
        self.hyper=nn.parameter.Parameter(torch.nn.init.xavier_uniform_(torch.rand(self.rs.shape[1],norbit)).\
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
        for ipsin in range(1,self.nipsin[0]):
            orbital=oe.contract("ji,ki -> jki",orbital,dist_vec,backend="torch").reshape(-1,totneighbour)
            angular[num:num+orbital.shape[0]]=orbital
            num+=orbital.shape[0]
        return angular  
    
    def forward(self,cart,numatoms,species,atom_index,shifts):
        """
        # input cart: coordinates (nbatch*numatom,3)
        # input shifts: coordinates shift values (unit cell)
        # input numatoms: number of atoms for each configuration
        # atom_index: neighbour list indice
        # species: indice for element of each atom
        """
        tmp_index=torch.arange(numatoms.shape[0],device=cart.device)*cart.shape[1]
        self_mol_index=tmp_index.view(-1,1).expand(-1,atom_index.shape[2]).reshape(1,-1)
        cart_=cart.flatten(0,1)
        totnatom=cart_.shape[0]
        padding_mask=torch.nonzero((shifts.view(-1,3)>-1e10).all(1)).view(-1)
        # get the index for the distance less than cutoff (the dimension is reduntant)
        atom_index12=(atom_index.view(2,-1)+self_mol_index)[:,padding_mask]
        selected_cart = cart_.index_select(0, atom_index12.view(-1)).view(2, -1, 3)
        shift_values=shifts.view(-1,3).index_select(0,padding_mask)
        dist_vec = selected_cart[0] - selected_cart[1] + shift_values
        distances = torch.linalg.norm(dist_vec,dim=-1)
        #dist_vec=dist_vec/distances.view(-1,1)
        species_ = species.index_select(0,atom_index12[1])
        orbital = oe.contract("ji,ik -> ijk",self.angular(dist_vec,self.cutoff_cosine(distances)),\
        self.gaussian(distances,species_),backend="torch")
        orb_coeff=torch.empty((totnatom,self.rs.shape[1]),dtype=cart.dtype,device=cart.device)
        mask=(species>-0.5).view(-1)
        orb_coeff.masked_scatter_(mask.view(-1,1),self.params.index_select(0,species[torch.nonzero(mask).view(-1)]))
        density=self.obtain_orb_coeff(0,totnatom,orbital,atom_index12,orb_coeff).view(totnatom,-1)
        for ioc_loop, (_, m) in enumerate(self.ocmod.items()):
            orb_coeff = orb_coeff + m(density,species)
            density = self.obtain_orb_coeff(ioc_loop+1,totnatom,orbital,atom_index12,orb_coeff)
        return density
 
    def obtain_orb_coeff(self,iteration:int,totnatom:int,orbital,atom_index12,orb_coeff):
        expandpara=orb_coeff.index_select(0,atom_index12[1])
        worbital=oe.contract("ijk,ik->ijk", orbital,expandpara,backend="torch")
        sum_worbital=torch.zeros((totnatom,orbital.shape[1],self.rs.shape[1]),dtype=orbital.dtype,device=orbital.device)
        sum_worbital=torch.index_add(sum_worbital,0,atom_index12[0],worbital)
        expandpara=self.hyper[iteration].index_select(0,self.index_para)
        hyper_worbital=oe.contract("ijk,jkm -> ijm",sum_worbital,expandpara,backend="torch")
        return torch.sum(torch.square(hyper_worbital),dim=1)
