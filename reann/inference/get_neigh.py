import torch
#import opt_einsum as oe

class Neigh_List(torch.nn.Module):
    def __init__(self,cutoff:float,nlinked:int):
        # nliked used for the periodic boundary condition in cell linked list
        super(Neigh_List,self).__init__()
        self.cutoff=cutoff
        self.cell_list=self.cutoff/nlinked
        r1 = torch.arange(-nlinked, nlinked + 1)
        self.linked=torch.cartesian_prod(r1, r1, r1).view(1,-1,3)

    def forward(self,period_table,coordinates,cell,mass):
        """Compute pairs of atoms that are neighbors
    
        Arguments:
            pbc (:class:`torch.double`): periodic boundary condition for each dimension
            coordinates (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
                defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            cutoff (float): the cutoff inside which atoms are considered pairs
            shifts (:class:`torch.Tensor`): tensor of shape (?, 3) storing shifts
        """
        numatom=coordinates.shape[0]
        inv_cell=torch.inverse(cell)
        inv_coor=torch.einsum("ij,jk -> ik", coordinates, inv_cell)
        deviation_coor=torch.round(inv_coor-inv_coor[0])
        inv_coor=inv_coor-deviation_coor
        coordinates[:,:]=torch.einsum("ij,jk -> ik",inv_coor,cell)
        totmass=torch.sum(mass)
        com=torch.einsum('i,ij->j',mass,coordinates)/totmass
        coordinates[:,:]=coordinates-com[None,:]
        num_repeats = torch.ceil(torch.min(self.cutoff/torch.abs(cell),dim=0)[0]).to(torch.int)
        # the number of periodic image in each direction
        num_repeats = period_table*num_repeats
        num_repeats_up = (num_repeats+1).detach()
        num_repeats_down = (-num_repeats).detach()
        r1 = torch.arange(num_repeats_down[0], num_repeats_up[0], device=coordinates.device)
        r2 = torch.arange(num_repeats_down[1], num_repeats_up[1], device=coordinates.device)
        r3 = torch.arange(num_repeats_down[2], num_repeats_up[2], device=coordinates.device)
        shifts=torch.cartesian_prod(r1, r2, r3).to(coordinates.dtype)
        #shifts=oe.contract("ij,jk ->ik",shifts,cell,backend="torch")
        shifts=torch.einsum("ij,jk ->ik",shifts,cell)
        num_shifts = shifts.shape[0]
        all_shifts = torch.arange(num_shifts, device=coordinates.device)
        all_atoms = torch.arange(numatom, device=coordinates.device)
        prod = torch.cartesian_prod(all_shifts,all_atoms).t().contiguous()
        # used the modified cell_linked algorithm determine the neighbour atoms
        # cut the box with periodic image expand to the ori cell+cutoff in each direction
        # deviation for prevent the min point on the border
        mincoor=torch.min(coordinates,0)[0]-self.cutoff-1e-6
        coordinates=coordinates-mincoor
        maxcoor=torch.max(coordinates,0)[0]+self.cutoff
        image=(coordinates[None,:,:]+shifts[:,None,:]).view(-1,3)
        # get the index in the range (ori_cell-rc,ori_cell+rs) in  each direction
        mask=torch.nonzero(((image<maxcoor)*(image>0)).all(1)).view(-1)
        image_mask=image.index_select(0,mask)
        # save the index(shifts, atoms) for each atoms in the modified cell 
        prod=prod[:,mask]
        ori_image_index=torch.floor(coordinates/self.cell_list)
        # the central atoms with its linked cell index
        cell_linked=self.linked.expand(numatom,-1,3).to(coordinates.device)
        neigh_cell=ori_image_index[:,None,:]+cell_linked
        # all the index for each atoms in the modified cell   
        image_index=torch.floor(image_mask/self.cell_list)
        max_cell_index=torch.ceil(maxcoor/self.cell_list)
        neigh_cell_index=neigh_cell[:,:,2]*max_cell_index[1]*max_cell_index[0]+\
        neigh_cell[:,:,1]*max_cell_index[0]+neigh_cell[:,:,0]
        nimage_index=image_index[:,2]*max_cell_index[1]*max_cell_index[0]+\
        image_index[:,1]*max_cell_index[0]+image_index[:,0]
        dim_image_index=nimage_index.shape[0]
        mask_neigh=torch.nonzero(neigh_cell_index[:,None,:]==nimage_index[None,:,None])
        atom_index=mask_neigh[:,0:2]
        atom_index=atom_index.t().contiguous()
        # step 5, compute distances, and find all pairs within cutoff
        selected_coordinate1 = coordinates.index_select(0, atom_index[0])
        selected_coordinate2 = image_mask.index_select(0, atom_index[1])
        distances = (selected_coordinate1 - selected_coordinate2).norm(2, -1)
        pair_index = torch.nonzero((distances< self.cutoff)*(distances>0.001)).reshape(-1)
        neigh_index = atom_index[:,pair_index]
        tmp=prod[:,neigh_index[1]]
        neigh_list=torch.vstack((neigh_index[0],tmp[1]))
        shifts = shifts.index_select(0, tmp[0])
        return neigh_list, shifts
