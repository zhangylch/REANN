#Copyright 2018- Xiang Gao and other ANI developers
#(https://github.com/aiqm/torchani/blob/master/torchani/aev.py)
# origin compute_shifts have problem in expanding image,we have modified
import math
import numpy as np
import torch
import opt_einsum as oe

@torch.jit.script
def neighbor_pairs(pbc, coordinates, species, cell, cutoff:float, neigh_atoms:int):
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
    padding_mask = ( species == -1)
    num_mols = padding_mask.shape[0]
    num_atoms = padding_mask.shape[1]
    coordinates = coordinates.detach()
    cell = cell.detach()
    num_repeats = [pbc[i]*torch.ceil(cutoff/torch.max(torch.abs(cell[:,i]))).to(cell.device) \
    for i in range(3)]
    r1 = torch.arange(-num_repeats[0], num_repeats[0] + 1, device=cell.device)
    r2 = torch.arange(-num_repeats[1], num_repeats[1] + 1, device=cell.device)
    r3 = torch.arange(-num_repeats[2], num_repeats[2] + 1, device=cell.device)
    shifts=torch.cartesian_prod(r1, r2, r3)
    #shifts=oe.contract("ij,jk ->ik",shifts,cell,backend="torch")
    shifts=torch.einsum("ij,jk ->ik",shifts,cell)

    num_shifts = shifts.shape[0]
    all_shifts = torch.arange(num_shifts, device=cell.device)
    all_atoms = torch.arange(num_atoms, device=cell.device)
    prod = torch.cartesian_prod(all_shifts, all_atoms, all_atoms).t().contiguous()
    shift_index = prod[0]
    p12_all = prod[1:]
    shifts_all = shifts.index_select(0, shift_index)

    # step 5, compute distances, and find all pairs within cutoff
    selected_coordinates = coordinates[:, p12_all.view(-1)].view(num_mols, 2, -1, 3)
    distances = (selected_coordinates[:, 0, ...] - selected_coordinates[:, 1, ...] + shifts_all).norm(2, -1)
    padding_mask = padding_mask[:, p12_all.view(-1)].view(num_mols, 2, -1).any(1)
    distances.masked_fill_(padding_mask, math.inf)  # dim=num_mols*(nshift*natom*natom)
    atom_index=torch.zeros((2,num_mols,num_atoms*neigh_atoms),device=cell.device,dtype=torch.long)
    shifts=-1e11*torch.ones((num_mols,num_atoms*neigh_atoms,3),device=cell.device)
    maxneigh=0 
    for inum_mols in range(num_mols):
        pair_index = torch.nonzero(((distances[inum_mols] <= cutoff)*(distances[inum_mols]>0.01))).reshape(-1)
        atom_index[:,inum_mols,0:pair_index.shape[0]] = p12_all[:,pair_index]
        shifts[inum_mols,0:pair_index.shape[0],:] = shifts_all.index_select(0, pair_index)
        maxneigh=max(maxneigh,pair_index.shape[0])
    return atom_index, shifts, maxneigh
