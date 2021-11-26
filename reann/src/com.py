import numpy as np

def get_com(coor,force,mass,scalmatrix,numatoms,maxnumatom,table_coor,start_table):
    # shift the com of coordinate of molecule to the origin for warranting the transitional invariance used for DM and polarizability
    ntotpoint=len(coor) 
    maxnumforce=maxnumatom*3
    order_force=None
    com_coor=np.zeros((ntotpoint,maxnumatom,3),dtype=scalmatrix.dtype)
    fcoor=np.zeros((maxnumatom,3),dtype=scalmatrix.dtype)
    if start_table==1: order_force=np.zeros((ntotpoint,maxnumforce),dtype=scalmatrix.dtype)
    for ipoint in range(ntotpoint):
        tmpmass=np.array(mass[ipoint],dtype=scalmatrix.dtype)
        matrix=np.linalg.inv(scalmatrix[ipoint])
        fcoor[0:numatoms[ipoint]]=coor[ipoint]
        if start_table==1: order_force[ipoint,0:numatoms[ipoint]*3]=np.array(force[ipoint],dtype=scalmatrix.dtype).reshape(-1)
        if table_coor==0: fcoor[0:numatoms[ipoint]]=np.matmul(fcoor[0:numatoms[ipoint]],matrix)
        inv_coor=np.round(fcoor[0:numatoms[ipoint]]-fcoor[0])
        fcoor[0:numatoms[ipoint]]-=inv_coor
        fcoor[0:numatoms[ipoint]]=np.matmul(fcoor[0:numatoms[ipoint]],scalmatrix[ipoint,:,:])
        com=np.matmul(tmpmass,fcoor[0:numatoms[ipoint],:])/np.sum(tmpmass)
        com_coor[ipoint,0:numatoms[ipoint]]=fcoor[0:numatoms[ipoint]]-com
    return com_coor,order_force 
