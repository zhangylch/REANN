# type: required: obj(file)/int(index)/np.array/list/np.array/np.array/np.array/np.array
#       optional: force: atomic force, cell: lattice parameters
def write_format(fileobj,num,pbc,element,mass,coordinates,properties,dos_ene=None,force=None,cell=None):    
    fileobj.write("point=   {} \n".format(num))
    if cell is not None:
        for i in range(3):
            fileobj.write("{}   {}  {} \n".format(cell[i][0],cell[i][1],cell[i][2]))
    else:
        fileobj.write("100.0    0.0    0.0 \n")
        fileobj.write("  0.0  100.0    0.0 \n")
        fileobj.write("  0.0    0.0  100.0 \n")

    fileobj.write("pbc {}  {}  {} \n".format(pbc[0],pbc[1],pbc[2]))

    for i,ele in enumerate(element):
        fileobj.write('{}  {}  {}  {}  {} '.format(ele,mass[i],coordinates[i,0],coordinates[i,1],coordinates[i,2]))
        if force is not None:
            fileobj.write('{}  {}  {} '.format(force[i,0],force[i,1],force[i,2]))
        fileobj.write("\n")

    fileobj.write("abprop: ")
    for m in properties:
        fileobj.write("{}  ".format(m))
    if dos_ene is not None:
        fileobj.write("dos_ene: {} ".format(dos_ene))
    fileobj.write(' \n')
            
