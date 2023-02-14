import numpy as np
import math

# read system configuration and energy/force
def Read_data(floderlist,nprob,start_table):
    coor=[]
    scalmatrix=[]
    abprop=[] 
    dos_ene=None 
    force=None
    atom=[]
    mass=[]
    numatoms=[]
    period_table=[]
    # tmp variable
    #===================variable for force====================
    if start_table==1:
       force=[]
    else:
       dos_ene=[]
    numpoint=[0 for _ in range(len(floderlist))]
    num=0
    for ifloder,floder in enumerate(floderlist):
        fname2=floder+'configuration'
        with open(fname2,'r') as f1:
            while True:
                string=f1.readline()
                if not string: break
                string=f1.readline()
                scalmatrix.append([])
                m=list(map(float,string.split()))
                scalmatrix[num].append(m)
                string=f1.readline()
                m=list(map(float,string.split()))
                scalmatrix[num].append(m)
                string=f1.readline()
                m=list(map(float,string.split()))
                scalmatrix[num].append(m)
                string=f1.readline()
                m=list(map(float,string.split()[1:4]))
                period_table.append(m)
                coor.append([])
                mass.append([])
                atom.append([])
                if start_table==1: force.append([])
                while True:
                    string=f1.readline()
                    m=string.split()
                    if m[0]=="abprop:":
                        abprop.append(list(map(float,m[1:1+nprob])))
                        if start_table==5: dos_ene.append(m[2+nprob])
                        break
                    atom[num].append(m[0]) 
                    tmp=list(map(float,m[1:]))
                    mass[num].append(tmp[0])
                    coor[num].append(tmp[1:4])
                    if start_table==1:
                        force[num].append(tmp[4:7])
                numpoint[ifloder]+=1
                numatoms.append(len(atom[num]))
                num+=1
    return numpoint,atom,mass,numatoms,scalmatrix,period_table,coor,abprop,dos_ene,force
