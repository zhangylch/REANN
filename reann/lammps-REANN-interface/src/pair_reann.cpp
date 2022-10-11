// Copyright 2018 Andreas Singraber (University of Vienna)
// //
// // This Source Code Form is subject to the terms of the Mozilla Public
// // License, v. 2.0. If a copy of the MPL was not distributed with this
// // file, You can obtain one at http://mozilla.org/MPL/2.0/.
#include <mpi.h>
#include <stdlib.h>
#include <pair_reann.h>
#include <string>
#include <numeric>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "atom.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "utils.h"

using namespace LAMMPS_NS;
using namespace std;

PairREANN::PairREANN(LAMMPS *lmp) : Pair(lmp) 
{
}

PairREANN::~PairREANN() 
{
    if (allocated) 
    {
         memory->destroy(setflag);
         memory->destroy(cutsq);
    }
    // delete the map from the global index to local index
    atom->map_delete();
    atom->map_style = Atom::MAP_NONE;
}

void PairREANN::allocate()
{
    allocated = 1;
    int n = atom->ntypes;
    memory->create(setflag,n+1,n+1,"pair:setflag");
    memory->create(cutsq,n+1,n+1,"pair:cutsq");
    for (int i = 1; i <= n; i++)
    {
        for (int j = i; j <= n; j++)
            setflag[i][j] = 0;
    }
}





void PairREANN::init_style()
{
    int irequest = neighbor->request(this,instance_me);
    neighbor->requests[irequest]->pair = 1;
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->full = 1;
    try 
    {
        //enable the optimize of torch.script
        torch::jit::GraphOptimizerEnabledGuard guard{true};
        torch::jit::setGraphExecutorOptimize(true);
        // load the model 
        // Deserialize the ScriptModule from a file using torch::jit::load().
        if (datatype=="double") module = torch::jit::load("REANN_LAMMPS_DOUBLE.pt");
        else 
        {
            module = torch::jit::load("REANN_LAMMPS_FLOAT.pt");
            tensor_type = torch::kFloat32;
        }
        // freeze the module
        int id;
        if (torch::cuda::is_available()) 
        {
            // used for assign the CUDA_VISIBLE_DEVICES= id
            MPI_Barrier(MPI_COMM_WORLD);
            // return the GPU id for the process
            id=select_gpu();
            torch::DeviceType device_type=torch::kCUDA;
            auto device=torch::Device(device_type,id);
            cout << "The simulations are performed on the GPU" << endl;
            option1=option1.pinned_memory(true);
            option2=option2.pinned_memory(true);
            module.to(device);
            device_tensor=device_tensor.to(device);
        }
        /*else 
        {
            device_type = torch::kCPU;
            device=torch::Device(device_type);
        }*/
        module.eval();
        module=torch::jit::optimize_for_inference(module);
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "error loading the model\n";
    }
    std::cout << "ok\n";

    // create the map from global to local
    if (atom->map_style == Atom::MAP_NONE) {
      atom->nghost=0;
      atom->map_init(1);
      atom->map_set();
    }
  
}

void PairREANN::coeff(int narg, char **arg)
{
    if (!allocated) 
    {
        allocate();
    }

    int n = atom->ntypes;
    int ilo,ihi,jlo,jhi;
    ilo = 0;
    jlo = 0;
    ihi = n;
    jhi = n;
    //utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
    //utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);
    cutoff = utils::numeric(FLERR, arg[2], false,lmp);
    datatype=arg[3];
    cutoffsq=cutoff*cutoff;
    for (int i = ilo; i <= ihi; i++) 
    {
        for (int j = MAX(jlo,i); j <= jhi; j++) 
        {
            setflag[i][j] = 1;
        }
    }
    //if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}


void PairREANN::settings(int narg, char **arg)
{
}


double PairREANN::init_one(int i, int j)
{
  return cutoff;
}
/*=========================copy from the integrate.ev_set===========================
   set eflag,vflag for current iteration
   invoke matchstep() on all timestep-dependent computes to clear their arrays
   eflag/vflag based on computes that need info on this ntimestep
   eflag = 0 = no energy computation
   eflag = 1 = global energy only
   eflag = 2 = per-atom energy only
   eflag = 3 = both global and per-atom energy
   vflag = 0 = no virial computation (pressure)
   vflag = 1 = global virial with pair portion via sum of pairwise interactions
   vflag = 2 = global virial with pair portion via F dot r including ghosts
   vflag = 4 = per-atom virial only
   vflag = 5 or 6 = both global and per-atom virial
=========================================================================================*/
//#pragma GCC push_options
//#pragma GCC optimize (0)

void PairREANN::compute(int eflag, int vflag)
{
    if(eflag || vflag) ev_setup(eflag,vflag);
    else evflag = vflag_fdotr = eflag_global = eflag_atom = 0;
    double **x=atom->x;
    double **f=atom->f;
    int *type = atom->type; 
    tagint *tag = atom->tag;
    int nlocal = atom->nlocal,nghost=atom->nghost;
    int *ilist,*jlist,*numneigh,**firstneigh;
    int i,ii,inum,j,jj,jnum,maxneigh;
    int totneigh=0,nall=nghost+nlocal;
    int totdim=nall*3;
     
    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;
    int numneigh_atom = accumulate(numneigh, numneigh + inum , 0);
    /*torch::Tensor cart = torch::empty({nall,3},torch::dtype(torch::kDouble));
    // for getting the index of local species list atom for all the local atom
    torch::Tensor local_species = torch::empty({inum},torch::dtype(torch::kLong));
    // for getting the index of neigh list atom for all the local atom
    torch::Tensor atom_index=torch::empty({numneigh_atom,2},torch::dtype(torch::kLong));
    // for getting the index of neigh species list atom for all the local atom
    torch::Tensor neigh_list=torch::empty({numneigh_atom},torch::dtype(torch::kLong));*/
    vector<double> cart(totdim);
    vector<long> atom_index(numneigh_atom*2);
    vector<long> neigh_list(numneigh_atom);
    vector<long> local_species(inum);
    double dx,dy,dz,d2;
    double xtmp,ytmp,ztmp;
    unsigned countnum=0;
    // assign the cart with x
    for (ii=0; ii<nall; ++ii)
    {
        for (jj=0; jj<3; ++jj)
        {
            cart[countnum]=x[ii][jj];
            ++countnum;
        }
    }
    for (ii=0; ii<inum; ++ii)
    {
        i=ilist[ii];
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        local_species[i]=type[i]-1;
        jnum=numneigh[i];
        jlist=firstneigh[i];
        for (jj=0; jj<jnum; ++jj)
        {
            j=jlist[jj];
            dx = xtmp - x[j][0];
            dy = ytmp - x[j][1];
            dz = ztmp - x[j][2];
            d2 = dx * dx + dy * dy + dz * dz;
            if (d2<cutoffsq)
            {
                atom_index[totneigh*2]=i;
                atom_index[totneigh*2+1]=j;
                // map the local index for neighbour to the global index, the convert the global index to the local index of center
                neigh_list[totneigh]=atom->map(tag[j]);   
                ++totneigh;
            }
        }
    }
    
    auto cart_=torch::from_blob(cart.data(),{nall,3},option1).to(device_tensor.device(),true).to(tensor_type);
    auto atom_index_=torch::from_blob(atom_index.data(),{totneigh,2},option2).to(device_tensor.device(),true).permute({1,0}).contiguous();
    auto neigh_list_=torch::from_blob(neigh_list.data(),{totneigh},option2).to(device_tensor.device(),true);
    auto local_species_=torch::from_blob(local_species.data(),{inum},option2).to(device_tensor.device(),true);
    auto outputs = module.forward({cart_,atom_index_,local_species_,neigh_list_}).toTuple()->elements();
    auto tensor_etot=outputs[0].toTensor().to(torch::kDouble).cpu();
    auto tensor_force=outputs[1].toTensor().to(torch::kDouble).cpu();
    auto tensor_atom_ene=outputs[2].toTensor().to(torch::kDouble).cpu();
    auto etot = tensor_etot.data_ptr<double>();
    auto force = tensor_force.data_ptr<double>();
    auto atom_ene = tensor_atom_ene.data_ptr<double>();
    for (i=0; i<nall; ++i)
    {
        for (j=0; j<3; ++j)
            f[i][j] += *force++; 
    }

    if (eflag_global)
        ev_tally(0,0,nlocal,1,etot[0],0.0,0.0,0.0,0.0,0.0);

    if (eflag_atom)
        for ( ii = 0; ii < nlocal; ++ii)
        {
            i=ilist[ii];
            eatom[ii] = atom_ene[i];
        }

    if (vflag_fdotr) virial_fdotr_compute();
}
//#pragma GCC pop_options
//
int PairREANN::select_gpu() 
{
    int totalnodes, mynode;
    int trap_key = 0;
    MPI_Status status;
    
    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode);
    if (mynode != trap_key)  //this will allow only process 0 to skip this stmt
        MPI_Recv(&trap_key, 1, MPI_INT, mynode - 1, 0, MPI_COMM_WORLD, &status);
    
    //here is the critical section 
    system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >gpu_info");
    ifstream gpu_sel("gpu_info");
    string texts;
    vector<double> memalloc;
    while (getline(gpu_sel,texts))
    {
         string tmp_str;
         stringstream allocation(texts);        
         allocation >> tmp_str;
         allocation >> tmp_str;
         allocation >> tmp_str;
         memalloc.push_back(std::stod(tmp_str));
    }
    gpu_sel.close();
    auto smallest=min_element(std::begin(memalloc),std::end(memalloc));
    auto id=distance(std::begin(memalloc), smallest);
    torch::Tensor device_tensor=torch::empty(1000,torch::Device(torch::kCUDA,id));
    if(mynode != totalnodes - 1)  // this will allow only the last process to skip this
        MPI_Send(&trap_key, 1, MPI_INT, mynode + 1, 0, MPI_COMM_WORLD);
    system("rm gpu_info");
    return id;
}
