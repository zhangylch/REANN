#omp_flags= "-D PKG_USER-OMP=ON -D BUILD_OMP=ON"
#bulid_type="-D CMAKE_BUILD_TYPE=RELEASE "
#build_mpi="-D BUILD_MPI=ON"
#shared_flags="-D CMAKE_SHARED_LINKER_FLAGS='-O3 -DNDEBUG' "
#cmake -D BUILD_MPI=ON -D BUILD_OMP=ON -D CMAKE_SHARED_LINKER_FLAGS='-O3 -DNDEBUG'  -D CMAKE_BUILD_TYPE=RELEASE ../cmake 
#cmake -D LAMMPS_MACHINE=mpi -D PKG_USER-INTEL=yes -D INTEL_ARCH=cpu -D INTEL_LRT_MODE=none -D PKG_OPT=yes -D CMAKE_BUILD_TYPE=RELEASE ../cmake
#export CMAKE_LIBRARY_PATH=/share/software/compiler/intel/intel-compiler-2017.5.239/compiler/lib/intel64:/share/software/compiler/intel/intel-compiler-2017.5.239/mkl/lib/intel64:/share/software/compiler/intel/intel-compiler-2017.5.239/tbb/lib/intel64:/share/group-soft/gcc-9.1/gcc-install/lib:$CMAKE_LIBRARY_PATH
cmake -D BUILD_MPI=ON -D PKG_USER-OMP=ON -D BUILD_OMP=ON -D LAMMPS_MACHINE=mpi -D CMAKE_BUILD_TYPE=RELEASE ../cmake
