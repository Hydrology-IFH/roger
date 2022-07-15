#!/bin/sh
# on BWUniCluster 2.0
module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1
echo {$HDF5_VERSION}
export HDF5_VERSION=1.12.1
CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/opt/bwhpc/common/lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1 pip install --no-binary=h5py h5py==3.6.0
pip install h5netcdf --no-build-isolation
pip install mpi4jax --no-build-isolation

# on BinAC
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
echo {$HDF5_VERSION}
export HDF5_VERSION=1.12.0
CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/cluster/bwhpc/common/lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2 pip install --no-binary=h5py h5py==3.6.0
pip install h5netcdf --no-build-isolation
pip install mpi4jax --no-build-isolation
