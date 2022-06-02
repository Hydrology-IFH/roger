#!/bin/sh
module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1
echo {$HDF5_VERSION}
export HDF5_VERSION=1.12.1
CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/opt/bwhpc/common/lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1 pip install --no-binary=h5py h5py
conda install h5netcdf --no-deps -c conda-forge
pip install mpi4jax --no-build-isolation
pip install diag-eff
