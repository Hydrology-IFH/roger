#!/bin/sh
conda activate roger-mpi
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
pip install mpi4py --no-binary mpi4py
echo {$HDF5_VERSION}
HDF5_VERSION=1.12.0 CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/cluster/bwhpc/common/lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2 pip install --no-binary=h5py h5py==3.7.0
pip install h5netcdf --no-build-isolation
pip install mpi4jax --no-build-isolation
