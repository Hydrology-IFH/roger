#!/bin/sh
conda activate roger-mpi
module load lib/hdf5/1.14.4-gnu-13.3-openmpi-4.1
pip install mpi4py --no-binary mpi4py
echo {$HDF5_VERSION}
HDF5_VERSION=1.14.4 CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/opt/bwhpc/common/lib/hdf5/1.14.4-gnu-13.3-openmpi-4.1 pip install --no-binary=h5py h5py==3.11.0
pip install h5netcdf --no-build-isolation
pip install mpi4jax --no-build-isolation