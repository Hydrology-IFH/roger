#!/bin/sh
# install Open MPI
# install parallel HDF5
pip install mpi4py --no-binary mpi4py
CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/usr/local pip install --no-binary=h5py h5py==3.6.0
pip install h5netcdf --no-build-isolation
pip install mpi4jax --no-build-isolation
