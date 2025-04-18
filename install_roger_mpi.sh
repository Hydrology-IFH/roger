#!/bin/sh
# requires Open MPI to be installed
# requires parallel HDF5 to be installed
pip install mpi4py --no-binary mpi4py
CC=mpicc HDF5_MPI="ON" HDF5_DIR=/usr/local/hdf5 pip install --no-binary=h5py h5py==3.11.0 --no-build-isolation
pip install h5netcdf --no-build-isolation
pip install mpi4jax --no-build-isolation
