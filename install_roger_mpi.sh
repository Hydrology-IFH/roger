#!/bin/sh
# requires Open MPI to be installed
# requires parallel HDF5 to be installed
conda install mpi4py
CC=mpicc HDF5_MPI="ON" HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/mpich pip install --no-binary=h5py h5py==3.15.0 --no-build-isolation
pip install h5netcdf --no-build-isolation
pip install mpi4jax --no-build-isolation
