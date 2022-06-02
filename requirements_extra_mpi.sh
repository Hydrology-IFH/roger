#!/bin/sh
# install parallel hdf5 from https://www.hdfgroup.org/downloads/hdf5/source-code/
CC=/usr/local/openmpi/bin/mpicc ./configure --enable-shared --enable-parallel --enable-hl --prefix /usr/local
make
make check
sudo make
h5pcc --showconfig
CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/usr/local pip install --no-binary=h5py h5py==3.6.0
