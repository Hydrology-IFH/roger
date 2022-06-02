#!/bin/sh
echo {$HDF5_VERSION}
export HDF5_VERSION=1.12.1
CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/opt/bwhpc/common/lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1 pip install --no-binary=h5py h5py==3.6.0
conda install h5netcdf --no-deps -c conda-forge
module load devel/cuda/10.2 devel/cudnn/10.2 lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1
module load devel/cuda/11.4
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install mpi4jax --no-build-isolation
pip install diag-eff
