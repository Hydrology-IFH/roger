#!/bin/sh
conda activate roger-gpu
module load lib/cudnn/10.2
module load lib/hdf5/1.14.4-gnu-13.3-openmpi-5.0  # loads CUDA 12.2 as well
pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip cache purge
# double-check whether open MPI has CUDA support
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
pip install mpi4py --no-binary mpi4py
HDF5_VERSION=1.14.4 CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/opt/bwhpc/common/lib/hdf5/1.14.4-gnu-13.3-openmpi-5.0 pip install --no-binary=h5py h5py==3.11.0
pip install h5netcdf --no-build-isolation
CUDA_ROOT=/opt/bwhpc/common/devel/cuda/12.2 pip install mpi4jax --no-build-isolation
