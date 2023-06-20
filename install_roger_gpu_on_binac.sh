#!/bin/sh
conda activate roger-gpu
module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
module load lib/cudnn/8.2-cuda-11.4
pip install --upgrade "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip cache purge
# double-check whether open MPI has CUDA support
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
pip install mpi4py --no-binary mpi4py
HDF5_VERSION=1.12.0 CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/opt/bwhpc/common/lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2 pip install --no-binary=h5py h5py==3.9.0
pip install h5netcdf --no-build-isolation
CUDA_ROOT=/opt/bwhpc/common/devel/cuda/11.4 pip install mpi4jax --no-build-isolation
