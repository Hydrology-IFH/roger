#!/bin/sh
conda activate roger-gpu
module load mpi/openmpi/4.1-gnu-14.2
module load devel/cuda/12.6
pip install -U "jax[cuda12]"
pip cache purge
# double-check whether open MPI has CUDA support
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
pip install mpi4py --no-binary mpi4py
module load lib/hdf5/1.12-gnu-14.2-openmpi-4.1
HDF5_VERSION=1.12.0 CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/opt/bwhpc/common/lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2 pip install --no-binary=h5py h5py==3.13.0
pip install h5netcdf --no-build-isolation
CUDA_ROOT=/opt/bwhpc/common/devel/cuda/11.4 pip install mpi4jax --no-build-isolation
