#!/bin/sh
conda activate roger-gpu
module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1
module load devel/cudnn/9.2
module load devel/cuda/11.4
pip install mpi4py --no-binary mpi4py
echo {$HDF5_VERSION}
export HDF5_VERSION=1.12.1
CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/opt/bwhpc/common/lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1 pip install --no-binary=h5py h5py==3.6.0
pip install h5netcdf --no-build-isolation
salloc -p gpu_4 --gres=gpu:1 -n 24 -t 02:00:00 --mem 376000
conda activate roger-gpu
module load compiler/llvm/12.0 # only required to build jaxlib from source
module load devel/cudnn/9.2
module load devel/cuda/11.4
# # build jaxlib from source
# # delete in case unsucessful installation
# cd ~/.cache
# rm -rf bazel
# cd ~
# rm -rf jax
# echo {$CUDA_PATH}
# echo {$CUDNN_PATH}
# cd ~
# git clone https://github.com/google/jax
# cd jax
# python build/build.py --enable_cuda --cudnn_path /opt/bwhpc/common/devel/cudnn/9.2 --cuda_path /opt/bwhpc/common/devel/cuda/11.4
# pip command below does not require build from source
pip install --upgrade "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip cache purge
CUDA_ROOT=/opt/bwhpc/common/devel/cuda/11.4 pip install mpi4jax --no-build-isolation
pip install diag-eff

# check whether open MPI has CUDA support
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value

# on BinAC
export HDF5_VERSION=1.12.0
CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/opt/bwhpc/common/lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2 pip install --no-binary=h5py h5py==3.6.0
