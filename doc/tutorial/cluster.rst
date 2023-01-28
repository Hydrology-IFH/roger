Running Roger on a cluster
==========================
This tutorial walks you through some of the most common challenges that are specific to large, shared architectures like clusters and supercomputers.

In case you having trouble setting up or running Roger on a cluster, you should first contact your cluster administrator. Otherwise, feel free to `open an issue <https://github.com/Hydrology-IFH/roger/issues>`__.

Installation for CPU computing
++++++++++++++++++++++++++++++

Probably the easiest way to try out Roger on a cluster is to, once again, :doc:`use Anaconda </introduction/get-started>`. Since Anaconda is platform independent and does not require elevated permissions, it is the perfect way to try out Roger without too much effort.

However, **in high-performance contexts, we advise to use pip install within your anaconda environment**. Getting optimal performance requires a software stacked that is linked to the correct system libraries, in particular MPI (see also :doc:`/introduction/advanced-installation`). This requires that Python packages that depend on C libraries (such as ``mpi4py``, ``mpi4jax``) are built from source. For example, installation on `bwForCluster BinAC <https://www.binac.uni-tuebingen.de/>`_:

..  code-block:: shell

  #!/bin/sh
  module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
  echo {$HDF5_VERSION}
  export HDF5_VERSION=1.12.0
  CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/cluster/bwhpc/common/lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2 pip install --no-binary=h5py h5py==3.6.0
  pip install h5netcdf --no-build-isolation
  pip install mpi4jax --no-build-isolation

Installation for CPU and GPU computing
++++++++++++++++++++++++++++++++++++++

..  code-block:: shell

  #!/bin/sh
  module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4
  module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
  module load lib/cudnn/8.2-cuda-11.4
  pip install --upgrade "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  pip cache purge
  # double-check whether open MPI has CUDA support
  ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
  pip install mpi4py --no-binary mpi4py
  HDF5_VERSION=1.12.0 CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/opt/bwhpc/common/lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2 pip install --no-binary=h5py h5py==3.7.0
  pip install h5netcdf --no-build-isolation
  CUDA_ROOT=/opt/bwhpc/common/devel/cuda/11.4 pip install mpi4jax --no-build-isolation

Submitting a CPU job to a cluster
+++++++++++++++++++++++++++++++++
In order to run your simulations on the CPUs of a computing cluster, requires the submission of a job script. One possible way to write such a job script for the scheduling manager MOAB is presented here:

..  code-block:: shell

  #!/bin/bash
  #PBS -l nodes=1:ppn=4
  #PBS -l walltime=2:00:00
  #PBS -l pmem=4000mb
  #PBS -N my_setup

  # activate your python environment
  eval "$(conda shell.bash hook)"
  conda activate roger-mpi

  # load your module dependencies
  module purge
  module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2

  # adapt command to your available scheduler / MPI implementation
  mpirun --bind-to core --map-by core -report-bindings python my_setup.py -b numpy -d cpu -n 2 2


Submitting a GPU job to a cluster
+++++++++++++++++++++++++++++++++

In order to run your simulations on the GPU of a computing cluster, requires the submission of a job script. One possible way to write such a job script for the scheduling manager MOAB is presented here:

..  code-block:: shell
  
  #!/bin/bash
  #PBS -l nodes=1:ppn=1:gpus=1:default
  #PBS -l walltime=2:00:00
  #PBS -l pmem=8000mb
  #PBS -N my_setup

  # activate your python environment
  eval "$(conda shell.bash hook)"
  conda activate roger-gpu

  # load your module dependencies
  module purge
  module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4
  module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
  module load lib/cudnn/8.2-cuda-11.4
  export OMP_NUM_THREADS=1

  # adapt command to your available scheduler / MPI implementation
  python my_setup.py -b jax -d gpu
