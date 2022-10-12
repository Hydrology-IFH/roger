Running Roger on a cluster
==========================
This tutorial walks you through some of the most common challenges that are specific to large, shared architectures like clusters and supercomputers.

In case you having trouble setting up or running Roger on a cluster, you should first contact your cluster administrator. Otherwise, feel free to `open an issue <https://github.com/Hydrology-IFH/roger/issues>`__.

Installation
++++++++++++

Probably the easiest way to try out Roger on a cluster is to, once again, :doc:`use Anaconda </introduction/get-started>`. Since Anaconda is platform independent and does not require elevated permissions, it is the perfect way to try out Roger without too much hassle.

However, **in high-performance contexts, we advise against using Anaconda**. Getting optimal performance requires a software stacked that is linked to the correct system libraries, in particular MPI (see also :doc:`/introduction/advanced-installation`). This requires that Python packages that depend on C libraries (such as ``mpi4py``, ``mpi4jax``) are built from source. For example, installation on `bwForCluster BinAC <https://www.binac.uni-tuebingen.de/>`_:

..  code-block:: shell

  #!/bin/sh
  module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
  echo {$HDF5_VERSION}
  export HDF5_VERSION=1.12.0
  CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/cluster/bwhpc/common/lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2 pip install --no-binary=h5py h5py==3.6.0
  pip install h5netcdf --no-build-isolation
  pip install mpi4jax --no-build-isolation
