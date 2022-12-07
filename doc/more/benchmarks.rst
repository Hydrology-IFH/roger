Benchmarks
==========
.. warning::

   The following benchmarks are for general orientation only. Benchmark results are highly platform dependent (e.g. processor frequency); your mileage may vary.


Varying problem size
--------------------

This benchmark varies the size of the computational domain and records the runtime per iteration. The computations are executed on a single computing node with 25 CPU cores and an Nvidia Tesla K80 GPU.

We run the same model code with all Roger backends (``numpy``, ``numpy-mpi``, ``jax``, ``jax-mpi``, ``jax-gpu``).


.. figure:: /_images/benchmarks/svat/SVAT_size_scaling.png
   :width: 500px
   :align: center

.. figure:: /_images/benchmarks/svat/SVAT_size_speedup_numpy.png
  :width: 500px
  :align: center


As a rule of thumb, we find that JAX and NumPy peforms equally well for a greater grid cell count. GPUs are a competitive alternative to CPUs, **as long as the problem fits into GPU memory**.


Varying number of MPI processes
-------------------------------

Roger is run for a fixed problem size, but varying number of processes. This allows us the evaluation of the scaling with increased CPU count. The problem size corresponds to 1 billion cells.

The computational benchmark experiment is executed on the `bwForCluster BinAC  <https://www.binac.uni-tuebingen.de/>`__ cluster. Each computing node contains 28 CPUs.

.. figure:: /_images/benchmarks/svat/SVAT_nproc_scaling.png
   :width: 500px
   :align: center

The results show that Roger scales well with increasing number of processes.
