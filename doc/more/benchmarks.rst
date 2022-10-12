Benchmarks
==========
.. warning::

   The following benchmarks are for general orientation only. Benchmark results are highly platform dependent; your mileage may vary.


Varying problem size
--------------------

This benchmark varies the size of the computational domain and records the runtime per iteration. It is executed on a single machine with 25 CPU cores and an Nvidia Tesla K80 GPU.

We run the same model code with all Roger backends (``numpy``, ``numpy-mpi``, ``jax``, ``jax-mpi``, ``jax-gpu``).


.. figure:: /_images/benchmarks/oneD_scaling_size.png
   :width: 500px
   :align: center


As a rule of thumb, we find that JAX is faster than NumPy for a greater grid cell count. GPUs are a competitive alternative to CPUs, **as long as the problem fits into GPU memory**.


Varying number of MPI processes
-------------------------------

In this benchmark, Roger is run for a fixed problem size, but varying number of processes. This allows us to check how model scales with increased CPU count. The problem size corresponds to 1 billion cells.

It is executed on the `bwForCluster BinAC  <https://www.binac.uni-tuebingen.de/>`__ cluster. Each cluster node contains 28 CPUs.



The results show that Roger scales well with increasing number of processes, even for this moderate problem size.
