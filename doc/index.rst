:tocdepth: 5


.. image:: /_images/roger-logo.png
   :align: center

|

Runoff Generation Research in Pure Python
==========================================
Roger, *Runoff Generation Research*, is a process-based hydrologic model that can be applied from plot to catchment scale. Roger is written in pure Python, which facilitates model setup and model workflows. We want to enable high-performance hydrologic modelling with a clear focus on flexibility and usability.

Roger supports a NumPy backend for small-scale problems, and a
high-performance `JAX <https://github.com/google/jax>`_ backend
with CPU and GPU support. Parallel computation is available via MPI and supports
distributed execution on any number of nodes/CPU cores, including multi-GPU architectures.

Inspired by `Veros <https://veros.readthedocs.io/en/latest/>`_.


Roger, *Runoff Generation Research*, is a process-based hydrologic model that supports anything between plot and catchment scale. Roger is written in pure Python, which facilitates model setup workflows.

*We want to enable high-performance hydrologic modelling with a clear focus on flexibility and usability.*

Roger supports a NumPy backend for small-scale problems, and a
high-performance `JAX <https://github.com/google/jax>`_ backend
with CPU and GPU support. It is fully parallelized via MPI and supports
distributed execution on any number of nodes, including multi-GPU architectures (see also ...).

Inspired by `Veros <https://veros.readthedocs.io/en/latest/>`_.

If you want to learn more about the background and capabilities of Roger, you should check out :doc:`introduction/introduction`. If you are already convinced, you can jump right into action, and :doc:`learn how to get started <introduction/get-started>` instead!


.. toctree::
   :maxdepth: 2
   :caption: Start here

   introduction/introduction
   introduction/get-started
   introduction/advanced-installation

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial/analysis
   tutorial/cluster
   tutorial/dev

.. toctree::
  :maxdepth: 2
  :caption: Model equations

  equations/equations
  equations/hydrologic_cycle
  equations/solute_transport

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/setup-gallery
   reference/settings
   reference/variables
   reference/diagnostics
   reference/cli
   reference/public-api

.. toctree::
   :maxdepth: 2
   :caption: More Information

   more/benchmarks
   more/howtocite
   more/publications
   Visit us on GitHub <https://github.com/Hydrology-IFH/roger>
