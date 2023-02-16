Getting started
===============

Installation
------------

Quick installation via pip
++++++++++++++++++++++++++

.. warning::

  You should only install Roger this way if you want to get going as quickly as possible, and do not plan to access or modify the model source code. The recommended way to install Roger is by checking out the repository (see below).

If you already have Python installed, the quickest way to get a working Roger installation is to run ::

  $ pip install roger

or, optionally::

  $ pip install roger[jax]

to use Roger with JAX.


Checking out the repository and using Conda (multi-platform)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

1. `Download and install Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__. If you are using Windows, you may use the Anaconda prompt to execute the following steps.

2. Clone the Roger repository

   .. exec::

      from roger import __version__ as roger_version
      if "+" in roger_version:
          roger_version, _ = roger_version.split("+")
      print(".. code-block::\n")
      print(f"   $ git clone https://github.com/Hydrology-IFH/roger.git -b v{roger_version}")

   (or `any other version of Roger <https://github.com/Hydrology-IFH/roger/releases>`__).

   If you do not have git installed, you can do so via ``conda install git``.

3. Create a new conda environment for Roger, and install all relevant dependencies by running ::

       $ conda env create -f conda-environment.yml

   from the Roger root directory.

4. To use Roger, just activate your new conda environment via ::

       $ conda activate roger


Checking out the repository and using pip (Linux / OSX)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

1. Ensure you have a working Python 3.x installation.

2. Clone our repository:

   .. exec::

      from roger import __version__ as roger_version
      if "+" in roger_version:
          roger_version, _ = roger_version.split("+")
      print(".. code-block::\n")
      print(f"   $ git clone https://github.com/Hydrology-IFH/roger.git -b v{roger_version}")

   (or `any other version of Roger <https://github.com/Hydrology-IFH/roger/releases>`__), or use ::

      $ pip download roger

   to download a tarball of the latest version (needs to be unpacked).

3. Install Roger (preferably in a virtual environment) via ::

      $ pip install -e ./roger

   You might have to add the ``--user`` flag to ``pip install`` if you are using your system interpreter. The ``-e`` flag ensures that changes to the code are immediately reflected without reinstalling.

4. Optionally, install JAX via ::

      $ pip install -e ./roger[jax]


Setting up a model
------------------

To run Roger, you need to set up a model - i.e., specify which settings and model domain you want to use. This is done by subclassing the :class:`Roger setup base class <roger.RogerSetup>` in a *setup script* that is written in Python. You should have a look at the pre-implemented model setups in the repository's :file:`models` folder, or use the :command:`roger copy-model` command to copy one into your current folder. A good place to start is the :class:`SVAT model <svat.SVATetup>`::

    $ roger copy-setup svat

By working through the existing models, you should quickly be able to figure out how to write your own simulation. Just keep in mind this general advice:

- You can (and should) use any (external) Python tools you want in your pre-processing or model setup. Before implementing a certain functionality, you should check whether it is already provided by a common library. Especially `the SciPy module family <https://www.scipy.org/>`_ provides countless implementations of common scientific functions (and SciPy is installed along with Roger).

- You have to decorate your methods with :func:`@roger_routine <roger.roger_routine>`. Only Roger routines are able to modify the :class:`model state object <roger.RogerState>`, which is passed as the first argument. The current numerical backend is available from the :mod:`roger.core.operators` module::

      from roger import RogerSetup, roger_routine
      from roger.core.operators import numpy as npx

      class MyRogerSetup(RogerSetup):
          ...
          @roger_routine
          def my_function(self, state):
              arr = npx.array([1, 2, 3, 4]) # "npx" uses either NumPy or JAX

- If you are curious about the general process how a model is set up and ran, you should read the source code of :class:`roger.RogerSetup` (especially the :meth:`setup` and :meth:`run` methods). This is also the best way to find out about the order in which routines are called.

- Out of all functions that need to be implemented by your subclass of :class:`roger.RogerSetup`, the only one that is called in every time step is :meth:`set_forcing` (at the beginning of each iteration). This implies that, to achieve optimal performance, you should consider moving calculations that are constant in time to other functions.

- There is another type of decorator called :func:`@roger_kernel <roger.roger_kernel>`. A kernel is a pure function that may be compiled to machine code by JAX. Kernels typically execute much faster, but are more restrictive to implement, as they cannot interact with the model state directly.

  A common pattern in large setups is to implement :meth:`set_forcing` as a kernel for optimal performance.


Running Roger
-------------

After adapting your setup script, you are ready to run your first simulation. Just execute the following::

   $ python my_setup.py

.. seealso::

   The Roger command line interface accepts a large number of options to configure your run; see :doc:`/reference/cli`.

.. note::

   You are not required to use the command line, and you are welcome to include your simulation class into other Python files and call it dynamically or interactively (e.g. in an IPython session). All you need to do is to call the ``setup()`` and ``run()`` methods of your :class:`roger.RogerSetup` object.


Reading Roger output
++++++++++++++++++++

All output is handled by :doc:`the available diagnostics </reference/diagnostics>`. The most basic diagnostic, :class:`collect <roger.diagnostics.Collect>`, writes some model variables to netCDF files in regular intervals (and puts them into your current working directory).

NetCDF is a binary format that is widely adopted in the geophysical modeling community. There are various packages for reading, visualizing and processing netCDF files (such as `ncview <http://meteora.ucsd.edu/~pierce/ncview_home_page.html>`_ and `ferret <http://ferret.pmel.noaa.gov/Ferret/>`_), and bindings for many programming languages (such as C, Fortran, MATLAB, and Python).

For post-processing in Python, we recommend that you use `xarray <http://xarray.pydata.org/en/stable/>`__::

   import xarray as xr

   ds = xr.open_dataset("SVAT.rate.nc", engine="h5netcdf")

   # plot macropore infiltration
   inf_mp = ds.inf_mp.isel(x=0, y=0)
   inf_mp.plot()


Re-starting from a previous run
+++++++++++++++++++++++++++++++

Restart data (in HDF5 format) is written at the end of each simulation or after a regular time interval if the setting :ref:`restart_frequency <setting-restart_frequency>` is set to a finite value. To use this restart file as initial conditions for another simulation, you will have to point :ref:`restart_input_filename <setting-restart_input_filename>` of the new simulation to the corresponding restart file. This can also be given via the command line (as all settings)::

   $ roger run my_setup.py -s restart_input_filename /path/to/restart_file.h5

.. _mpi-exec:

Running Roger on multiple processes via MPI
+++++++++++++++++++++++++++++++++++++++++++

.. note::

  This assumes that you are familiar with running applications through MPI, and is most useful on large architectures like a compute cluster. For smaller architectures, it is usually easier to stick to the thread-based parallelism of JAX.

Running Roger through MPI requires some additional dependencies. For optimal performance, you will need to install ``mpi4py``, ``h5py``, and ``mpi4jax``, linked to your MPI library.

.. seealso::

   :doc:`advanced-installation`

After you have installed everything, you can start Roger on multiple processes like so:::

   $ mpirun -np 4 python my_setup.py -n 2 2

In this case, Roger would run on 4 processes, each process computing one-quarter of the domain. The arguments of the `-n` flag specify the number of domain partitions in x and y-direction, respectively.

.. seealso::

   For more information, see :doc:`/tutorial/cluster`.

Enhancing Roger
---------------

Roger was written with a strong focus on extensibility. If you already know some Python and have worked with NumPy, you are pretty much ready to write your own extension. The model code is located in the :file:`roger` subfolder, while all of the numerical routines are located in :file:`roger/core`.

We believe that the best way to learn how Roger works is to read its source code. Starting from the :py:class:`Roger base class <roger.RogerSetup>`, you should be able to work your way through the flow of the program, and figure out where to add your modifications. If you installed Roger through :command:`pip -e` or :command:`setup.py develop`, all changes you make will immediately be reflected when running the code.

In case you want to add additional output capabilities or compute additional quantities without changing the main solution of the simulation, you should consider :doc:`adding a custom diagnostic </reference/diagnostics>`.

A convenient way to implement your modifications is to create your own fork of Roger on GitHub, and submit a `pull request <https://github.com/Hydrology-IFH/roger/pulls>`_ if you think your modifications could be useful for the Roger community.

.. seealso::

   More information is available in :doc:`our developer guide </tutorial/dev>`.
