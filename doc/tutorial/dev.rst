Making changes in RoGeR
=======================

Code conventions
----------------

When contributing to RoGeR, please adhere to the following general guidelines:

- Your first guide should be the surrounding RoGeR code. Look around, and be consistent with your modifications.
- Unless you have a very good reason not to do so, please stick to `the PEP8 style guide <https://www.python.org/dev/peps/pep-0008/>`_ throughout your code. One exception we make in RoGeR is in regard to the maximum line length - since numerical operations can take up quite a lot of horizontal space, you may use longer lines if it increases readability.
- In particular, please follow the PEP8 naming conventions, and use meaningful, telling names for your variables, functions, and classes. The variable name :data:`stretching_factor` is infinitely more meaningful than :data:`k`. This is especially important for settings and generic helper functions.
- Document your functions using `Google-style docstrings <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_. This is especially important if you are implementing a user-facing API (such as a diagnostic, a setup, or tools that are meant to be called from setups).
- We use ``flake8`` for linting and ``black`` for code formatting. We automatically validate all changes to RoGeR through a pre-commit hook. To install it, run::

      $ pip install pre-commit
      $ pre-commit install

   After this, black and flake8 will run automatically on every commit.


Distributed memory support
--------------------------

By default, all core routines should support distributed execution via MPI.
In this case, every processor only operates on a chunk of the total data.
By using :py:func:`roger.variables.allocate`, you can make sure that allocated data always has the right shape.

Since none of the processes have access to the global data, you need to take special care during reductions (e.g. ``sum``) and accumulations (e.g. ``cumsum``) along horizontal dimensions.
Use functions from :mod:`roger.distributed` (e.g. :func:`roger.distributed.global_max`) where appropriate.


The dist_safe keyword
+++++++++++++++++++++

If you are not comfortable writing code that is safe for distributed execution, you can use the ``dist_safe`` keyword to :func:`roger.decorators.roger_routine`:::

   @roger_routine(dist_safe=False, local_variables=["temp"])
   def my_function(state):
       # this function is now guaranteed to be executed on the main process
       vs = state.variables

       # since temp is declared as a local variable, we have access to all of the data
       vs.ta = update(vs.ta, at[2:-2, 2:-2], np.max(vs.ta))

       # this would throw an error, since salt is not in local_variables
       # vs.S = vs.S * 0

       # after execution, the updated contents of vs.temp are scattered to all processes,
       # and distributed execution continues

When encountering a ``roger_routine`` that is marked as not safe for distributed execution (``dist_safe=False``), RoGeR gathers all relevant data from the worker processes,
copies it to the main process, and executes the function there.
This ensures that you can write your code exactly as in the non-distributed case (but it comes with a performance penalty).

Running tests and benchmarks
----------------------------

Performance tweaks
------------------

If your changes to RoGeR turn out to have a negative effect on the runtime of the model, there several ways to investigate and solve performance problems:

- Run your model with the ``-v debug``, ``-v trace``, and / or ``--profile-mode`` options to get additional debugging output (such as timings for each time step, and a timing summary after the run has finished).
- You should try and avoid explicit loops over arrays at all cost (but if you have to, you can use :func:`roger.core.operators.for_loop`, which is reasonably efficient in JAX). You should always try to work on the whole array at once.
- If you are still having trouble, don't hesitate to ask for help (e.g. `on GitHub <https://github.com/schwemro/roger/issues>`_).
