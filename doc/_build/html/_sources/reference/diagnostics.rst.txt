.. _diagnostics:

Diagnostics
===========

Diagnostics are separate objects (instances of subclasses of :class:`RogerDiagnostic`)
responsible for handling I/O, restart mechanics, and monitoring of the numerical
solution. All available diagnostics are instantiated and added to a dictionary
attribute :attr:`RogerState.diagnostics` (with a key determined by their `name` attribute).
Options for diagnostics may be set during the :meth:`RogerSetup.set_diagnostics` method:

Output aggregated to daily:

::

   class MyModelSetup(RogerSetup):
       ...
       def set_diagnostics(self, state):
           diagnostics = state.diagnostics
           diagnostics['rate'].output_variables = ['transp', 'q_ss']
           diagnostics['rate'].sampling_frequency = 1
           diagnostics['rate'].output_frequency =  24 * 60 * 60

Output aggregated to hourly:

::

  class MyModelSetup(RogerSetup):
      ...
      def set_diagnostics(self, state):
          diagnostics = state.diagnostics
          diagnostics['rate'].output_variables = ['transp', 'q_ss']
          diagnostics['rate'].sampling_frequency = 1
          diagnostics['rate'].output_frequency =  60 * 60


Output aggregated to 10 minutes:

::

  class MyModelSetup(RogerSetup):
      ...
      def set_diagnostics(self, state):
          diagnostics = state.diagnostics
          diagnostics['rate'].output_variables = ['transp', 'q_ss']
          diagnostics['rate'].sampling_frequency = 1
          diagnostics['rate'].output_frequency =  10 * 60

Please not, that within the adaptive-time stepping greater time steps
than provided `output_frequency` will not be downscaled (e.g. from daily to hourly).

Base class
----------

This class implements some common logic for all diagnostics. This makes it easy
to write your own diagnostics: Just derive from this class, and implement the
virtual functions.

.. autoclass:: roger.diagnostics.base.RogerDiagnostic
   :members: name, initialize, diagnose, output

Available diagnostics
---------------------

Currently, the following diagnostics are implemented and added to
:obj:`RogerState.diagnostics`:

Snapshot
++++++++

.. autoclass:: roger.diagnostics.snapshot.Snapshot
   :members: name, output_variables, sampling_frequency, output_frequency, output_path

Average
++++++++

.. autoclass:: roger.diagnostics.average.Average
   :members: name, output_variables, sampling_frequency, output_frequency, output_path

Minimum
+++++++

.. autoclass:: roger.diagnostics.minimum.Minimum
  :members: name, output_variables, sampling_frequency, output_frequency, output_path

Maximum
+++++++

.. autoclass:: roger.diagnostics.maximum.Maximum
  :members: name, output_variables, sampling_frequency, output_frequency, output_path

Rate
+++++++++++

.. autoclass:: roger.diagnostics.rate.Rate
   :members: name, output_variables, sampling_frequency, output_frequency, output_path

Collect
+++++++
.. autoclass:: roger.diagnostics.collect.Collect
   :members: name, output_variables, sampling_frequency, output_frequency, output_path

Constant
++++++++
.. autoclass:: roger.diagnostics.constant.Constant
  :members: name, output_variables, sampling_frequency, output_frequency, output_path

Water monitor
+++++++++++++

.. autoclass:: roger.diagnostics.water_monitor.WaterMonitor
 :members: name, sampling_frequency, output_frequency

Tracer monitor
++++++++++++++

.. autoclass:: roger.diagnostics.tracer_monitor.TracerMonitor
  :members: name, sampling_frequency, output_frequency
