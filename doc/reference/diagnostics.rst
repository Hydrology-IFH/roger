.. _diagnostics:

Diagnostics
===========

Diagnostics are separate objects (instances of subclasses of :class:`RogerDiagnostic`)
responsible for handling I/O, restart mechanics, and monitoring of the numerical
solution. All available diagnostics are instantiated and added to a dictionary
attribute :attr:`RogerState.diagnostics` (with a key determined by their `name` attribute).
Options for diagnostics may be set during the :meth:`RogerSetup.set_diagnostics` method:

::

   class MyModelSetup(RogerSetup):
       ...
       def set_diagnostics(self, state):
           diagnostics = state.diagnostics
           diagnostics['rates'].output_variables = ['transp','q_ss']
           diagnostics['rates'].sampling_frequency = 1
           diagnostics['rates'].sampling_frequency =  24 * 60 * 60

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

Averages
++++++++

.. autoclass:: roger.diagnostics.averages.Averages
   :members: name, output_variables, sampling_frequency, output_frequency, output_path

Minimum
+++++++

.. autoclass:: roger.diagnostics.minimum.Minimum
  :members: name, output_variables, sampling_frequency, output_frequency, output_path

Maximum
+++++++

.. autoclass:: roger.diagnostics.maximum.Maximum
  :members: name, output_variables, sampling_frequency, output_frequency, output_path

Rates
+++++++++++

.. autoclass:: roger.diagnostics.rates.Rates
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
