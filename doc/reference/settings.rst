Constant model parameters
=========================

The following list of available settings is automatically created from the file :file:`settings.py` in the RoGeR main folder.
They are available as attributes of the :class:`RoGeR settings object <roger.state.RogerSettings>`, e.g.: ::

   >>> simulation = MyRogerSetup()
   >>> settings = simulation.state.settings
   >>> print(settings.pi)

.. exec::
  from roger.settings import SETTINGS
  for key, sett in SETTINGS.items():
      print(".. _setting-{}:".format(key))
      print("")
      print(".. py:attribute:: RogerSettings.{} = {}".format(key, sett.default))
      print("")
      print("   {}".format(sett.description))
      print("")
