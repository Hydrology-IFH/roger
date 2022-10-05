Model equations
===============
The model equations are presented here. Each model setup represents specific a
combination of equations. We refer to :doc:`/reference/settings` and
:doc:`/reference/variables` representing the described parameters
and variables as objects (e.g. float, array).

Time-stepping
-------------
Three different time-stepping schemes are available to solve the equations. The
time-stepping depends on the model setup:

- adaptive time-stepping is used for long-term hydrologic modelling (i.e. modelling the water balance over multiple years)
- fixed time-stepping is used for short-term hydrologic modelling (i.e. modelling a single rainfall event)
- fixed time-stepping with internal substepping is used for solute transport models
