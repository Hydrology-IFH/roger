Surface runoff
==============


Hortonian surface runoff
------------------------
Hortonian surface runoff :math:`q_{HOF}` at time step :math:`t` (mm :math:`\Delta t^{-1}`):

.. math::
  q_{HOF} = \begin{cases}
  PREC - INT - INF & PREC - INT - INF > 0 \\
  0 & PREC - INT - INF \leq 0
  \end{cases}



Saturation surface runoff
-------------------------
Saturation surface runoff :math:`q_{HOF}` at time step :math:`t$ (mm :math:`\Delta t^{-1}`):

.. math::
  q_{SOF} = \begin{cases}
  S_{soil} - S_{sat-soil} & S_{soil} - S_{sat-soil} > 0 \\
  0 & S_{soil} - S_{sat-soil} \leq 0
  \end{cases}

where :math:`S_{sat-soil}` is soil water content at saturation (mm) and :math:`S_{soil}`
and soil water content at time step :math:`t` (mm).
