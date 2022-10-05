Snow accumulation and snow melt
===============================

Solid precipitation accumulates in the interception storage and at the land
surface if air temperatures are below 0 °C. Snow melt occurs for air
temperatures above 0 °C and is based on degree-day approach. Snow melt runoff
is initiated if liquid storage of the snow cover (:math:`S_{snow-l}``) exceeds
the retention capacity of the snow cover :math:`S_{snow-ret}` (mm):

.. math::
  S_{snow-ret}=\frac{10000}{\frac{100-r_{max}}{100}} \cdot  swe

where :math:`r_{max}` is the retention factor of the snow cover (%) and
:math:`swe` is the snow water equivalent of the snow cover (mm).

Snow melt :math:`q_{snow}` (mm) is calculated as:

.. math::
  q_{snow}=s_f \cdot (TA-TA_m) \cdot \Delta t

where :math:`s_f` is the degree-day factor (mm °:math:`C^{-1}` :math:`h^{-1}`),
:math:`TA` is the air temperature (°C), :math:`TA_m` is equal to
0 °C and :math:`\Delta t` is time step (h).
