Evapotranspiration
==================
The calculation of evapotranspiration requires daily potential evapotranspiration.
Actual evapotranspiration is energy-limited or water-limited, respectively. The
evapotranspiration processes sequentially subtract from available PET.

Evaporation from interception storage
-------------------------------------

Evaporation from interception storage :math:`EVAP_{int}` (mm :math:`\Delta t^{-1}`):

.. math::
  EVAP_{int}=\left\{\begin{array}{lr}
  S_{int} & PET_{res} > S_{int} \\
  PET_{res} & PET_{res} \leq S_{int}
  \end{array}\right.


Soil evaporation
----------------

Soil evaporation :math:`EVAP_{soil}` (mm :math:`\Delta t^{-1}`) implemented with
Stage I-Stage II approach (Or et al., 2013). Threshold between Stage I and Stage
II is defined by readily evaporable water :math:`S_{rew}`. Within Stage I capillary
flow connects to soil surface (i.e. constant evaporation rate) whereas within
Stage II capillary flow collapses (i.e. vapour diffusion rate).

.. math::
  EVAP=PET_{res} \cdot c_{evap}

with

.. math::
  c_{evap}=\begin{cases}
  1 - \frac{f_{ground-cover}}{max(f_{ground-cover})} & EVAP_{d} \leq S_{rew} \\
  1 - \frac{f_{ground-cover}}{max(f_{ground-cover})} \cdot \frac{S_{tew} - EVAP_{d}}{S_{tew} - S_{rew}} & S_{rew} < EVAP_{d} \leq S_{tew} \\
  0 & EVAP_{d} > S_{tew}
  \end{cases}

where :math:`EVAP_{d}` is the cumulated soil evaporation since last rainfall (mm) and
:math:`S_{tew}` is the total evaporable water (mm)

.. math::
  S_{tew}=(\theta_{fc} - 0.5 \cdot \theta_{pwp}) \cdot z_{evap}

.. math::
  S_{rew}=\left\{\begin{array}{lr}
  0.02 & \theta_{pwp} < 0.02\\
  \frac{\theta_{pwp}}{0.24}& \theta_{pwp} \geq 0.02 \&  \theta_{pwp} \leq 0.24\\
  0.24 & \theta_{pwp} > 0.24
  \end{array}\right.

.. math::
  z_{evap}=\frac{S_{rew}}{0.24} \cdot z_{evap-max}

where :math:`z_{evap-max}` is the maximum length of soil capillaries connected
to the soil surface (mm; :math:`z_{evap-max}=150`)

Transpiration
-------------

Transpiration :math:`TRANSP` (mm :math:`\Delta t^{-1}`) with seasonally-variant transpiration coefficients :math:`c_{transp}` and water stress coeffcient of transpiration :math:`c_{ws}`:

.. math::
  TRANSP=PET_{res} \cdot c_{transp}

with

.. math::
  c_{transp}=\left\{\begin{array}{lr}
  \frac{f_{ground-cover}}{max(f_{ground-cover})} & c_{ws} \geq 1 \\
  \frac{f_{ground-cover}}{max(f_{ground-cover})} \cdot c_{ws} & c_{ws} < 1
  \end{array}\right.

.. math::
  c_{ws}=\frac{\theta - \theta_{pwp}}{f_{pwt} \cdot \theta_{fc} - \theta_{pwp}}

where :math:`f_{pwt}` is the fraction of plant water stress (-; :math:`f_{pwt}=0.75`)
