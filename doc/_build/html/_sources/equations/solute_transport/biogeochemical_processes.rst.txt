Biogeochemical processes
========================

Soil temperature
----------------

Soil temperature :math:`TA_{soil}` (°C) at time step t:

.. math::
  TA_{soil}=TA_{year} + (TA - TA_{year}) \cdot \frac{sin(\omega t + \omega \Phi -\frac{0.5 \cdot z_{soil}}{D})}{e^\frac{-0.5 \cdot z_{soil}}{D}}

with

.. math::
  \omega = \frac{2\pi}{365}

where :math:`TA_{year}` is the annual average air temperature (°C),
:math:`TA` is the daily air temperature (°C) at time step t, :math:`D`
is the dampening depth (mm), t is the day of year and :math:`\Phi` is a constant
offset (days).

Nitrogen cycle
--------------

.. math::
  \frac{\partial NO_{3-soil}(T,t)}{\partial t} + \frac{\partial NO_{3-soil}(T,t)}{\partial T} = 0.3 \cdot N_{fert}(t) + Nit_{soil}(t) - Denit_{soil}(t) - N_{up}(t)

.. math::
  \frac{\partial Nmin(T,t)}{\partial t} + \frac{\partial Nmin(T,t)}{\partial T} = 0.7 \cdot N_{fert}(t) + Min_{soil}(t) - Nit_{soil}(t)

Denitrification
---------------

Denitrification rate of the soil :math:`Denit_{soil}` (:math:`kg N ha^{-1} year^{-1}`) at time step t:

.. math::
  Denit_{soil}=\begin{cases}
  (Denit_{soil-max} \cdot \frac{NO_{3-soil}(T,t)}{k_{denit-soil} + NO_{3-soil}(T,t)}) \cdot c_{denit} & S_{t} \geq 0.7 \cdot S_{sat}\\
  0 & S_{t} < 0.7 \cdot S_{sat}
  \end{cases}

with

.. math::
  c_{denit}=\begin{cases}
  \frac{TA_{soil}}{50 - 5} & 5 \leq TA_{soil} \leq 50 \\
  0 & 5 > TA_{soil} \lor TA_{soil} > 50
  \end{cases}

where :math:`Denit_{soil-max}` is the maximum dentrification rate (:math:`kg N ha^{-1} year^{-1}`),
:math:`k_{denit-soil}` is the Michaelis constant (:math:`kg N ha^{-1} year^{-1}`),
:math:`NO_{3-soil}` is the age-ranked nitrate nitrogen storage of the soil (:math:`kg NO_{3}-N`),
:math:`S` is the storage volume (mm) at time step t and
:math:`S_{sat}` is the storage volume at saturation (mm).

Denitrification rate of the groundwater :math:`Denit_{gw}` (:math:`kg N ha^{-1} year^{-1}`) at time step t:

.. math::
  Denit_{gw}=Denit_{gw-max} \cdot \frac{NO_{3-gw}(T,t)}{k_{denit-gw} + NO_{3-gw}(T,t)}

where :math:`Denit_{gw-max}` is the maximum dentrification rate (:math:`kg N ha^{-1} year^{-1}`),
:math:`k_{denit-gw}` is the Michaelis constant (:math:`kg N ha^{-1} year^{-1}`),
:math:`NO_{3-gw}` is the age-ranked nitrate nitrogen storage of the groundwater (:math:`kg NO_{3}-N`),

Nitrification
-------------

Nitrification rate of the soil :math:`Denit_{soil}` (:math:`kg N ha^{-1} year^{-1}`) at time step t:

.. math::
  Nit_{soil}=\begin{cases}
  Nit_{soil}=(Nit_{soil-max} \cdot \frac{Nmin(T,t)}{k_{nit} + Nmin(T,t)}) \cdot c_{nit} & S_{t} < 0.7 \cdot S_{sat}\\
  0 & S_{t} > 0.7 \cdot S_{sat}
  \end{cases}

with

.. math::
  c_{nit}=\begin{cases}
  \frac{TA_{soil}}{50 - 5} & 5 \leq TA_{soil} \leq 50 \\
  0 & 5 > TA_{soil} \lor TA_{soil} > 50
  \end{cases}

where :math:`Nit_{soil-max}` is the maximum nitrification rate (:math:`kg N ha^{-1} year^{-1}`),
:math:`k_{nit-soil}` is the Michaelis constant (:math:`kg N ha^{-1} year^{-1}`) and
:math:`Nmin` is the age-ranked mineral nitrogen storage of the soil (:math:`kg N`).

Nitrogen mineralization
-----------------------

Nitrogen mineralization rate of the soil :math:`Min_{soil}` (:math:`kg N ha^{-1} year^{-1}`) at time step t:

.. math::
  Min_{soil}=k_{min} \cdot c_{min}

with

.. math::
  c_{min}=\begin{cases}
  \frac{TA_{soil}}{50 - 5} & 5 \leq TA_{soil} \leq 50 \\
  0 & 5 > TA_{soil} \lor TA_{soil} > 50
  \end{cases}

where :math:`k_{min}` is the maximum nitrogen mineralization rate of the soil (:math:`kg N ha^{-1} year^{-1}`),


Nitrogen uptake by crops
------------------------

Nitrogen uptake by crops :math:`N_{up}` (mg :math:`\Delta t^{-1}`) at time step t:

.. math::
  N_{up}= transp \cdot \int_{T=0}^{\infty} C_S(T,t) \cdot \alpha_p \cdot \overleftarrow{p}_{transp}(T,t) dT
