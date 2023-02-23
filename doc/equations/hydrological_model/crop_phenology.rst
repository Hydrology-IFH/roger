Crop phenology
==============

Growing degree day :math:`GDD` at time step :math:`t` (:math:`°C day^{-1}`):

.. math::
  GDD = \begin{cases}
  TA_{ceil} - TA_{base} & \frac{(TA_{max}+TA_{min})}{2} > TA_{ceil} \\
  \frac{(TA_{max}+TA_{min})}{2}]-TA_{base} & TA_{base} \leq \frac{(TA_{max}+TA_{min})}{2} \leq TA_{ceil} \\
  TA_{base} & \frac{(TA_{max}+TA_{min})}{2} < TA_{base}
  \end{cases}

where :math:`TA_{max}` is the daily maximum air temperature (:math:`°C`), :math:`TA_{min}` is the daily minimum air temperature (:math:`°C`),
:math:`TA_{base}` is the lower threshold of crop growth (:math:`°C`) and :math:`TA_{ceil}` is the upper threshold of crop growth (:math:`°C`).


Time since growth (i.e sum of growing degree day) :math:`t_{grow}` (:math:`°C day^{-1}`):

.. math::
  t_{grow}=\sum_{t=t_{sowing}}^{t_{harvesting}} GDD_t

Time since decay (i.e sum of growing degree day) :math:`t_{grow}` (:math:`°C day^{-1}`):

.. math::
  t_{decay}=\sum_{t=t_{0-decay}}^{t_{harvesting}} GDD_t

Coefficient of crop water stress :math:`c_{ws}` (-):

.. math::
  c_{ws}=\begin{cases}
  1 & \theta>\theta_{ws} \\
  \frac{\theta-\theta_{pwp}}{\theta_{ws}-\theta_{pwp}} & \theta_{pwp} \leq \theta \leq \theta_{ws}
  \end{cases}


Growth of crop canopy:

.. math::
  CC = \begin{cases}
  CC_0 \cdot e^{c_{ccg} \cdot c_{ws} \cdot t_{grow}} & CC < CC_{max} / 2  \\
  CC_{max} - (CC_{max} - CC_0) \cdot e^{-(c_{ccg} \cdot c_{ws} \cdot  t_{grow})} & CC \geq CC_{max} / 2 \\
  \end{cases}

where :math:`CC` is the fraction of ground covered by crop canopy (-) at
time step :math:`t`, :math:`CC_0` is the initial fraction of ground covered by crop
canopy (-), :math:`CC_{max}` is the the maximum fraction of ground covered by
crop canopy (-), :math:`c_{ccg}` is the coefficient of crop canopy growth (-) and
:math:`t_{grow}` is the time since crop growth (:math:`°C day^{-1}`).

Decay of crop canopy:

.. math::
  CC=CC_{max} \cdot [1-0.05(e^{\frac{c_{ccd}}{CC_{max}} \cdot t_{decay}} - 1)]

where :math:`c_{ccd}` is the coefficient of crop canopy decay (-) and :math:`t_{decay}`
is the time since crop decay (:math:`°C day^{-1}`).


Crop root growth:

.. math::
  z_{root}=z_{root-max}-(z_{root-max}-z_{root-0}) * e^{c_{crg} \cdot c_{ws} \cdot t_{grow}}

where :math:`z_{root}` is the crop root depth (m) at time step :math:`t`, :math:`z_{root-0}`
is the initial crop root depth (m), :math:`z_{root-max}` is the the maximum crop
root depth (m) and :math:`c_{crg}` is the coefficient of crop root growth (-).


Crop transpiration :math:`TRANSP_{crop}` at time step :math:`t` (:math:`mm \Delta t^{-1}`):

.. math::
  TRANSP_{crop}=PET_{res} * k_{cb} * c_{ws}

.. math::
  k_{cb}=k_{cb-min} + CC \cdot (k_{cb-mid} - k_{cb-min})

where :math:`k_{cb}` is the basal crop coeffcient (-), :math:`k_{cb-min}` is the
minimum basal crop coeffcient (-) and :math:`k_{cb-mid}` is the the basal crop
coeffcient at full canopy development (-).
