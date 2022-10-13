Groundwater flow
================


Transmissivity of the aquifer :math:`T_{gw}` (m) at time step t:

.. math::

    T_{gw} =  \int_{z_{gw}}^{Z_{gw-tot}} k_f \cdot exp (-\frac{z_{gw}}{b})dz_{gw}

where :math:`z_{gw}` is the depth of the groundwater table at time step t (m),
:math:`Z_{gw-tot}` is the thickness of the aquifer (m), :math:`k_f` is the
hydraulic conductivity the aquifer at the soil surface (-) and :math:`b` is the
decay coeffcient.


Lateral groundwater flow :math:`q_{gw}` (m :math:`\Delta t^{-1}`) at time step t:

.. math::

    q_{gw} =  T_{gw} \cdot \i_{gw} \cdot w

where :math:`i_{gw}` is the slope of the groundwater table at time step t (-) and
:math:`w` is the width of the flow (m)


Baseflow :math:`q_{bf}` (m :math:`\Delta t^{-1}`) at time step t:

.. math::

    q_{bf} =  k_{f} \cdot \beta \cdot w \cdot z_{bf}

where :math:`z_{bf}` is the thickness of the baseflow (m).
