Groundwater flow
================


Transmissivity of groundwater :math:`T_{gw}` (m) at time step t:

.. math::

    T_{gw} =  \int_{z_{gw}}^{Z_{gw-tot}} k_f \cdot exp (-\frac{z_{gw}}{b})dz_{gw}


Lateral groundwater flow :math:`q_{gw}` (m :math:`\Delta t^{-1}`) at time step t:

.. math::

    q_{gw} =  T_{gw} \cdot \beta \cdot w


Baseflow :math:`q_{bf}` (m :math:`\Delta t^{-1}`) at time step t:

.. math::

    q_{bf} =  k_{f} \cdot \beta \cdot w \cdot z_{bf}
