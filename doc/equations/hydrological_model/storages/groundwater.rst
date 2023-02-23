Groundwater
===========

Storativity of groundwater :math:`S_{gw}` (m) at time step :math:`t`:

.. math::

    S_{gw} =  \int_{z_{gw}}^{Z_{gw-tot}} n_0 \cdot exp (-\frac{z_{gw}}{b})dz_{gw}

where :math:`z_{gw}` is the depth of the groundwater table at time step :math:`t` (m),
:math:`Z_{gw-tot}` is the thickness of the aquifer (m), :math:`n_0` is the
specific yield of the aquifer at the soil surface (-) and :math:`b` is the
decay coeffcient.
