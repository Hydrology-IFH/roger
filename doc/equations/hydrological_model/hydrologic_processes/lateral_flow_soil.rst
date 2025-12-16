Subsurface runoff
=================

Lateral subsurface runoff via macropores
----------------------------------------

Flow velocities of fast subsurface flow via macropores :math:`v_{mp}(j)` (m :math:`h^{-1}`, [Steinbrich2016]_):

.. image:: /_images/lateral_macropore_flow_velocity.png
   :width: 500
   :align: center

Potential lateral subsurface flow via macropores :math:`SSF_{mp-pot}` (:math:`m^3 h^{-1}`):

.. math::
  SSF_{mp-pot}=\sum_{j=1}^8 z_{sat}(j) \cdot v_{mp}(i) \cdot \rho_{mph} \cdot A_{mp}

with :math:`i` slope (:math:`m m^{-1}`), :math:`j` horizon, :math:`z_{sat}(j)` saturated thickness of the jth horizon (m), :math:`v_{mp}(i)`slope dependent velocity in the jth horizon (m/h), :math:`\rho_{mph}` density of slope parallel macropores (:math:`m^{-2}`) and :math:`A_{mp}` cross-sectional area of one macropore (:math:`m^2`)


Lateral subsurface runoff via soil matrix
-----------------------------------------


Potential lateral subsurface flow via macropores :math:`SSF_{mp-pot}` (:math:`m^3 h^{-1}`):

.. math::
  SSF_{mat-pot}= z_{sat} \cdot k_s \cdot i \cdot w

with :math:`w` width of the flow (m), :math:`z_{sat}` saturated thickness (m) and :math:`k_s` saturated hydraulic conductivity (:math:`m h^{-1}`).
