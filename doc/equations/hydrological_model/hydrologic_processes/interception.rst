Interception
============
Interception storage is represented by a bucket. The storage is filled by
liquid and solid precipitation and spills if the storage is full.

Interception from upper interception storage
--------------------------------------------

Interception by upper interception storage :math:`INT_{upper}` at time step :math:`t` (mm :math:`\Delta t^{-1}`):

.. math::
  INT_{upper}=\left\{\begin{array}{lr}
  PREC \cdot (1 - c_{throughfall}) & PREC \cdot (1 - c_{throughfall}) \leq S_{tot-int-upper} - S_{int-upper} \\
  S_{tot-int-upper} - S_{int-upper} & PREC \cdot (1 - c_{throughfall}) > S_{tot-int-upper} - S_{int-upper}
  \end{array}\right.

where :math:`PREC` is precipitation (mm :math:`\Delta t^{-1}`), :math:`c_{throughfall}` is the throughfall coeffcient of the canopy, :math:`S_{int-upper}` is the
storage volume of the upper interception storage at time step :math:`t` (mm) and :math:`S_{tot-int-upper}` is the  available storage volume of the upper interception storage (mm).


Interception from lower interception storage
--------------------------------------------

Interception by lower interception storage :math:`INT_{lower}` at time step :math:`t` (mm :math:`\Delta t^{-1}`):

.. math::
  INT_{lower}=\left\{\begin{array}{lr}
  PREC & PREC \leq S_{tot-int-lower} - S_{int-lower} \\
  S_{tot-int-lower} - S_{int-lower} & PREC > S_{tot-int-lower} - S_{int-lower}
  \end{array}\right.

where :math:`PREC` is precipitation (mm :math:`\Delta t^{-1}`), :math:`S_{int-lower}` is the
storage volume of the lower interception storage at time step :math:`t` (mm) and :math:`S_{tot-int-lower}` is the available storage volume of the lower interception storage (mm).
