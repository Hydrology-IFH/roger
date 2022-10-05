Interception
============
Interception storage is represented by a bucket. The storage is filled by
liquid and solid precipitation and spills if the storage is full.

Interception from upper interception storage
--------------------------------------------


Interception from lower interception storage
--------------------------------------------

.. math::
  \frac{\Delta S_{int-lower}}{\Delta t}=\left\{\begin{array}{lr}
  PREC(i) & PREC(i) \leq S_{tot-int-lower} - S_{int-lower} \\
  S_{tot-int-lower} - S_{int-lower} & PREC(i) > S_{tot-int-lower} - S_{int-lower}
  \end{array}\right.
