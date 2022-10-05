Percolation / capillary rise
============================

Vertical flux :math:`q_{v}` (mm :math:`\Delta t^{-1}`):

.. math::
  q_{v}=\left\{\begin{array}{lr}
  \frac{(z_{sat} / h_a)^{-n}-(h / h_a)^{-n}}{1+(h / h_a)^{-n_{+}}(n-1)(z_{sat} / h_a)^{-n}} & z_{gw} \leq 10 \\
  \frac{(z_{sat} / h_a)^{-n}}{1+(n-1)(z_{sat} / h_a)^{-n}} & z_{gw} > 10
  \end{array}\right.

where :math:`z_{gw}` is the depth of groundwater table (m). For :math:`q_v` :math:`<` 0 soil water
moves in downward direction and for with :math:`q_v` :math:`>` 0 soil water moves in upward direction.


Percolation
-----------

Percolation :math:`q_{perc}` (mm :math:`\Delta t^{-1}`):

.. math::
  q_{perc}=\begin{cases}
  k_s & z_{sat} > 0\\
  q_v \cdot (-1) & q_v < 0 \& z_{sat} = 0 \\
  0 & q_v \geq 0 \& z_{sat} = 0
  \end{cases}

where :math:`z_{sat}` is saturation water level at the soil-bedrock interface (mm).
Percolation might be limited by permeability of bedrock (:math:`k_f`, mm :math:`\Delta t^{-1}`),
if :math:`q_{perc}` exceeds :math:`k_f`.

Saturation water level :math:`z_{sat}` (mm) rises while saturation from top is connected
to the bedrock interface:

.. math::
  z_{sat}=\left\{\begin{array}{lr}
   \frac{S_{lp-ss}}{\theta_{ac}} & \frac{S_{lp-ss}}{\theta_{ac}} \geq z_{nomp}\\
  z_{sat} & \frac{S_{lp-ss}}{\theta_{ac}} < z_{nomp}
  \end{array}\right.

with thickness without macropores :math:`z_{nomp}` (mm):

.. math::
  z_{nomp}=\left\{\begin{array}{lr}
  0 & z_{soil} - l_{mpv} - z_{sat} < 0\\
  z_{soil} - l_{mpv} - z_{sat} & z_{soil} - l_{mpv} - z_{sat} > 0
  \end{array}\right.

where :math:`S_{lp-ss}` is soil water content in large pores of subsoil (mm). :math:`z_{sat}`
is reduced by percolation :math:`q_{perc}`.


Capillary rise
--------------

Capillary rise :math:`q_{cpr}` (mm :math:`\Delta t^{-1}`):

.. math::
  q_{cpr}=\left\{\begin{array}{lr}
  0 & q_v < 0 \\
  q_v & q_v \geq 0 \& z_{sat} = 0
  \end{array}\right.
