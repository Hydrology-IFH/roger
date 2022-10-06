Infiltration
============

At the onset of rainfall or snow melt, we calculate event-specific parameters (e.g.
soil moisture deficit :math:`\Delta \theta`). For each event, we use two wetting
fronts (:math:`wf1` and :math:`wf2`). The second wetting front is active after a rainfall pause
(i.e. calculation of event-specific parameters of :math:`wf2`). :math:`wf2` is active while
wetting front depth of :math:`wf2` is less than wetting front depth of :math:`wf1`. In the
following are the equations applied for dual-wetting front approach.

Total infiltration :math:`INF` at time step :math:`t` (mm :math:`\Delta t^{-1}`):

.. math::
INF=INF_{mat }+INF_{mp}+INF_{sc}


Matrix infiltration
-------------------
Determining interval (:math:`i_s`) when rainfall exceeds infiltrability within the current event:

.. math::
  i_s=\left\{\begin{array}{lr}
  no value & (PREC(i)-k_s \cdot \Delta t) \cdot \sum_{i=1}^i PREC(i) \leq k_s \cdot \Delta t \cdot \Delta \theta \cdot \psi_f \\
  i & (PREC(i)-k_s \cdot \Delta t \cdot \sum_{i=1}^i PREC(i)>k_s \cdot \Delta t \cdot \Delta \theta \cdot \psi_f
  \end{array}\right.

where :math:`PREC` is precipitation (mm :math:`\Delta t^{-1}`), :math:`k_s` is the saturated hydraulic conductivity of the soil matrix, :math:`\Delta \theta` is the
soil moisture deficit (-) and :math:`\psi_f` is the wetting front suction (mm)

Threshold rainfall intensity :math:`PREC_{gr}` (mm :math:`\Delta t^{-1}`):

.. math::
  PREC_{gr}=k_s \cdot \Delta t \cdot\left(\frac{\Delta \theta \cdot \psi_f}{\sum_{i=1}^{i_s-1} PREC(i)}+1\right)

.. math::
  i_s=\left\{\begin{array}{lr}
  (i_s - 1) \cdot \Delta t & PREC(i_s) < PREC_{gr} \\
  (i_s-1) \cdot \Delta t+\frac{k_s \cdot \Delta t \cdot \Delta \theta \cdot \psi_f}{PREC(i_s) \cdot(PREC(i_s)-k_s \cdot \Delta t)}\frac{\Delta t}{PREC(i_s)} \sum_{v=1}^{i_s-1} PREC_v & PREC(i_s) \geq PREC_{gr}\end{array}\right.


Infiltration at time step of saturation :math:`F_s` (mm :math:`\Delta t^{-1}`):

.. math::
  F_s=\frac{k_s \cdot \Delta t \cdot \theta_d \cdot \psi_f}{PREC(i_s)-k_s \cdot \Delta t}


Matrix infiltration :math:`INF_{mat}` at time step :math:`t` (mm :math:`\Delta t^{-1}`):

.. math::
  INF_{mat}=\left\{\begin{array}{lr}
  z_0 \cdot & z_0 \leq INF_{mp-pot} \\
  INF_{mat-pot} & z_0 > INF_{mp-pot} \\
  \end{array}\right.

where :math:`z_0` is the surface ponding (mm; i.e. residual rainfall after interception or
snow melt).

with potential matrix infiltration at time step :math:`t` :math:`INF_{mat-pot}` (mm :math:`\Delta t^{-1}`):

.. math::
  INF_{mat-pot}=\left\{\begin{array}{lr}
  PREC(t) & t_s \geq t \\
  PREC(t) \cdot(t_s-t-\Delta t)+\frac{k_s}{2}(1+\frac{1+\frac{2B}{A}}{\sqrt{1+\frac{4B}{A}+\frac{4 F_s^2}{A^2}}}) & t - \Delta t<t_S<t \\
  \frac{k_s}{2}(1+\frac{1+\frac{2 B}{A}}{\sqrt{1+\frac{4 B}{A}+\frac{4 F_s^2}{A^2}}}) & t_s < t
  \end{array}\right.


with auxiliary variables:

.. math::
  A=K_S \cdot\left(t-t_s\right)

.. math::
  B=F_s+2 \cdot \Delta \theta \cdot \psi_f


Wetting front depth :math:`z_{wf}` (mm):

.. math::
  z_{wf}=\frac{\sum_i^i INF_{mat}(i)}{\Delta \theta}

where :math:`i_e` is interval of the event start.


Macropore infiltration
----------------------

Macropore infiltration :math:`INF_{mp}` at time step :math:`t` (mm :math:`\Delta t^{-1}`; Weiler, 2005):

.. math::
  INF_{mp}=\left\{\begin{array}{lr}
  z_0 \cdot (1 - e^{-(\frac{\rho_{mpv}}{82})^{0.887}}) & 0 < z_0 \cdot (1 - e^{-(\frac{\rho_{mpv}}{82})^{0.887}}) \leq INF_{mp-pot} \\
  INF_{mp-pot} & z_0 \cdot (1 - e^{-(\frac{\rho_{mpv}}{82})^{0.887}}) > INF_{mp-pot} \\
  \end{array}\right.

where :math:`z_0` is the surface ponding (mm; i.e. matrix infiltration excess).

with potential macropore infiltration :math:`INF_{mp-pot}` at time step :math:`t` (mm :math:`\Delta t^{-1}`)

.. math::
  INF_{mp-pot}=\pi \cdot(y_{mp}(t)^2-y_{mp}(t-\Delta t)^2) \cdot \rho_{mpv} \cdot \frac{\Delta z_{mp} \cdot \Delta \theta}{\Delta t}

where :math:`\rho_{mpv}` is density of vertical macropores (:math:`m^2`) and :math:`\Delta z_{mp}` depth of non-saturated macropore (mm)

Radial distance of the macropore wetting front :math:`y_{mp}` (mm):

.. math::
  y_{mp}=\frac{1}{2} \cdot \frac{b^{(1 / 3)}}{\Delta \theta}+\frac{1}{2} \cdot \frac{a}{b^{(1 / 3)}}+\frac{1}{2} \cdot r_{mp}

.. math::
  a=\Delta \theta \cdot r_{mp}^2

.. math::
  b=r \cdot \Delta \theta \cdot(12 c-a+2 \sqrt{6} \cdot \sqrt{c \cdot(6 c-a)}

.. math::
  c=t_{mp} \cdot k_s \cdot \psi_s


Duration of macropore infiltration :math:`t_{mp}` (:math:`y_{mp}=r_{mp}` at time t=0)

.. math::
  t_{mp}=\frac{\Delta \theta}{k_s \cdot \Psi_s \cdot r_{mp}} \cdot(\frac{y_{mp}^3}{3}-\frac{y_{mp}^2 r}{2}-\frac{r_{mp}^3}{6})

where :math:`r_{mp}` is the radius of the macropore (mm; :math:`r_{mp}`=2.5). Macropore
infiltration stops if :math:`z_{wf}` is greater than :math:`l_{mpv}`.



Shrinkage crack infiltration
----------------------------

Shrinkage crack infiltration :math:`INF_{cs}` at time step :math:`t` (mm :math:`\Delta t^{-1}`; Steinbrich et al., 2016):

.. math::
  INF_{sc}=\left\{\begin{array}{lr}
  z_0 & z_0 \leq INF_{sc-pot} \\
  INF_{sc-pot} & z_0 > INF_{sc-pot} \\
  \end{array}\right.

where :math:`z_0` is the surface ponding (mm; i.e. macropore infiltration excess).

Potential shrinkage crack infiltration :math:`INF_{sc-pot}` at time step :math:`t` (mm :math:`\Delta t^{-1}`; Steinbrich et al., 2016):

.. math::
  INF_{sc-pot}=2 \cdot l_{sc} \cdot(y_{sc}(t)-y_{sc}(t-\Delta t)) \cdot \frac{\Delta z_{sc} \cdot \Delta \theta}{\Delta t}

where :math:`l_{sc}` is the horizontal length of shrinkage cracks (mm :math:`m^{-2}`) and :math:`\Delta z_{sc}` is the depth of non-saturated shrinkage crack (mm)

Horizontal distance of the shrinkage crack wetting front :math:`y_{sc}` (mm):

.. math::
  y_{sc}(t)=\sqrt{\frac{2 \cdot k_s \cdot \Psi_s \cdot t_{sc}}{\Delta \theta}}

.. math::
  t_{sc}=\frac{y_{sc}(t-\Delta t)^2 \cdot \Delta \theta}{2 \cdot k_s \cdot \psi_s}


Calculation of depth of shrinkage cracks :math:`z_{sc}` at beginning of event:

.. math::
  z_{sc} = \begin{cases}
  700 \cdot clay & \theta_{rz} < \theta_{4} \\
  700 \cdot clay \cdot (1 - \frac{\theta_{rz}}{\theta_{27} - \theta_{4}}) & \theta_{4} \leq \theta_{rz} \leq \theta_{27} \\
  0 & \theta_{rz} > \theta_{27}
  \end{cases}

with clay content of soil :math:`clay` (-)

.. math::
  clay=\frac{clay_{max} \cdot (\theta_6 - clay_{min})}{0.3}

where :math:`clay_{min}` is the lower limit of clay content (-; :math:`clay_{min}`=0.01) and :math:`clay_{max}` is the upper limit of clay content
(-; :math:`clay_{max}`=0.71). :math:`INF_{sc}` occurs only if shrinkage cracks are available and stops if :math:`z_{wf}`
is greater than :math:`z_{sc}`.
