Soil
====
Soil storage divides into a root zone layer :math:`rz` (i.e. upper soil) and a subsoil
layer :math:`ss` (i.e. lower soil). The two soil layers share the same soil hydraulic
parameters. However, absolute storage values are different due to different thickness of the layers.
Root depth (:math:`z_{root}`; mm) depends on land-use:

Soil hydraulic properties
-------------------------

Soil hydraulic parameters are calculated with the Brooks-Corey scheme (Brooks and Corey, 1966):
Pore size distribution parameter :math:`\lambda`:

.. math::
  \lambda=\frac{1}{\frac{\log(\frac{h_{fc}}{h_{pwp}})}{\log(\frac{\omega_{fc}}{\omega_{pwp}})}}

where :math:`h_{fc}` is soil water potential at field capacity (hPa; :math:`h_{fc}=63`) and :math:`h_{pwp}` is the soil water potential at permanent wilting point (hPa; :math:`h_{pwp}=15850`)

Pore size disconnectedness index :math:`m`:

.. math::
  m=b+\frac{a}{\lambda}

where :math:`a` and :math:`b` are parameters with a fixed value of 2.

Salvucci exponent :math:`n` (Salvucci, 1993):

.. math::
  n=\lambda \cdot a+b


Effective soil water content at field capacity :math:`\omega_{fc}` (-):

.. math::
  \omega_{fc}=\frac{\theta_{fc}}{\theta_{sat}}

where :math:`\theta_{fc}` is soil water content at field capacity (-) and :math:`\theta_{sat}` is soil water content at saturation (-).

Effective soil water content at permanent wilting point :math:`\omega_{pwp}` (-):

.. math::
  \omega_{pwp}=\frac{\theta_{pwp}}{\theta_{sat}}

where :math:`\theta_{pwp}` is soil water content at permanent wilting point (-).

Effective soil water content :math:`\omega` (-):

.. math::
  \omega=\frac{\theta}{\theta_{sat}}


Air entry value :math:`h_a` (i.e. bubbling pressure, hPa):

.. math::
  h_a=\omega_{pwp}^{\frac{1}{\lambda}} \cdot(-1) \cdot h_{pwp}


Soil water potential :math:`h` (hPa):

.. math::
  h=\frac{h_a}{\omega^{\frac{1}{\lambda}}}


Wetting front suction :math:`\psi_f` (mm):

.. math::
  \psi_f=\frac{2+3 \lambda}{(1+3 \lambda) \cdot \frac{h_a}{2} \cdot(-10)}


Hydraulic conductivity :math:`k` (mm h-1):

.. math::
  k=\frac{k_s}{1+\omega^m}

with :math:`k_s` saturated hydraulic conductivity (:math:`m h^{-1}`).


Soil water content at :math:`10^{2.7}` hPa :math:`\theta_{27}` (-):

.. math::
  theta_{27}=\frac{h_a}{-10^{2.7}}^{\lambda_{bc} \cdot \theta_{sat}}


Soil water content at :math:`10^4` hPa :math:`\theta_6` (-):

.. math::
  theta_4=\frac{h_a}{-10^4}^{\lambda_{bc} \cdot \theta_{sat}}


Soil water content at :math:`10^6` hPa :math:`\theta_6` (-):

.. math::
  theta_6=\frac{h_a}{-10^6}^{\lambda_{bc} \cdot \theta_{sat}}


Soil moisture deficit :math:`\Delta \theta` (-):

.. math::
  \Delta \theta=\theta_{sat} - \theta_{rz}
