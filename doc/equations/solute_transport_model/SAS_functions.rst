StorAge selection (SAS) functions
=================================

.. image:: /_images/SAS_functions.png
   :width: 300
   :align: center

Travel time distributions are calculated with fractional SAS functions
(-; see [vanderVelde2012]_):

.. math::
  \overleftarrow{p}_{Q}(T, t)=\frac{\partial}{\partial T} \Omega_Q(P_S(T, t), t)

with

.. math::
  P_S(T, t)=\frac{S_T(T, t)}{S(t)}

where :math:`T` is the water age, :math:`t` is the time step, is the backward travel time
distribution of a specific hydrologic flux, :math:`Q(T,t)` is the probability
distribution of the hydrologic flux (where :math:`Q(T,t)` is the cumulative probability distribution),
:math:`S_T(T,t)` is the cumulative age-ranked storage (mm), :math:`S(t)` is the mobile storage volume
(mm; i.e. storage volume below permanent wilting point is not considered) and
:math:`P_S (T,t)` is the cumulative probability distribution of the storage (where :math:`p_S (T,t)` is the probability distribution).

Uniform
-------
The uniform distribution function has no age preference.

.. math::
  \Omega_Q(T,t)=P_S(T,t)

Power
-----
The power distribution function provides flexibility to
represent a preference for younger water (:math:`k < 1`)
or preference for older water (:math:`k > 1`).

.. math::
  \Omega_Q(T,t)=P_S(T,t)^k

Kumaraswamy
-----------
The Kumaraswamy distribution function (Kumaraswamy, 1980) provides flexibility to
represent a preference for younger water (:math:`\alpha_Q = 1` and :math:`\beta_Q > 1`)
or preference for older water (:math:`\alpha_Q > 1` and :math:`\beta_Q = 1`).

.. math::
  \Omega_Q(T,t)=1-((1-(P_S(T,t))^{\alpha_Q})^{\beta_Q})

Exponential
-----------

.. math::
  \Omega_Q(T,t)=1-e^{-k \cdot (P_S(T,t)}

Dirac
-----

.. math::
  \Omega_Q(T,t)= \begin{cases}0, & T \leq T_{dirac} \\
  1, & T > T_{dirac} \end{cases}

where :math:`T_{dirac}` is the water age of the pulse.
Please note, that a closed form of :math:`P_Q` using the Dirac distribution
is not available.


Gamma
-----

.. math::
  \Omega_Q(T,t)=\frac{\gamma(\alpha, \beta \cdot P_S(T,t)}{\Gamma(\alpha)}

where :math:`\gamma` is the regularized lower incomplete gamma function.
Please note, that a closed form of :math:`P_Q` using the Gamma distribution
function is not available (see [Harman2015]_).

Exponential
-----------

.. math::
  \Omega_Q(T,t)=1-e^{-k \cdot (P_S(T,t)}


Time-variant SAS function parameters
------------------------------------

SAS function parameters can be time-variant. For example, time-variant may be
described by a linear relationship of the storage volume:

.. math::
  b_Q(t)=c_1+c_2 \cdot (\frac{S(t)}{S_{sat}-S_{pwp}})

Short description of SAS parameterization
-----------------------------------------
SAS parameters are defined in `sas_params_q` where `_q` corresponds
to the flux e.g. transp.

- `1`: Uniform SAS function
- `2`: Dirac SAS function
- `3`: Kumaraswamy SAS function
- `31`: Kumaraswamy SAS function with time-variant preference for younger water
- `32`: Kumaraswamy SAS function with time-variant preference for older water
- `35`: Kumaraswamy SAS function with time-variant preference (e.g. preference for younger water while wetter conditions and preference for older water while drier conditions)
- `36`: Kumaraswamy SAS function with time-variant parameter `a` and constant parameter `b`
- `37`: Kumaraswamy SAS function with time-variant with time-variant parameter `b` and constant parameter `a`
- `4`: Gamma SAS function
- `5`: Exponential SAS function
- `6`: Power SAS function

The array of `sas_params_q` encompasses eight dimensions:
- First array dimension of `sas_params_q` contains SAS function type (e.g. 1)
- Second array dimension of `sas_params_q` contains first SAS parameter (only considered if SAS function type is Kumaraswamy, Gamma, Exponential or Power)
- Third array dimension of `sas_params_q` contains second SAS parameter (only considered if SAS function type is Kumaraswamy or Gamma)
- Fourth array dimension of `sas_params_q` contains lower boundary for temporal variation of SAS parameter (only considered if SAS function type is Kumaraswamy, Gamma, Exponential or Power)
- Fifth array dimension of `sas_params_q` contains upper boundary for temporal variation of SAS parameter (only considered if SAS function type is Kumaraswamy, Gamma, Exponential or Power)
- Sixth array dimension of `sas_params_q` contains lower boundary of storage used for temporal variation of SAS parameter (e.g. 200 mm; only considered if SAS function type is Kumaraswamy, Gamma, Exponential or Power)
- Seventh array dimension of `sas_params_q` contains upper boundary of storage used for temporal variation of SAS parameter (e.g. 400 mm; only considered if SAS function type is Kumaraswamy, Gamma, Exponential or Power)
