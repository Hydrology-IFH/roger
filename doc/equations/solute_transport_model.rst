Solute transport model
======================

Solute transport is implemented with StorAge selection (SAS) functions:

.. math::
  C_Q=\int_{T=0}^{\infty} C_S(T,t) \cdot \alpha_p \cdot \overleftarrow{p}_{Q}(T,t) dT

where :math:`C_{Q}` is the solute concentration of the considered flux at time step :math:`t` (mg :math:`l^{-1}` or ‰),
:math:`C_{S}` is the solute concentration of the considered StorAge at time step :math:`t` (mg :math:`l^{-1}` or ‰),
:math:`\alpha_{p}` is the partition coeffcient (-),
:math:`P_{Q}` is the travel time distribution of the considered flux at time step :math:`t` (-)
and :math:`T` is the water age (days).

Currently, Roger can be used to simulate with the following solutes:
- stable water isotopes
- bromide
- chloride
- nitrate

.. toctree::
   :maxdepth: 3

   solute_transport_model/SAS_functions
   solute_transport_model/biogeochemical_processes
   solute_transport_model/model_structures
