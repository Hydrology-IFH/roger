A short introduction to Roger
=============================

The model architecture
----------------------

1. **Easy to access**: Python modules are simple to install, and projects like `Anaconda <https://www.continuum.io/anaconda-overview>`_ are doing a great job in creating platform-independent environments.
2. **Easy to use**: Anyone with some experience can use their favorite Python tools to set up, control, and post-process Roger.
3. **Easy to modify**: Due to Python's popularity, available abstractions, and dynamic nature, Roger can be extended and modified with relatively little effort.

However, choosing Python over a compiled language like Fortran or C usually comes at a high computational cost. We overcome this gap by using `JAX <https://github.com/google/jax>`_, a framework that can act as a high-performance replacement for NumPy. JAX takes care of all performance optimizations in the background, and runs on CPUs and GPUs.

Available processes
+++++++++++++++++++

Soil hydraulic parameters are approximated by the Brooks-Corey scheme.

**Interception**:

- rainfall and snowfall interception by vegetation

**Snow**:

- snow accumulation
- snow melt based on day-degree approach
- rain-on-snow

**Infiltration**:

- matrix-driven infiltration based on Green-Ampt approach ([Weiler2005]_)
- gravity-driven infiltration based on viscous flow approach ([Germann2018]_)

**Evaporation**:

- evaporation from interception and surface storage
- soil evaporation based on Stage I (i.e. energy limiting stage) and Stage II (i.e. falling rate stage) ([Torres2010]_).

**Transpiration**:

- combination of residual potential evapotranspiration and vegetation-specific coeffcient

**Subsurface Runoff**:

- gravity-driven infiltration based on viscous flow approach ([Germann2018]_)
- lateral subsurface runoff in the soil as described in [Steinbrich2016]_

**Percolation**:

- capillary-driven vertical drainage ([Salvucci1993]_)
- gravity-driven vertical drainage based on viscous flow approach ([Germann2018]_)

**Capillary rise**:

- capillary-driven vertical uplift ([Salvucci1993]_)

**Groundwater flow**:

- spatial explicit representation of shallow groundwater following [Stoll2010]_

**Crop phenology and crop rotation**:

- time-varying crop canopy cover and crop root depth is implemented as in [Steduto2009]_

**Offline transport**:

- StorAge selection (SAS) functions ([Rinaldo2015]_) are used to calculate travel time distributions, residence time distribution and solute concentrations


Available model structures
+++++++++++++++++++++++++++

**SVAT**:

- only vertical processes are considered
- no lateral processes (i.e. no lateral exchange between grid cells)

**SVAT-CROP**:

- same as SVAT, but crop phenology (i.e. varying rooting depth and varying canopy cover) is explicitly represented

**ONED**:

- vertical and lateral processes are considered

**ONED-EVENT**:

- vertical and lateral processes are considered
- simulation of a single event

**SVAT_TRANSPORT**:

- calculates offline coupled solute transport based on the hydrologic simulations from the SVAT model


Diagnostics
+++++++++++

Diagnostics are responsible for handling all model output, sanity checks of the solution, and restart file handling. They are implemented in a modular fashion, so additional diagnostics can be implemented easily. Already implemented diagnostics handle snapshot output, aggregation of variables, and monitoring of mass balance.

For more information, see :doc:`/reference/diagnostics`.


Pre-configured model setups
+++++++++++++++++++++++++++

Roger supports a wide range of pre-configured models. Several setups are already implemented that highlight some of the capabilities of Roger, and that serve as a basis for users to set up their own configuration: :doc:`/reference/model-gallery`.


Current limitations
+++++++++++++++++++

Roger is still in development. There are many open issues that we would like to fix later on:

- ...

References
++++++++++

.. [Salvucci1993] Salvucci, G. D.: An approximate solution for steady vertical flux of moisture through an unsaturated homogeneous soil, Water Resources Research, 29, 3749-3753, 1993.

.. [Weiler2005] Weiler, M.: An infiltration model based on flow variability in macropores: development, sensitivity analysis and applications, Journal of Hydrology, 310, 294-315, 2005.

.. [Steduto2009] Steduto, P., Hsiao, T. C., Raes, D., and Fereres, E.: AquaCrop—The FAO Crop Model to Simulate Yield Response to Water: I. Concepts and Underlying Principles, Agronomy Journal, 101, 426-437, 2009.

.. [Stoll2010] Stoll, S. and Weiler, M.: Explicit simulations of stream networks to guide hydrological modelling in ungauged basins, Hydrol. Earth Syst. Sci., 14, 1435-1448, 2010.

.. [Torres2010] Torres, E. A. and Calera, A.: Bare soil evaporation under high evaporation demand: a proposed modification to the FAO-56 model, Hydrological Sciences Journal, 55, 303-315, 2010.

.. [Rinaldo2015] Rinaldo, A., Benettin, P., Harman, C. J., Hrachowitz, M., McGuire, K. J., van der Velde, Y., Bertuzzo, E., and Botter, G.: Storage selection functions: A coherent framework for quantifying how catchments store and release water and solutes, Water Resources Research, 51, 4840-4847, 2015.

.. [Steinbrich2016] Steinbrich, A., Leistert, H., and Weiler, M.: Model-based quantification of runoff generation processes at high spatial and temporal resolution, Environmental Earth Sciences, 75, 1423, 2016.

.. [Germann2018] Germann, P. F. and Prasuhn, V.: Viscous Flow Approach to Rapid Infiltration and Drainage in a Weighing Lysimeter, Vadose Zone Journal, 17, 170020, 2018.
