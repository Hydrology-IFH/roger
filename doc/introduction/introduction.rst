A short introduction to Roger
=============================

The vision
----------

1. **Easy to access**: Python modules are simple to install, and projects like `Anaconda <https://www.continuum.io/anaconda-overview>`_ are doing a great job in creating platform-independent environments.
2. **Easy to use**: Anyone with some experience can use their favorite Python tools to set up, control, and post-process Roger.
3. **Easy to modify**: Due to Python's popularity, available abstractions, and dynamic nature, Roger can be extended and modified with relatively little effort.

However, choosing Python over a compiled language like Fortran or C usually comes at a high computational cost. We overcome this gap by using `JAX <https://github.com/google/jax>`_, a framework that can act as a high-performance replacement for NumPy. JAX takes care of all performance optimizations in the background, and runs on both CPUs and GPUs.

Available processes
+++++++++++++++++++

**Interception**:
- rainfall and snowfall interception by vegetation

**Snow**:
- snow accumulation
- snow melt based on day-degree approach
- rain-on-snow

**Infiltration**:
- matrix-driven infiltration based on Green-Ampt approach
- gravity-driven infiltration based on viscous flow approach

**Evaporation**:
- evaporation from interception and surface storage
- soil evaporation based on Stage I (i.e. energy limiting stage) and Stage II (i.e. falling rate stage).

**Transpiration**:
- combination of residual potential evapotranspiration and vegetation-specific coeffcient

**Subsurface Runoff**:
- matrix-driven vertical soil drainage based on Buckingham-Darcy
- gravity-driven infiltration based on viscous flow approach

**Capillary rise**:
- ...

**Groundwater Runoff**:
- ...

**Crop phenology and crop rotation**:
- ...

**Offline transport**:
- ...

Available model structures
+++++++++++++++++++++++++++
Soil hydrologic parameters are approximated by Brooks-Corey scheme.

**SVAT**:
- only vertical processes are considered
- no lateral processes (i.e. no lateral exchange between grid cells)

**SVATCROP**:
- same as SVAT, but crop development (i.e. varying rooting depth and varying canopy cover) is
explicitly represented 

**SVATFILM**:
- ...

**DIST**:
- ...

**DISTCROP**:
- ...

**DISTGROUNDWATER**:
- ...


**TRANSPORT**:
- ...


Diagnostics
+++++++++++

Diagnostics are responsible for handling all model output, runtime checks of the solution, and restart file handling. They are implemented in a modular fashion, so additional diagnostics can be implemented easily. Already implemented diagnostics handle snapshot output, aggregation of variables, and monitoring of mass balance.

For more information, see :doc:`/reference/diagnostics`.


Pre-configured model setups
+++++++++++++++++++++++++++

Roger supports a wide range of model configurations. Several setups are already implemented that highlight some of the capabilities of Roger, and that serve as a basis for users to set up their own configuration: :doc:`/reference/setup-gallery`.


Current limitations
+++++++++++++++++++

Roger is still in development. There are many open issues that we would like to fix later on:

- ...

References
++++++++++

.. [Weiler2005] Weiler, M.: An infiltration model based on flow variability in macropores: development, sensitivity analysis and applications, Journal of Hydrology, 310, 294-315, 2005.

.. [Steduto2008] Steduto, P., Hsiao, T. C., Raes, D., and Fereres, E.: AquaCrop—The FAO Crop Model to Simulate Yield Response to Water: I. Concepts and Underlying Principles, Agronomy Journal, 101, 426-437, 2009.

.. [Stoll2010] Stoll, S. and Weiler, M.: Explicit simulations of stream networks to guide hydrological modelling in ungauged basins, Hydrol. Earth Syst. Sci., 14, 1435-1448, 2010.

.. [Torres2010] Torres, E. A. and Calera, A.: Bare soil evaporation under high evaporation demand: a proposed modification to the FAO-56 model, Hydrological Sciences Journal, 55, 303-315, 2010.

.. [Rinaldo2015] Rinaldo, A., Benettin, P., Harman, C. J., Hrachowitz, M., McGuire, K. J., van der Velde, Y., Bertuzzo, E., and Botter, G.: Storage selection functions: A coherent framework for quantifying how catchments store and release water and solutes, Water Resources Research, 51, 4840-4847, 2015.

.. [Steinbrich2016] Steinbrich, A., Leistert, H., and Weiler, M.: Model-based quantification of runoff generation processes at high spatial and temporal resolution, Environmental Earth Sciences, 75, 1423, 2016.

.. [Germann2018] Germann, P. F. and Prasuhn, V.: Viscous Flow Approach to Rapid Infiltration and Drainage in a Weighing Lysimeter, Vadose Zone Journal, 17, 170020, 2018.
