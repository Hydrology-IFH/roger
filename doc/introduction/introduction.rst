A short introduction to Roger
=============================

The model architecture
----------------------

We adapted the model architecture from [Haefner2018]_.

1. **Easy to access**: Python modules are simple to install, and projects like `Anaconda <https://www.continuum.io/anaconda-overview>`_ are doing a great job in creating platform-independent environments.
2. **Easy to use**: Anyone with some experience can use their favourite Python tools to pre-process rquired data, set up, modify, and post-process simulations with Roger.
3. **Easy to modify**: Due to Python's popularity, available abstractions, and dynamic nature, Roger can be extended and modified with relatively little effort.

However, choosing Python over a compiled language like Fortran or C usually comes at a high computational cost. We overcome this gap by using `JAX <https://github.com/google/jax>`_, a framework that can act as a high-performance replacement for NumPy. JAX takes care of all performance optimizations in the background, and runs on CPUs and GPUs.

Available processes
+++++++++++++++++++

Here, we provide a brief overview for the available processes and their underlying theories:

**Soil hydraulics**:

- soil hydraulic parameters are approximated by the Brooks-Corey scheme ([Brooks1966]_).

**Interception**:

- rainfall and snowfall interception by vegetation ([Larsim2021]_)

**Snow**:

- snow accumulation
- delayed snow melt is based on degree-day approach and water retention of snow cover ([Larsim2021]_)
- rain-on-snow

**Infiltration**:

- matrix-driven infiltration based on Green-Ampt approach ([Peschke1985]_, [Weiler2005]_)
- gravity-driven infiltration based on viscous flow approach ([Germann2018]_)

**Evaporation**:

- evaporation from interception and surface storage
- soil evaporation based on Stage I (i.e. energy limiting stage) and Stage II (i.e. falling rate stage) ([Torres2010]_).

**Transpiration**:

- combination of residual potential evapotranspiration and vegetation-specific coeffcient

**Subsurface Runoff**:

- lateral subsurface runoff in the soil as described in [Steinbrich2016]_

**Percolation**:

- capillary-driven vertical drainage ([Salvucci1993]_)
- gravity-driven vertical drainage based on viscous flow approach ([Germann2018]_)

**Capillary rise**:

- capillary-driven vertical uplift ([Salvucci1993]_)

**Groundwater flow**:

- spatial explicit representation of shallow groundwater follows the aproach presented in [Stoll2010]_

**Crop phenology and crop rotation**:

- time-varying crop canopy cover and crop root depth is implemented as in [Steduto2009]_

**Solute transport**:

- StorAge selection (SAS) functions ([Rinaldo2015]_) are coupled with hydrologic simulations. SAS functions are used to calculate travel time distributions, residence time distribution and solute concentrations

**Biogeochemical processes**:

- Solute specific transformation processes, for example, denitrification ([Kunkel2012]_) or soil temperature ([Hillel1998]_)

Available pre-defined model structures
++++++++++++++++++++++++++++++++++++++

**SVAT**:

- only vertical processes are considered
- no lateral processes

**SVAT-CROP**:

- same as SVAT, but crop phenology (i.e. varying rooting depth and varying canopy cover) is explicitly represented

**ONED**:

- vertical and lateral processes are considered

**ONED-EVENT**:

- vertical and lateral processes are considered
- simulation of a single event

**SVAT-OXYGEN18**:

- calculates offline coupled oxygen-18 transport based on the hydrologic simulations with the SVAT model

**SVAT-BROMIDE**:

- calculates offline coupled bromide transport based on the hydrologic simulations with the SVAT model


Diagnostics
+++++++++++

Diagnostics are responsible for handling all model output, sanity checks of the solution, and restart file handling. They are implemented in a modular fashion, so additional diagnostics can be implemented easily. Already implemented diagnostics handle snapshot output, aggregation of variables, and monitoring of mass balance.

For more information, see :doc:`/reference/diagnostics`.


Pre-defined model setups
++++++++++++++++++++++++

Roger supports a wide range of pre-configured models. Several setups are already implemented that highlight some of the capabilities of Roger, and that serve as a basis for users to set up their own configuration: :doc:`/reference/model-gallery`.


Current limitations
+++++++++++++++++++

Roger is still in development. There are many open issues that we would like to fix later on:

- A routing scheme is not implemented, yet
- Simulations with biogeochemical processes have not been compared to measured data
- Simulations with gravity-driven infiltration have not been compared to measured data
- Sowing and harvesting of crops is time-invariant i.e. fixed dates are assumed for sowing and harvesting

References
++++++++++

.. [Brooks1966] Brooks, R. H., and Corey, A. T.: Properties of porous media affecting fluid flow, Journal of the Irrigation and Drainage Division, 92, 61-90, 1966.

.. [Germann2018] Germann, P. F. and Prasuhn, V.: Viscous Flow Approach to Rapid Infiltration and Drainage in a Weighing Lysimeter, Vadose Zone Journal, 17, 170020, 2018.

.. [Haefner2018] Häfner, D., Jacobsen, R. L., Eden, C., Kristensen, M. R. B., Jochum, M., Nuterman, R., and Vinter, B.: Veros v0.1 – a fast and versatile ocean simulator in pure Python, Geosci. Model Dev., 11, 3299-3312, 2018.

.. [Harman2015] Harman, C. J.: Time-variable transit time distributions and transport: Theory and application to storage-dependent transport of chloride in a watershed, Water Resources Research, 51, 1-30, 2015.

.. [Hillel1998] Hillel, D.: Environmental soil physics, Academic Press, London, UK, 1998.

.. [Kunkel2012] Kunkel, R., and Wendland, F.: Diffuse Nitrateinträge in die Grund- und Oberflächengewässer von Rhein und Ems - Ist-Zustands- und Maßnahmenanalysen, Forschungszentrum Jülich, Jülich, Germany, 143, 2012.

.. [Larsim2021] LARSIM-Entwicklergemeinschaft: Das Wasserhaushaltsmodell LARSIM: Modellgrundlagen und Anwendungsbeispiele, LARSIM-Entwicklergemeinschaft - Hochwasserzentralen LUBW, BLfU, LfU RP, HLNUG, BAFU, 258, 2021.

.. [Peschke1985] Peschke, G.: Zur Bildung und Berechnung von Regenabfluss, Wissenschaftliche Zeitschrift der Technischen Universität Dresden, 34, 1985.

.. [Rinaldo2015] Rinaldo, A., Benettin, P., Harman, C. J., Hrachowitz, M., McGuire, K. J., van der Velde, Y., Bertuzzo, E., and Botter, G.: Storage selection functions: A coherent framework for quantifying how catchments store and release water and solutes, Water Resources Research, 51, 4840-4847, 2015.

.. [Salvucci1993] Salvucci, G. D.: An approximate solution for steady vertical flux of moisture through an unsaturated homogeneous soil, Water Resources Research, 29, 3749-3753, 1993.

.. [Steduto2009] Steduto, P., Hsiao, T. C., Raes, D., and Fereres, E.: AquaCrop—The FAO Crop Model to Simulate Yield Response to Water: I. Concepts and Underlying Principles, Agronomy Journal, 101, 426-437, 2009.

.. [Steinbrich2016] Steinbrich, A., Leistert, H., and Weiler, M.: Model-based quantification of runoff generation processes at high spatial and temporal resolution, Environmental Earth Sciences, 75, 1423, 2016.

.. [Stoll2010] Stoll, S. and Weiler, M.: Explicit simulations of stream networks to guide hydrological modelling in ungauged basins, Hydrol. Earth Syst. Sci., 14, 1435-1448, 2010.

.. [Torres2010] Torres, E. A. and Calera, A.: Bare soil evaporation under high evaporation demand: a proposed modification to the FAO-56 model, Hydrological Sciences Journal, 55, 303-315, 2010.

.. [vanderVelde2012] van der Velde, Y., Torfs, P. J. J. F., van der Zee, S. E. A. T. M., and Uijlenhoet, R.: Quantifying catchment-scale mixing and its effect on time-varying travel time distributions, Water Resources Research, 48, 2012.

.. [Weiler2005] Weiler, M.: An infiltration model based on flow variability in macropores: development, sensitivity analysis and applications, Journal of Hydrology, 310, 294-315, 2005.