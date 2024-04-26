<p align="center">
<img src="doc/_images/roger-logo.png">
</p>

<p align="center">
<i>Runoff Generation Research - a process-based hydrological toolbox model in Python</i>
</p>

<p align="center">
  <a href="http://roger.readthedocs.io/?badge=latest">
    <img src="https://readthedocs.org/projects/roger/badge/?version=latest" alt="Documentation status">
  </a>
  <a href="https://github.com/Hydrology-IFH/roger/actions/workflows/test-all.yml">
    <img src="https://github.com/Hydrology-IFH/roger/actions/workflows/test-all.yml/badge.svg" alt="Test status">
  </a>
  <a href="https://codecov.io/gh/Hydrology-IFH/roger" > 
  <img src="https://codecov.io/gh/Hydrology-IFH/roger/branch/main/graph/badge.svg?token=KXSVNGDDNH"/> 
  </a>
  <a href="https://zenodo.org/badge/latestdoi/536477819"><img src="https://zenodo.org/badge/536477819.svg" alt="DOI"></a>
</p>

RoGeR, *Runoff Generation Research*, is a process-based hydrological model that can be applied from plot to catchment scale. RoGeR is written in pure Python, which facilitates model setup and model workflows. We want to enable high-performance hydrological modelling with a clear focus on flexibility and usability.

RoGeR supports a NumPy backend for small-scale problems, and a
high-performance [JAX](https://github.com/google/jax) backend
with CPU and GPU support. Parallel computation is available via MPI and supports
distributed execution on any number of nodes/CPU cores.

Inspired by [Veros](https://veros.readthedocs.io/en/latest/).

## Documentation

We strongly recommend to [visit our documentation](https://roger.readthedocs.io/en/latest/).


## Features

<p align="center">
  <a href="https://vimeo.com/889894624">
      <img src="doc/_images/fluxes_theta_and_tt_rt.gif?raw=true" alt="RoGeR - 25 square meter resolved simulations of the Eberbaechle catchment, Germany (2019-2022)">
  </a>
</p>

<p align="center">
(25 square meter resolved simulations 
of the Eberbaechle catchment, 
Germany (2019-2022), click for better
quality)
</p>

RoGeR provides

-   grid-based **1D models**
-   **offline solute transport** with several **StorAge selection (SAS) functions**
-   solute-specific biogeochemical processes
-   implementations of **capillary-driven infiltration (Green-Ampt)**
-   several **pre-implemented diagnostics** such as averages or collecting values
    at given time interval, variable time aggregation, travel time distributions
    and residence time distributions (written to netCDF4 output)
-   **pre-configured idealized and realistic setups** that are ready to
    run and easy to adapt
-   **accessibility and extensibility** due to high-level programming language Python


## Basic usage

To run RoGeR, you need to set up a model --- i.e., specify which settings
and model domain you want to use. This is done by subclassing the
`RogerSetup` base class in a *setup script* that is written in Python. A good
place to start is the
[SVAT Tutorial](https://github.com/Hydrology-IFH/roger/blob/master/roger/examples/plot_scale/svat_tutorial):


After setting up your model, all you need to do is call the model setup:
```bash
# move into the folder containing the model script
python svat.py
```

For more information on using RoGeR, have a look at [our
documentation](http://roger.readthedocs.io).

## Contributing

Contributions to RoGeR are always welcome, no matter if you spotted an
inaccuracy in [the documentation](https://roger.readthedocs.io), wrote a
new setup, fixed a bug, or even extended RoGeR\' core mechanics. There
are 2 ways to contribute:

1.  If you want to report a bug or request a missing feature, please
    [open an issue](https://github.com/Hydrology-IFH/roger/issues). If you
    are reporting a bug, make sure to include all relevant information
    for reproducing it (ideally through a *minimal* code sample).
2.  If you want to fix the issue yourself, or wrote an extension for
    Roger - great! You are welcome to submit your code for review by
    committing it to a repository and opening a [pull
    request](https://github.com/Hydrology-IFH/roger/pulls). However,
    before you do so, please check [the contribution
    guide](http://roger.readthedocs.io/quickstart/get-started.html#enhancing-Roger)
    for some tips on testing and benchmarking, and to make sure that
    your modifications adhere with our style policies. Most importantly,
    please ensure that you follow the [PEP8
    guidelines](https://www.python.org/dev/peps/pep-0008/), use
    *meaningful* variable names, and document your code using
    [Google-style
    docstrings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

## How to cite

If you use Roger in scientific work, please consider citing [the following publication](...):

```bibtex
@article{
	title = {Roger v3.0.3 – a process-based hydrologic toolbox model in {Python}},
	volume = {...},
	doi = {https://doi.org/10.5194/gmd-2023-118},
	journal = {Geosci. Model Dev.},
	author = {Schwemmle, Robin, and Leistert, Hannes, and Weiler, Markus},
	year = {2023},
	pages = {...},
}
```

Or have a look at [our documentation](https://roger.readthedocs.io/en/latest/more/publications.html)
for more publications involving Roger.

## TODO
- implement runoff and channel routing (e.g. kinematic wave or hydraulic approach)
- implement distributed model with run-on infiltration
- use coarser spatial and temporal resolution for computation of
groundwater-related processes
- implement baseflow in the groundwater routine. requires surface water depth.
- implement surface runoff generation for gravity-driven infiltration
- implement gravity-driven infiltration and percolation and include it into the transport routine
- implement time-variant sowing and harvesting of crops

## License
This software can be distributed freely under the MIT license. Please read the LICENSE for further information.
© 2024, Robin Schwemmle (<robin.schwemmle@hydrology.uni-freiburg.de>)
