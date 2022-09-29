# Modelling oxygen-18 transport and bromide transport with RoGeR at Rietholzbach lysimeter site

- `post_processing.py`: Produces figures and tables from data of the modelling experiment
- `write_moab.py`: Generates job scripts for computation on BinAC cluster
- `write_slurm.py`: Generates job scripts for computation on BwUniCluster 2.0

## observations
Contains measured lysimeter data as .nc-file and measured data from bromide experiment as .csv.

## hydrus_benchmark
Contains data from HYDRUS-1D simulations.

---

The following folders contain model setups. Each folder contains a subfolder
`input` from which the model reads the input data, a Python-script with the
implementation of the model setup and corresponding job-script(s) for computation
on a cluster, and Python-scripts for post-processing.

## svat
Single run of hydrologic model

## svat_monte_carlo
Monte Carlo simulations with hydrologic model

## svat_sensitivity
Saltelli simulations with hydrologic model

## svat_transport
Single run of transport model. Requires hydrologic simulations.

## svat_transport_monte_carlo
Monte Carlo simulations with oxygen-18 transport models. Requires hydrologic simulations.

## svat_transport_sensitivity
Saltelli simulations with oxygen-18 transport model. Requires hydrologic simulations.

## svat_bromide_benchmark
Virtual bromide experiments with bromide transport model. Requires hydrologic simulations.
