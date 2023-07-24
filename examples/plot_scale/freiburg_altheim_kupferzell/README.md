# Modelling virtual tracer transport with Roger using climate projections and land cover scenarios

All results have been generated with Roger version 3.0.

- `make_figures_and_tables.py`: Produces figures and tables from data of the modelling experiment
- `write_moab_jobs.py`: Generates job scripts for computation on BinAC cluster using the MOAB workload manager

## climate_projections
Contains climate projections from regional climate models

## dwd
Contains daily minimum and maximum air temperature measured at DWD stations

---

The following folders contain the model setups. Each folder contains a subfolder
`input` from which the model reads the input data and a Python-script with the
implementation of the model setup and the corresponding job-script(s) for computation
on a cluster, and Python-scripts for post-processing. If the file name of the
job scripts contains `_gpu` computations run on GPU.

## Hydrologic simulations
### svat


## Transport simulations
### svat_transport
