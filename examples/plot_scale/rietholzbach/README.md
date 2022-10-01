# Modelling oxygen-18 transport and bromide transport with RoGeR at Rietholzbach lysimeter site

- `post_processing.py`: Produces figures and tables from data of the modelling experiment
- `write_moab.py`: Generates job scripts for computation on BinAC cluster
- `write_slurm.py`: Generates job scripts for computation on BwUniCluster 2.0

## observations
Contains measured lysimeter data as .nc-file and measured data from bromide experiment as .csv.

## hydrus_benchmark
Contains data from HYDRUS-1D simulations.

---

The following folders contain the model setups. Each folder contains a subfolder
`input` from which the model reads the input data, a Python-script with the
implementation of the model setup and the corresponding job-script(s) for computation
on a cluster, and Python-scripts for post-processing.

## Hydrologic simulations
### svat
Single run of hydrologic model

### svat_monte_carlo
Monte Carlo simulations with hydrologic model used for parameter estimation
- `mc.sh`: job script to run Monte Carlo simulations

### svat_sensitivity
Saltelli simulations with hydrologic model used for sensitvity analysis
- `sa.sh`: job script to run Saltelli simulations

## Transport simulations
In order to solve the transport with StorAge selection (SAS) functions, three schemes (sas_solver) are available:
- `deterministic`: Loop over deterministic sequence of hydrologic processes and solve SAS by internal loops for each process.
- `Euler`: Solve SAS with an explicit Euler scheme
- `RK4`: Solve SAS with an explicit Runge-Kutta fourth-order scheme

### svat_transport
Single run of transport model. Requires hydrologic simulations.

### svat_transport_monte_carlo
Monte Carlo simulations with oxygen-18 transport models used for parameter estimation. Requires hydrologic simulations.
- `oxygen18_*_*_moab.sh`: job script to run Monte Carlo simulations with the provided sas solver and provided transport model structure (e.g. ad=advection-dispersion)

### svat_transport_sensitivity
Saltelli simulations with oxygen-18 transport model used for sensitvity analysis. Requires hydrologic simulations.
- `oxygen18_*_*_moab.sh`: job script to run Saltelli simulations with the provided sas solver and provided transport model structure (e.g. ad=advection-dispersion)
- `param_bounds.yml`: contains parameter boundaries for Saltelli sampling

### svat_bromide_benchmark
Virtual bromide experiments with bromide transport model. Requires hydrologic simulations.
