# Modelling oxygen-18 transport and bromide transport with RoGeR at Rietholzbach lysimeter site

- `post_processing.py`: Produces figures and tables from data of the modelling experiment
- `write_moab_jobs.py`: Generates job scripts for computation on BinAC cluster using the MOAB workload manager

## observations
Contains measured lysimeter data as .nc-file and measured data from bromide experiment as .csv.

## hydrus_benchmark
Contains data from HYDRUS-1D simulations.

---

The following folders contain the model setups. Each folder contains a subfolder
`input` from which the model reads the input data and a Python-script with the
implementation of the model setup and the corresponding job-script(s) for computation
on a cluster, and Python-scripts for post-processing. If the file name of the
job scripts contains `_gpu` computations run on GPU.

## Hydrologic simulations
### svat
Single run of the hydrologic model

### svat_monte_carlo
Monte Carlo simulations with hydrologic model used for parameter estimation
- `param_bounds.yml`: contains parameter boundaries for uniform sampling
- `svat_mc.sh`: job script to run Monte Carlo simulations
- `states_hm_mc.mc`: Hydrologic Monte Carlo simulations
- `states_hm1.mc`: States of best hydrologic simulations
- `states_hm100.mc`: States of best 100 hydrologic simulations

### svat_sensitivity
Saltelli simulations with hydrologic model used for sensitvity analysis
- `sample_params.py`: Samples parameter sets using Saltelli's extension of the Sobol' sequence
- `param_bounds.yml`: contains parameter boundaries for Saltelli sampling
- `params_saltelli.nc`: Sampled hydrologic model parameters and transport model parameters
- `svat_sa_for_*.sh`: job script to run hydrologic simulations required by transport model structures
- `states_hm_sa_for_*.nc`: Hydrologic simulations for corresponding model structure

## Transport simulations
In order to solve the transport with StorAge selection (SAS) functions, three schemes (`sas_solver`) are available:
- `deterministic`: Loops over deterministic sequence of hydrologic processes and solve SAS by internal loops (i.e. substepping) for each hydrologic process.
- `Euler`: Solve SAS with an explicit Euler scheme
- `RK4`: Solve SAS with an explicit Runge-Kutta fourth-order scheme

### svat_oxygen18
Single run of oxygen 18 transport model. Requires hydrologic simulations.

### svat_oxygen18_monte_carlo
Monte Carlo simulations with oxygen-18 transport models used for parameter estimation. Requires hydrologic simulations.
- `param_bounds.yml`: contains parameter boundaries for uniform sampling
- `oxygen18_*_*_mc.sh`: job script to run Monte Carlo simulations with the provided sas solver and provided transport model structure (e.g. ad=advection-dispersion).
- `bootstrap.py`: Bootstrap best 100 hydrologic simulations based on size of Monte Carlo samples (e.g. 1000)
- `states_hm100_bootstrap.nc`: Bootstrapped samples of best 100 hydrologic simulations
- `states_*_mc.nc`: Monte Carlo transport simulations with provided transport model structure
- `states_hm_best_for_*.nc`: Hydrologic simulation corresponding to best transport simulation

### svat_oxygen18_sensitivity
Saltelli simulations with oxygen-18 transport model used for sensitvity analysis. Requires hydrologic simulations.
- `oxygen18_*_*_sa.sh`: job script to run Saltelli simulations with the provided sas solver and provided transport model structure (e.g. ad=advection-dispersion)
- `states_*_sa.nc`: Saltelli transport simulations with provided transport model structure
- `param_bounds.yml`: contains parameter boundaries for Saltelli sampling
- `params_saltelli.nc`: Sampled hydrologic model parameters and transport model parameters

### svat_bromide_benchmark
Virtual bromide experiments with bromide transport model. Requires hydrologic simulations.
- `bromide_*_*.sh`: job script to run simulations with the provided sas solver and provided transport model structure (e.g. ad=advection-dispersion)
- `states_hm_best_for_*.nc`: Hydrologic simulation corresponding to best transport simulation
- `sas_params_*.nc`: SAS parameters of best transport simulation
