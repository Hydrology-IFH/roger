# Modelling oxygen-18 transport and bromide transport with Roger at Rietholzbach lysimeter site

All results have been generated with Roger version 3.0

- `post_processing.py`: Produces figures and tables from data of the modelling experiment
- `write_moab_jobs.py`: Generates job scripts for computation on BinAC cluster using the MOAB workload manager

## observations
Contains measured lysimeter data as .nc-file and measured data from bromide experiment as .csv.

## hydrus_benchmark
Contains data from HYDRUS-1D simulations. The corresponding workflows are available at https://doi.org/10.5281/zenodo.7632281

---

The following folders contain the model setups. Each folder contains a subfolder
`input` from which the model reads the input data and a Python-script with the
implementation of the model setup and the corresponding job-script(s) for computation
on a cluster, and Python-scripts for post-processing. If the file name of the
job scripts contains `_gpu` computations run on GPU.

## simulated hydrologic fluxes
### svat
Single run of the hydrologic model

### svat_monte_carlo
Monte Carlo simulations with hydrologic model used for parameter estimation
- `param_bounds.yml`: contains parameter boundaries for Monte Carlo sampling
- `svat_mc.sh`: job script to run Monte Carlo simulations
- `svat_mc_gpu.sh`: job script to run Monte Carlo simulations on GPU
- `states_hm_mc.mc`: Monte Carlo simulations of hydrologic fluxes
- `states_hm1.mc`: States of best simulated hydrologic fluxes
- `states_hm100.mc`: States of best 100 simulated hydrologic fluxes

### svat_sensitivity
Saltelli simulations with hydrologic model used for sensitvity analysis
- `sample_params.py`: Samples parameter sets using Saltelli's extension of the Sobol' sequence
- `param_bounds.yml`: contains parameter boundaries for Saltelli sampling
- `params_saltelli.nc`: Sampled hydrologic model parameters and transport model parameters
- `svat_sa_for_*.sh`: job script to run simulated hydrologic fluxes required by transport model structures
- `svat_sa_for_*_gpu.sh`: job script to run simulated hydrologic fluxes on GPU required by transport model structures
- `states_hm_sa_for_*.nc`: simulated hydrologic fluxes for corresponding transport model structure

## Solute transport simulations
In order to solve the transport with StorAge selection (SAS) functions, three schemes (`sas_solver`) are available:
- `deterministic`: Loops over deterministic sequence of hydrologic processes and solve SAS by internal loops (i.e. substepping) for each hydrologic process.
- `Euler`: Solve SAS with an explicit Euler scheme
- `RK4`: Solve SAS with an explicit Runge-Kutta fourth-order scheme

### svat_oxygen18
Single run of oxygen 18 transport model. Requires simulated hydrologic fluxes.

### svat_oxygen18_monte_carlo
Monte Carlo simulations with oxygen-18 transport models used for parameter estimation. Requires simulated hydrologic fluxes.
- `param_bounds.yml`: contains parameter boundaries for Monte Carlo sampling
- `oxygen18_*_*_mc_gpu.sh`: job script to run Monte Carlo simulations on GPU with the provided sas solver and provided transport model structure (e.g. adp=advection-dispersio-power).
- `bootstrap.py`: Bootstrap best 10 or 100 simulated hydrologic fluxes based on size of Monte Carlo samples (e.g. 1000)
- `states_hm10_bootstrap.nc`: Bootstrapped samples of best 10 simulated hydrologic fluxes
- `states_hm100_bootstrap.nc`: Bootstrapped samples of best 100 simulated hydrologic fluxes
- `states_*_mc.nc`: Monte Carlo transport simulations with provided transport model structure
- `states_hm_best_for_*.nc`: Hydrologic simulation corresponding to best transport simulation

### svat_oxygen18_sensitivity
Saltelli simulations with oxygen-18 transport model used for sensitvity analysis. Requires simulated hydrologic fluxes.
- `oxygen18_*_*_sa_gpu.sh`: job script to run Saltelli simulations on GPU with the provided sas solver and provided transport model structure (e.g. adp=advection-dispersion-power)
- `states_*_sa.nc`: Saltelli transport simulations with provided transport model structure
- `param_bounds.yml`: contains parameter boundaries for Saltelli sampling
- `params_saltelli.nc`: Sampled hydrologic model parameters and transport model parameters

### svat_bromide_benchmark
Virtual bromide experiments with bromide transport model. Requires simulated hydrologic fluxes.
- `bromide_*_*.sh`: job script to run simulations with the provided sas solver and provided transport model structure (e.g. ad=advection-dispersion)
- `states_hm_best_for_*.nc`: Hydrologic simulation corresponding to best transport simulation
