# Modelling oxygen-18 transport and bromide transport with RoGeR at the Rietholzbach lysimeter site

All results have been generated with Roger version 3.0.5.

Workflow:
1. Run the Monte-Carlo simulations following the instructions in `svat_monte_carlo/`
2. Run the Monte-Carlo simulations following the instructions in `svat_oxygen18_monte_carlo/`
3. Simulate the bromide leaching following the instructions in `svat_bromide_benchmark/`
4. Simulate the water ages following the instructions in `svat_oxygen18/`
5. Run the Saltelli simulations following the instructions in `svat_sensitivity/`
6. Run the Saltelli simulations following the instructions in `svat_oxygen18_sensitivity/`
7. Postprocessing of the simulations to generate figures and tables for publication:
- `check_lysimeter_water_balance.py`: Produces figures to verify the fluxes and storage change of the lysimeter
- `diagnostic_polar_plots.py`: Produces diagnostic polar plots (requires simulations of `svat_monte_carlo/` and `svat_oxygen18_monte_carlo/`)
- `dotty_plots.py`: Produces dotty plots for the Monte-Carlo simulations (requires simulations of `svat_monte_carlo/` and `svat_oxygen18_monte_carlo/`)
- `plot_bromide_experiment.py`: Produces figures to present the results of the virtual bromide experiment (requires simulations of `svat_bromide_benchmark/`)
- `plot_figures_for_talk.py`: Plot selected figures with increased element size (e.g. larger font size)
- `plot_sas_functions.py`: Produces figures to illustrate the shape of the SAS functions
- `plot_simulated_duration_curves.py`: Compares simulated and observed duration curves of oxygen-18 in percolation (requires simulations of `svat_oxygen18_monte_carlo/`)
- `plot_simulated_macropore_infiltration.py`: Visualizes the macropore infiltration simulated by RoGeR (requires simulations of `svat_monte_carlo/`)
- `plot_simulated_time_series.py`: Compares the simulated variables with observations (requires simulations of `svat_monte_carlo/` and `svat_oxygen18_monte_carlo/`)
- `plot_water_ages.py`: Produces figures with water age distributions (e.g. travel time distribution of percolation; requires simulations of `svat_oxygen18_monte_carlo/`)
- `sobol_indices.py`: Calculation and visualization of Sobol' indices (requires simulations of `svat_sensitivity/` and `svat_oxygen18_sensitivity/`)

## Abbreviations of transport model structures
- pi: piston (implemented with power law distribution function using fixed parameters)
- cm: complete-mixing (implemented with uniform distribution function)
- adp: advection-dispersion (implemented with power law distribution function)
- adpt: time-variant advection-dispersion (implemented with power law distribution function using soil water storage dependent parameters)

## observations
Contains measured lysimeter data as .nc-file and measured data of the bromide experiment as .csv.

## hydrus_benchmark
Contains data from HYDRUS-1D simulations. The corresponding workflows are available at https://doi.org/10.5281/zenodo.7632281

---

The following folders contain the model setups. Each folder contains a subfolder
`input` from which the model reads the input data and a Python-script with the
implementation of the model setup and the corresponding job-script(s) for computation
on a cluster, a subfolder named `output` which contains simulations and Python-scripts for post-processing. If the file name of the job scripts contains `_gpu` computations run on GPU.

## simulated hydrologic fluxes
### svat
Single run of the hydrologic model
- `input/`: contains precipitation data (`PREC.txt`; 10 minutes time steps), air temperature data (`TA.txt`; daily time steps) and potential evapotranspiration data (`PET.txt`; daily time steps).
- `output/`: contains model output
- `merge_output.py`: Merges the model output into a single file

Workflow:
1. Run `svat.sh`
2. Run `merge_output.py`

### svat_monte_carlo
Monte Carlo simulations with hydrologic model used for parameter estimation
- `input/`: Contains precipitation data (`PREC.txt`; 10 minutes time steps), air temperature data (`TA.txt`; daily time steps) and potential evapotranspiration data (`PET.txt`; daily time steps).
- `param_bounds.yml`: contains parameter boundaries for Monte Carlo sampling
- `svat_mc.sh`: job script to run Monte Carlo simulations
- `svat_mc_gpu.sh`: job script to run Monte Carlo simulations on GPU
- `output/states_hm_mc.mc`: Monte Carlo simulations of hydrologic fluxes
- `output/states_hm1.mc`: States of best simulated hydrologic fluxes
- `output/states_hm100.mc`: States of best 100 simulated hydrologic fluxes
- `merge_output.py`: Merges the model output into a single file
- `evaluate_simulations.py`: Calculates metrics for each simulation

Workflow:
1. Run `svat_mc.sh`
2. Run `merge_output.py`
3. Run `evaluate_simulations.py`

### svat_sensitivity
Saltelli simulations with hydrologic model used for sensitvity analysis
- `input/`: Contains precipitation data (`PREC.txt`; 10 minutes time steps), air temperature data (`TA.txt`; daily time steps) and potential evapotranspiration data (`PET.txt`; daily time steps).
- `sample_params.py`: Samples parameter sets using Saltelli's extension of the Sobol' sequence
- `param_bounds.yml`: contains parameter boundaries for Saltelli sampling
- `params_saltelli.nc`: Sampled hydrologic model parameters and transport model parameters
- `svat_sa_for_*.sh`: job script to run simulated hydrologic fluxes required by transport model structures
- `svat_sa_for_*_gpu.sh`: job script to run simulated hydrologic fluxes on GPU required by transport model structures
- `output/states_hm_sa_for_*.nc`: simulated hydrologic fluxes for corresponding transport model structure
- `merge_output.py`: Merges the model output into a single file
- `evaluate_simulations.py`: Calculates metrics for each simulation

Workflow:
1. Run `svat_sa_for_*.sh` for the considered transport model structures (*)
2. Run `merge_output.py`
3. Run `evaluate_simulations.py` for the considered transport model structures using the command line arguments (`--transport-model-structures`)

## Solute transport simulations
In order to solve the transport with StorAge selection (SAS) functions, three schemes (`sas_solver`) are available:
- `deterministic`: Loops over deterministic sequence of hydrologic processes and solve SAS by internal loops (i.e. substepping) for each hydrologic process.
- `Euler`: Solve SAS with an explicit Euler scheme
- `RK4`: Solve SAS with an explicit Runge-Kutta fourth-order scheme

### svat_oxygen18
Single run of oxygen 18 transport model. Requires simulated hydrologic fluxes and transport model parameters.
- `input/`: contains data for oxygen-18 in precipitation data (`d18O.txt`; daily time steps) and hydrological fluxes and storages (`states_hm.nc`; daily time steps).
- `oxygen18_deterministic_svat_*_gpu.sh`: job script to run Monte Carlo simulations on GPU with the provided sas solver and provided transport model structure (e.g. adp=advection-dispersio-power).
- `output/`: contains model output

Workflow:
1. Run `oxygen18_determistic_svat_*_gpu.sh` for the considered transport model structures (*; see [transport model structures](##Abbreviations-of-transport-model-structures))
2. Run `merge_output.py`
3. Run `evaluate_simulations.py` for the considered transport model structures using the command line arguments (`--transport-model-structures`)

### svat_oxygen18_monte_carlo
Monte Carlo simulations with oxygen-18 transport models used for parameter estimation. Requires simulated hydrologic fluxes.
- `input/`: contains data for oxygen-18 in precipitation data (`d18O.txt`; daily time steps) and hydrological fluxes and storages (`states_hm100_bootstrap.nc`; daily time steps).
- `param_bounds.yml`: contains parameter boundaries for Monte Carlo sampling
- `oxygen18_deterministic_svat_*_mc_*_gpu.sh`: job script to run Monte Carlo simulations on GPU with the provided sas solver and provided transport model structure (e.g. adp=advection-dispersion-power).
- `bootstrap.py`: Bootstrap best 10 or 100 simulated hydrologic fluxes based on size of Monte Carlo samples (e.g. 1000)
- `input/states_hm10_bootstrap.nc`: Bootstrapped samples of best 10 simulated hydrologic fluxes
- `input/states_hm100_bootstrap.nc`: Bootstrapped samples of best 100 simulated hydrologic fluxes
- `output/states_*_mc.nc`: Monte Carlo transport simulations with provided transport model structure
- `states_hm_best_for_*.nc`: Hydrologic simulation corresponding to best transport simulation
- `merge_output.py`: Merges the model output into a single file
- `evaluate_simulations.py`: Calculates metrics for each simulation

Workflow:
1. Run `oxygen18_determistic_svat_*_mc_*_gpu.sh` for the considered transport model structures (*; see [transport model structures](##Abbreviations-of-transport-model-structures)) and split number (i.e. we split the run in 1000 simulations since the 10000 simulations does not fit in the GPU memory of the NVIDIA K80)
2. Run `merge_output.py`
3. Run `evaluate_simulations.py` for the considered transport model structures using the command line arguments (`--transport-model-structures`)

### svat_oxygen18_sensitivity
Saltelli simulations with oxygen-18 transport model used for sensitvity analysis. Requires simulated hydrologic fluxes.
- `input/`: contains data for oxygen-18 in precipitation data (`d18O.txt`; daily time steps) and hydrological fluxes and storages (`states_hm_saltelli_for_*.nc`; daily time steps; *=[transport model structures](##Abbreviations-of-transport-model-structures)).
- `oxygen18_*_*_sa_gpu.sh`: job script to run Saltelli simulations on GPU with the provided sas solver and provided transport model structure (e.g. adp=advection-dispersion-power)
- `output/states_*_sa.nc`: Saltelli transport simulations with provided transport model structure
- `param_bounds.yml`: contains parameter boundaries for Saltelli sampling
- `sample_params.py`: Generates parameters (`params_saltelli.nc`) for Saltelli simulations used for sensitivity analysis
- `params_saltelli.nc`: Sampled hydrologic model parameters and transport model parameters
- `merge_output.py`: Merges the model output into a single file
- `evaluate_simulations.py`: Calculates metrics for each simulation

Workflow:
1. Run `sample_params.py`
2. Run `oxygen18_determistic_svat_*_sa_*_*_gpu.sh` for the considered transport model structures (*; see [transport model structures](##Abbreviations-of-transport-model-structures)) and split number (i.e. we split the run in 1000 simulations since the 10000 simulations does not fit in the GPU memory of the NVIDIA K80)
3. Run `merge_output.py`
4. Run `evaluate_simulations.py` for the considered transport model structures using the command line arguments (`--transport-model-structures`)

### svat_bromide_benchmark
Virtual bromide experiments with bromide transport model. Requires simulated hydrologic fluxes and transport model parameters.
- `bromide_*_*.sh`: job script to run simulations with the provided sas solver and provided transport model structure (e.g. ad=advection-dispersion)
- `merge_output.py`: Merges the model output into a single file
- `evaluate_simulations.py`: Calculates metrics for each simulation

Workflow:
1. Run `bromide_determistic_svat_*_*.sh` for the considered transport model structures (*; see [transport model structures](##Abbreviations-of-transport-model-structures)) and year of experiment 
2. Run `merge_output.py`