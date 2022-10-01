# Modelling crop phenology, bromide transport and nitrate transport with RoGeR at Reckenholz lysimeter facility

- `post_processing.py`: Produces figures and tables from data of the modelling experiment
- `write_moab.py`: Generates job scripts for computation on BinAC cluster
- `write_slurm.py`: Generates job scripts for computation on BwUniCluster 2.0

## observations
Contains measured lysimeter data of 9 lysimeter experiments as .nc-file:
- `lys1`: Reference lysimeter experiment covered with grass
- `lys2`: Lysimeter experiment covered with different crops and 130% fertilization
- `lys3`: Lysimeter experiment covered with different crops and 100% fertilization
- `lys8`: Lysimeter experiment covered with different crops and 70% fertilization
- `lys4`: Lysimeter experiment covered with different crops and organic intensive fertilization
- `lys9`: Lysimeter experiment covered with different crops and organic intensive (PEP) fertilization
- `lys2_bromide`: Bromide tracer experiment
- `lys8_bromide`: Bromide tracer experiment
- `lys9_bromide`: Bromide tracer experiment

---

The following folders contain the model setups. Each folder contains a subfolder
`input` from which the model reads the input data, a Python-script with the
implementation of the model setup and the corresponding job-script(s) for computation
on a cluster, and Python-scripts for post-processing.

## Hydrologic simulations
SVAT indicates that only vertical processes are active.
### svat
Single run of hydrologic model
- `svat.py`: Crop phenology/crop rotation is turned **off**.
- `svat_crop.py`: Crop phenology/crop rotation is turned **on**.

### svat_monte_carlo
Monte Carlo simulations with hydrologic model used for parameter estimation
- `*_svat_mc_moab.sh`: job scripts to run Monte Carlo simulations with crop phenology/crop rotation is turned **off** at provided lysimeter.
- `*_mc_moab.sh`: job scripts to run Monte Carlo simulations with crop phenology/crop rotation is turned **on** at provided lysimeter.

### svat_sensitivity
Saltelli simulations with hydrologic model used for sensitvity analysis
- `*_svat_sa_moab.sh`: job scripts to run Saltelli simulations with crop phenology/crop rotation is turned **off** at provided lysimeter.
- `*_sa_moab.sh`: job scripts to run Saltelli simulations with crop phenology/crop rotation is turned **on** at provided lysimeter.

## Transport simulations
In order to solve the transport with StorAge selection (SAS) functions, three schemes (sas_solver) are available:
- `deterministic`: Loop over deterministic sequence of hydrologic processes and solve SAS by internal loops for each process.
- `Euler`: Solve SAS with an explicit Euler scheme
- `RK4`: Solve SAS with an explicit Runge-Kutta fourth-order scheme

### svat_transport
Single run of transport model. Requires hydrologic simulations.

### svat_transport_monte_carlo
Monte Carlo simulations with bromide transport models and nitrate used for parameter estimation. Requires hydrologic simulations.
- `nitrate1_svat_crop_*_*_moab.sh`: job script to run Monte Carlo simulations with the provided sas solver and provided transport model structure (e.g. pow=power law). Crop phenology/crop rotation is turned **on** at provided lysimeter. SAS parameters are estimated from bromide experiments.
- `nitrate2_svat_crop_*_*_moab.sh`: job script to run Monte Carlo simulations with the provided sas solver and provided transport model structure (e.g. pow=power law). Crop phenology/crop rotation is turned **on** at provided lysimeter.
- `bromide_svat_crop_*_*_moab.sh`: job script to run Monte Carlo simulations with the provided sas solver and provided transport model structure (e.g. pow=power law). Crop phenology/crop rotation is turned **on** at provided lysimeter.
- `nitrate2_svat_*_*_moab.sh`: job script to run Monte Carlo simulations with the provided sas solver and provided transport model structure (e.g. pow=power law). Crop phenology/crop rotation is turned **off** at provided lysimeter.

### svat_transport_sensitivity
Saltelli simulations with bromide and nitrate transport model used for sensitvity analysis. Requires hydrologic simulations.
- `nitrate1_svat_crop_*_*_moab.sh`: job script to run Saltelli simulations with the provided sas solver and provided transport model structure (e.g. pow=power law). Crop phenology/crop rotation is turned **on** at provided lysimeter. SAS parameters are estimated from bromide experiments.
- `nitrate2_svat_crop_*_*_moab.sh`: job script to run Saltelli simulations with the provided sas solver and provided transport model structure (e.g. pow=power law). Crop phenology/crop rotation is turned **on** at provided lysimeter.
- `bromide_svat_crop_*_*_moab.sh`: job script to run Saltelli simulations with the provided sas solver and provided transport model structure (e.g. pow=power law). Crop phenology/crop rotation is turned **on** at provided lysimeter.
- `nitrate2_svat_*_*_moab.sh`: job script to run Saltelli simulations with the provided sas solver and provided transport model structure (e.g. pow=power law). Crop phenology/crop rotation is turned **off** at provided lysimeter.
