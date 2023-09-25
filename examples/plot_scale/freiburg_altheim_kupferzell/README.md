# Modelling virtual tracer transport with Roger using climate projections and land cover scenarios

- `pre_processing.py`: Prepare the input data and plot delta changes of climate signal
- `write_moab_jobs.py`: Generate job scripts for computation on BinAC cluster using the MOAB workload manager
- `write_slurm_jobs.py`: Generate job scripts for computation on BwUniCluster 2.0 cluster using the SLURM workload manager
- `submit_*.sh`: Script to submit batch jobs
- `plot_time_series.py`: Plot time series
- `calculate_delta_changes.py`: Calculate delta changes for near future (2040 - 2060) and far future (2080 - 2100)
- `calculate_change_attribution.py`: Quantify change attribution of soil physical properties and climate signal using linear models

## climate_projections
Contains climate projections from regional climate models. We use the data of the following climate projections:
- CCCma-CanESM2 CCLM4-8-17 ("dry" and "warm")
- MPI-M-MPI-ESM-LR RCA4 ("wet" and "cool")

## dwd
Contains daily minimum and maximum air temperature measured at DWD stations

---

The following folders contain the model setups. Each folder contains a subfolder
`input` from which the model reads the input data and a Python-script with the
implementation of the model setup and the corresponding job-script(s) for computation
on a cluster, and Python-scripts for post-processing. If the file name of the
job scripts contains `_gpu` computations run on GPU.

## input/land_cover_scenario
- grass
- Corn
- Corn and catch crop
- Crop rotation (winter barley, catch crop, sugar beet, winter wheat)

## Hydrological simulations
### svat
- `svat.py`: SVAT-model setup
- `svat_crop.py`: SVAT-CROP-model setup
- `svat_*.sh`: Job scripts to run the SVAT-model
- `svat_crop_*.sh`: Job scripts to run the SVAT-CROP-model
- `merge_output.py`: Merges the model output into a single file

## Transport simulations
### svat_transport
- `svat.py`: SVAT-model setup
- `svat_crop.py`: SVAT-CROP-model setup
- `svat_*.sh`: Job scripts to run the SVAT-model
- `svat_crop_*.sh`: Job scripts to run the SVAT-CROP-model
- `merge_output.py`: Merges model output into a single file

## Workflow

1. Make the input data ready for the simulations:
```
python pre_processing.py
```

2. Here, I provide an example shell scripts can be generated. These scripts can be submitted to a computing cluster:
```
python write_slurm_jobs.py
```

3. Run the model to simulate hydrological fluxes and storages by submitting the model script to a computing cluster:
```
./submit_svat_grass_jobs_slurm.sh
./submit_svat_corn_jobs_slurm.sh
./submit_svat_corn_catch_crop_jobs_slurm.sh
./submit_svat_crop_rotation_jobs_slurm.sh
```

4. Merge the model output into single files:
```
cd svat
python merge_output.py
```

5. Run the model to simulate virtual tracer concentrations and travel times by submitting the model script to a computing cluster:
```
./submit_svat_grass_transport_jobs_slurm.sh
./submit_svat_corn_transport_jobs_slurm.sh
./submit_svat_corn_catch_crop_transport_jobs_slurm.sh
./submit_svat_crop_rotation_transport_jobs_slurm.sh
```

6. Merge the model output into single files
```
cd svat_transport
python merge_output.py
```

7. Plot the time series:
```
python plot_time_series.py
```

8. Calculate the delta changes of the simulated variables:
```
python calculate_delta_changes.py
```

9. Quantify the change attribution:
```
python calculate_change_attribution.py
```