# Modelling nitrate transport with RoGeR at multiple agricultural regions (BOADKH) in Baden-Wuerttemberg, Germany

The following regions (BOADKH) are considered:
- Lake Constance (B) using meteorological data from the DWD-stations Singen (Hohentwiel; station_id: 6263), Azenweiler (station_id: 931) and Unterraderach (station_id: 6258)
- Upper rhine vally (O) using meteorological data from the DWD-stations Müllheim im Markgräflerland (station_id: 259), Freiburg (station_id: 1443) and Ihringen (station_id: 5453)
- Alb-Danube (AD) using meteorological data from the DWD-stations Riedlingen (station_id: 4189), Kirchen (station_id: 3418) and Mähringen (station_id: 15444)
- Kraichgau (K) using meteorological data from the DWD-stations Heidelsheim (station_id: 731), Elsenz (station_id: 1255) and Zaberfeld (station_id: 7498)
- Hohenlohe (H) using meteorological data from the DWD-stations Rechbach (station_id: 2787), Stachenhausen (station_id: 6260) and Öhringen (station_id: 3761)

- `write_moab_jobs.py`: Generate job scripts for computation on BinAC cluster using the MOAB workload manager
- `write_slurm_jobs.py`: Generate job scripts for computation on BwUniCluster 2.0 cluster using the SLURM workload manager
- `submit_*.sh`: Script to submit batch jobs
- `plot_time_series.py`: Plot time series
- `plot_nitrogen_balance.py`: Plot nitrogen balance of regions and crop rotation scenarios

---

The following folders contain the model setups. Each folder contains a subfolder
`input` from which the model reads the input data and a Python-script with the
implementation of the model setup and the corresponding job-script(s) for computation
on a cluster, and Python-scripts for post-processing. If the file name of the
job scripts contains `_gpu` computations run on GPU.

## input/crop_rotation_scenarios
The following crop combinations are rotated:
- Winter wheat and clover
- Winter wheat and corn
- Summer wheat and winter wheat
- Summer wheat, clover and winter wheat
- Winter wheat, clover and corn
- Summer wheat, winter wheat and winter rape
- Winter wheat and winter rape
- Summer wheat, winter wheat and corn
- Winter wheat, winter grain pea and winter rape
- Winter wheat, sugar beet and corn

The name of the folder contains the considered crop combinations. Each folder contains a file that describes the crop rotation in `crop_rotation.csv`.

## Hydrological simulations
### svat_crop
- `svat_crop.py`: SVAT-CROP-model setup
- `svat_crop_*.sh`: Job scripts to run the SVAT-CROP-model
- `merge_output.py`: Merges the model output into a single file

## Transport simulations
### svat_transport
- `svat_crop_nitrate.py`: SVAT-CROP-NITRATE-model setup
- `svat_crop_nitrate_*.sh`: Job scripts to run the SVAT-CROP-model
- `merge_output.py`: Merges model output into a single file