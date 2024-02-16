# Modelling nitrate transport with RoGeR for multiple agricultural regions (BOADKH) in Baden-Wuerttemberg, Germany

The following regions (BOADKH) are considered:
- Lake Constance (B) using meteorological data from the DWD-stations Singen (Hohentwiel; station_id: 6263), Azenweiler (station_id: 931) and Unterraderach (station_id: 6258)
- Upper rhine vally (O) using meteorological data from the DWD-stations Müllheim im Markgräflerland (station_id: 259), Freiburg (station_id: 1443) and Ihringen (station_id: 5453)
- Alb-Danube (AD) using meteorological data from the DWD-stations Riedlingen (station_id: 4189), Kirchen (station_id: 3418) and Mähringen (station_id: 15444)
- Kraichgau (K) using meteorological data from the DWD-stations Heidelsheim (station_id: 731), Elsenz (station_id: 1255) and Zaberfeld (station_id: 7498)
- Hohenlohe (H) using meteorological data from the DWD-stations Rechbach (station_id: 2787), Stachenhausen (station_id: 6260) and Öhringen (station_id: 3761)

- `write_slurm_jobs.py`: Generate job scripts for computation on BwUniCluster 2.0 cluster using the SLURM workload manager
- `plot_time_series.py`: Plot time series
- `plot_nitrogen_balance.py`: Plot nitrogen balance of regions and crop rotation scenarios
- `assign_simulated_values_to_polygon.py`: Plot nitrogen balance of regions and crop rotation scenarios

---

The following folders contain the model setups. Each folder contains a subfolder
`input` from which the model reads the input data and a Python-script with the
implementation of the model setup and the corresponding job-script(s) for computation
on a cluster, and Python-scripts for post-processing. If the file name of the
job scripts contains `_gpu` computations run on GPU.

## input/crop_rotation_scenarios
The following crop combinations are rotated:
- Winter wheat, clover grass, clover grass, winter wheat
- Winter wheat, silage maize, winter wheat, silage maize
- Spring wheat, winter wheat, summer wheat, winter wheat
- Spring wheat, clover grass, clover grass, winter wheat
- Winter wheat, clover grass, clover grass, silage maize
- Winter wheat, sugar beet, winter wheat, silage maize
- Spring wheat, winter wheat, silage maize
- Spring wheat, winter wheat, winter wheat, winter oilseed rape
- Winter wheat, winter wheat, winter wheat, winter oilseed rape
- Winter wheat, soya bean (legume), winter wheat, winter oilseed rape
- Sugar beet, winter wheat, winter wheat, winter barley
- grain maize, winter wheat, winter oilseed rape, winter wheat
- Kernel maize, winter wheat, winter barley
- Kernel maize, winter wheat, clover grass, clover grass
- Winter wheat, silage maize, winter wheat, silage maize, mustard (catch crop)
- Spring wheat, winter wheat, summer wheat, winter wheat, mustard (catch crop)
- Winter wheat, sugar beet, winter wheat, silage maize, mustard (catch crop)
- Spring wheat, winter wheat, silage maize, mustard (catch crop)
- Spring wheat, winter wheat, winter wheat, winter rape, mustard (catch crop)
- Sugar beet, winter wheat, winter wheat, winter barley, mustard (catch crop)
- grain maize, winter wheat, winter rape, winter wheat, mustard (catch crop)
- Kernel maize, winter wheat, winter barley, mustard (catch crop)

The name of the folder contains the considered crop combinations. Each folder contains a file that describes the crop rotation in `crop_rotation.csv`.

## Hydrological simulations
### svat_crop
- `svat_crop.py`: SVAT-CROP-model setup
- `svat_crop_*.sh`: Job scripts to run the SVAT-CROP-model
- `merge_output.py`: Merges the model output into a single file

## Nitrate simulations
### svat_crop_nitrate
- `svat_crop_nitrate.py`: SVAT-CROP-NITRATE-model setup
- `svat_crop_nitrate_*.sh`: Job scripts to run the SVAT-CROP-model
- `merge_output.py`: Merges model output into a single file

---
## Columns of the attribute table
- fid: Identifier of the polygon
- SHP_ID: Identifier of the polygon
- Qsur_avg: Averaged annual sums of surface runoff for the period 2013-2022 (in mm/year)
- GC_avg: Averaged annual ground cover for the period 2013-2022 (-)
- NO3PERC_lowNfert_avg: Averaged annual sums of nitrate leaching using low nitrogen fertilization intensity for the period 2013-2022 (in kg N/ha/year)
- NO3PERC_mediumNfert_avg: Averaged annual sums of nitrate leaching using medium nitrogen fertilization intensity for the period 2013-2022 (in kg N/ha/year)
- NO3PERC_highNfert_avg: Averaged annual sums of nitrate leaching using high nitrogen fertilization intensity for the period 2013-2022 (in kg N/ha/year)

