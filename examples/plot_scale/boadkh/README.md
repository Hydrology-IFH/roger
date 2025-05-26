# Modelling nitrate leaching using RoGeR of five agricultural regions (BOADKH) in Baden-Wuerttemberg, Germany
This repository provides scripts and data to calculate surface runoff, percolation/groundwater recharge and nitrate leaching of five agricultural regions (BOADKH) in Baden-Wuerttemberg, Germany over the period 2013 to 2022 using the hydrological model RoGeR. The following five regions (BOADKH) in Baden-Wuerttemberg, Germany are considered:
- Lake Constance (B) using meteorological data of the DWD-stations Gottmadingen (station_id: 1711), Stockach (station_id: 4881) and Weingarten (Kr. Ravensburg: station_id: 4094)
- Upper rhine valley (O) using meteorological data of the DWD-stations Müllheim im Markgräflerland (station_id: 259), Freiburg (station_id: 1443) and Lahr (station_id: 2812)
- Alb-Danube (AD) using meteorological data of the DWD-stations Hayingen (station_id: 2072), Ehingen-Kirchen (station_id: 3418) and Merklingen (station_id: 2814)
- Kraichgau (K) using meteorological data of the DWD-stations Bruchsal-Heidelsheim (station_id: 731), Bretten (Kreis Karlsruhe; station_id: 7490) and Eppingen-Elsenz (station_id: 1255)
- Hohenlohe (H) using meteorological data of the DWD-stations Kupferzell-Rechbach (station_id: 2787), Vellberg-Kleinaltdorf (station_id: 5206) and Öhringen (station_id: 3761)

---

## input
Contains meteorological time series of 15 DWD stations and crop rotation scenarios (`/crop_rotation_scenarios`). The meteorological data has been downloaded from [WeatherDB](https://apps.hydro.uni-freiburg.de/de/weatherdb/).

- `gottmadingen/`: precipitation (`PREC.txt`), air temperature (`TA.txt`), and potential evapotranspiration (`PET.txt`) time series collected at the DWD-station Gottmadingen (station_id: 1711) for the period 2013 to 2022.
- `stockach/`: precipitation (`PREC.txt`), air temperature (`TA.txt`), and potential evapotranspiration (`PET.txt`) time series collected at the DWD-station Stockach (station_id: 4881) for the period 2013 to 2022.
- `weingarten/`: precipitation (`PREC.txt`), air temperature (`TA.txt`), and potential evapotranspiration (`PET.txt`) time series collected at the DWD-station Weingarten (Kr. Ravensburg: station_id: 4094) for the period 2013 to 2022.
- `muellheim/`: precipitation (`PREC.txt`), air temperature (`TA.txt`), and potential evapotranspiration (`PET.txt`) time series collected at the DWD-station Müllheim im Markgräflerland (station_id: 259) for the period 2013 to 2022.
- `freiburg/`: precipitation (`PREC.txt`), air temperature (`TA.txt`), and potential evapotranspiration (`PET.txt`) time series collected at the DWD-station Freiburg (station_id: 1443) for the period 2013 to 2022.
- `lahr/`: precipitation (`PREC.txt`), air temperature (`TA.txt`), and potential evapotranspiration (`PET.txt`) time series collected at the DWD-station Lahr (station_id: 2812) for the period 2013 to 2022.
- `hayingen/`: precipitation (`PREC.txt`), air temperature (`TA.txt`), and potential evapotranspiration (`PET.txt`) time series collected at the DWD-station Hayingen (station_id: 2072) for the period 2013 to 2022.
- `ehingen-kirchen/`: precipitation (`PREC.txt`), air temperature (`TA.txt`), and potential evapotranspiration (`PET.txt`) time series collected at the DWD-station Ehingen-Kirchen (station_id: 3418) for the period 2013 to 2022.
- `merklingen/`: precipitation (`PREC.txt`), air temperature (`TA.txt`), and potential evapotranspiration (`PET.txt`) time series collected at the DWD-station Merklingen (station_id: 2814) for the period 2013 to 2022.
- `bruchsal-heidelsheim/`: precipitation (`PREC.txt`), air temperature (`TA.txt`), and potential evapotranspiration (`PET.txt`) time series collected at the DWD-station Bruchsal-Heidelsheim (station_id: 731) for the period 2013 to 2022.
- `bretten/`: precipitation (`PREC.txt`), air temperature (`TA.txt`), and potential evapotranspiration (`PET.txt`) time series collected at the DWD-station Bretten (Kreis Karlsruhe; station_id: 7490) for the period 2013 to 2022.
- `eppingen-elsenz/`: precipitation (`PREC.txt`), air temperature (`TA.txt`), and potential evapotranspiration (`PET.txt`) time series collected at the DWD-station Eppingen-Elsenz (station_id: 1255) for the period 2013 to 2022.
- `kupferzell/`: precipitation (`PREC.txt`), air temperature (`TA.txt`), and potential evapotranspiration (`PET.txt`) time series collected at the DWD-station Kupferzell-Rechbach (station_id: 2787) for the period 2013 to 2022.
- `vellberg-kleinaltdorf/`: precipitation (`PREC.txt`), air temperature (`TA.txt`), and potential evapotranspiration (`PET.txt`) time series collected at the DWD-station Vellberg-Kleinaltdorf (station_id: 5206) for the period 2013 to 2022.
- `oehringen/`: precipitation (`PREC.txt`), air temperature (`TA.txt`), and potential evapotranspiration (`PET.txt`) time series collected at the DWD-station Öhringen (station_id: 3761) for the period 2013 to 2022.


### crop_rotation_scenarios
The following crop combinations are rotated:
- Winter wheat, clover grass, clover grass, winter wheat
- Winter wheat, silage maize, winter wheat, silage maize
- Spring wheat, winter wheat, spring wheat, winter wheat
- Spring wheat, clover grass, clover grass, winter wheat
- Winter wheat, clover grass, clover grass, silage maize
- Winter wheat, sugar beet, winter wheat, silage maize
- Spring wheat, winter wheat, silage maize
- Spring wheat, winter wheat, winter wheat, winter oilseed rape
- Winter wheat, winter wheat, winter wheat, winter oilseed rape
- Winter wheat, soya bean (legume), winter wheat, winter oilseed rape
- Sugar beet, winter wheat, winter wheat, winter barley
- grain maize, winter wheat, winter oilseed rape, winter wheat
- grain maize, winter wheat, winter barley
- grain maize, winter wheat, clover grass, clover grass
- Winter wheat, mustard (catch crop), silage maize
- mustard (catch crop), spring wheat, winter wheat
- Winter wheat, mustard (catch crop), sugar beet, winter wheat, mustard (catch crop), silage maize
- mustard (catch crop), spring wheat, winter wheat, mustard (catch crop), silage maize
- mustard (catch crop), spring wheat, winter wheat, winter wheat, winter rape
- mustard (catch crop), sugar beet, winter wheat, winter wheat, winter barley
- mustard (catch crop), grain maize, winter wheat, winter rape, winter wheat
- mustard (catch crop), grain maize, winter wheat, winter barley
The name of the folder contains the considered crop combinations. Each folder contains a file that describes the crop rotation in `crop_rotation.csv` by `lu_id` which is required by RoGeR.

## Pre-processing 
- `write_slurm_jobs.py`: Generate job scripts for computation on BwUniCluster 3.0 cluster using the SLURM workload manager
- `extend_and_shift_meteo.py`: Extend the meteorological time series and shift by 1 year to account for extreme years

## Hydrological simulations (surface runoff and percolation/groundwater recharge)
### svat_crop
Contains Python scripts and shell scripts to perform the hydrological simulations.

- `write_parameters.py`: Writes the model parameters to `parameters.csv` and `parameters.nc`
- `parameters.nc`/`parameters.csv`: Parameters used by the SVAT-CROP-model of RoGeR
- `svat_crop.py`: SVAT-CROP-model setup to calculate the surface runoff and percolation/groundwater recharge
- `svat_crop_*.sh`: Job scripts to run the SVAT-CROP-model
- `merge_output.py`: Merges the model output into a single file
- `merge_output.sh`: Job scripts to merge model output into single files
- `chain_jobs.sh`: Submit job scripts to queue

Model output is written to `/output/svat_crop/`.

## Nitrate simulations (nitrate leaching)
### svat_crop_nitrate
Contains Python scripts and shell scripts to perform the nitrate simulations.

- `write_parameters.py`: Writes the model parameters to `parameters.csv` and `parameters.nc`
- `parameters.nc`/`parameters.csv`: Parameters used by the SVAT-CROP-NITRATE-model of RoGeR
- `svat_crop_nitrate.py`: SVAT-CROP-NITRATE-model setup to calculate the nitrate leaching by an advective-dispersive transport assumption using power law distribution functions for the StorAge selection (SAS).
- `svat_crop_nitrate_*.sh`: Job scripts to run the SVAT-CROP-NITRATE-model
- `merge_output.py`: Merges model output into a single file
- `merge_output.sh`: Job scripts to merge model output into single files
- `chain_jobs.sh`: Submit job scripts to queue

Model output is written to `/output/svat_crop_nitrate/`.

## Analyse the simulations
- `assign_simulated_values_to_geometries.py`: Write the geometries as .shp-file and the values as .csv-file. Values and geometries are linked by the OID. Path to files have to be adjusted in the script.
- `calculate_area_of_representative_soil_types.py`: Calculates the area of the representative agricultural soil types
- `barplot_nitrate_leaching.py`: Plot the simulated nitrate leaching using bar plots
- `boxplot_nitrate_leaching.py`: Plot the simulated nitrate leaching using box plots
- `make_maps_of_nitrate_leaching.py`: Plot the simulated nitrate leaching as maps for each crop and fertilisation intensity
- `plot_nitrate_leaching_time_series.py`: Plot the simulated nitrate leaching time series
- `plot_annual_nitrogen_balance.py`: Plot the annual nitrogen balance of the locations, crop rotation scenarios and fertilisation intensities
- `plot_surface_runoff_and_ground_cover.py`: Plot the daily and annual time series of simulated surface runoff and ground cover
- `analyse_vulnerability_for_nitrate_leaching.py`: Correlation analysis between soil physical properties and groundwater recharge/nitrate leaching

## output
*.nc model output files of RoGeR are written to this directory.

## figures
All figures that are generated by the provided scripts are stored to this directory.

## Workflow
! Windows user may change from `/` to `\` in the provided *.sh-files. Please check beforehand. !


### Installation of RoGeR
1. Install RoGeR and the required Python libraries using Anaconda `conda env create -f conda-environment.yml`
2. Activate the anaconda environment `conda activate roger`
3. Install additional Python libraries:
```
conda install xarray
conda install geopandas
# libraries below are only required by make_maps_of_nitrate_leaching.py
conda install contexily
pip install matplotlib-map-utils
pip install adjustText
```

### Pre-processing
4. Prepare the meteorological data as input to RoGeR: `python extend_and_shift_meteo.py`
5. Write the job scripts: `python write_slurm_jobs.py`

### Simulate the surface runoff and percolation/groundwater recharge 
6. Write the job scripts: `python write_slurm_jobs.py`
7. Move to `svat_crop/`: `cd svat_crop`
8. Write the model parameters: `python write_parameters.py`
9. Simulate the surface runoff and percolation/groundwater recharge as a chain job: 
```
chmod +x chain_jobs.sh
nohup ./chain_jobs.sh &
```
10. After the simulations are finalized, merge the model output `python merge_output.py` or submit `merge_output.sh` as a compute job

### Simulate the nitrate leaching
! In order to simulate nitrate leaching using SVAT-CROP-NITRATE model of RoGeR, simulations of the SVAT-CROP model need to be completed first !

11. Move to `svat_crop_nitrate/`: `cd svat_crop_nitrate`
12. Write the model parameters: `python write_parameters.py`
13. Simulate the nitrate leaching as a chain job: 
```
chmod +x chain_jobs.sh
nohup ./chain_jobs.sh &
```
14. After the simulations are finalzied, merge the model output `python merge_output.py` or submit `merge_output.sh` as a compute job

### Write geometries and values for the web tool NBiomasseBW 
15. Write the data: `python assign_simulated_values_to_geometries.py`

### Plot and analyse the simulations
16. Calculate areas: `python calculate_area_of_representative_soil_types.py`
17. Generate plots:
```
python plot_nitrate_leaching_time_series.py
python barplot_nitrate_leaching.py
python boxplot_nitrate_leaching.py
python make_maps_of_nitrate_leaching.py
python plot_annual_nitrogen_balance.py
python plot_surface_runoff_and_ground_cover.py
```
17. Conduct vulnerability analysis: `python analyse_vulnerability_for_nitrate_leaching.py`

## Future work
`write_input_data_from_climate_projections.py` prepares the meteorological input data from climate projections (CCCma-CanESM2_CCLM4-8-17 and MPI-M-MPI-ESM-LR_RCA4) for the considered 15 DWD-stations. This data can bed used to investigate groundwater recharge and nitrate leaching.

