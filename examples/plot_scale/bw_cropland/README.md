# Crop irrigation demand and the impact of irrigation on groundwater recharge and nitrate leaching in Baden-Wuerttemberg, Germany

This modelling experiment investigates the impact of crop irrigation on groundwater recharge and nitrate leaching. The impact is systematically investigated using a combination of soil types, irrigation demand rules and crop rotation scenarios.

- `figures/`: contains figures
- `output/`: contains the simulations of RoGeR
- `output/*/simulation.csv`: contains the time series of the variables simulated by RoGeR (precip: precipitation in mm/day; pet: potential evapotranspiration in mm/day; pt: potential transpiration in mm/day; irrig: irrigation in mm/day; canopy_cover: crop canopy cover in -; z_root: crop root depth in mm; theta_rz: soil water content of the upper soil layer in -; theta_irrig: irrigation threshold of soil water content of the upper soil layer in -; theta_fc: soil water content at field capacity in -; transp: actual transpiration in mm/day; evap_soil: actual soil evaporation in mm/day; perc: percolation in mm/day; irrigation_demand: soil water deficit in mm; root_ventilation: available soil air for root ventilation in %; heat_stress: occurence of crop heat stress (1=occurence and 0=no occurence); lu_id: land use identifier or RoGeR; crop_type: name of the crop)
- `input/`: contains meteorological data (`PREC.txt`, `PET.txt`, `TA.txt`) of the DWD stations covering BW and the defined crop rotations (`crop_rotation_scenarios/`) from 2000 to 2024
- `input/stress_tests_meteo/`: contains projected meteorological data (`PREC.txt`, `PET.txt`, `TA.txt`) of the DWD stations covering BW and the defined crop rotations (`crop_rotation_scenarios/`) from 2000 to 2024
- `crop_water_stress.csv`: contains fraction of usable field capacity when crop water stress starts. IMPORTANT: Do not modify the file.
- `parameters.csv`: contains the soil hydraulic parameters of RoGeR for three different soil types (z_soil: soil depth in mm; dmpv: density of vertical macropores in 1/$m^2$; lmpv: length of vertical macropores in mm; theta_ac: air capacity in -; theta_ufc: usable field capacity in -; theta_pwp: permanent wilting point in -; ks: saturated hydraulic conductivity in mm/h; kf: hydraulic conductivity of bedrock in mm/h; soil_fertility: soil fertility; clay: clay content in -)
- `representative_agricultural_soil_types_parameters.csv`: contains the soil hydraulic parameters of RoGeR for the representative agricultural soil types (z_soil: soil depth in mm; dmpv: density of vertical macropores in 1/$m^2$; lmpv: length of vertical macropores in mm; theta_ac: air capacity in -; theta_ufc: usable field capacity in -; theta_pwp: permanent wilting point in -; ks: saturated hydraulic conductivity in mm/h; kf: hydraulic conductivity of bedrock in mm/h; soil_fertility: soil fertility; clay: clay content in -)
- `config.yml`: configuration file. Irrigation rules and crop rotation scenarios should be defined from the list below.
- `calculate_gw_recharge_dyck-chardabellas-1963.py`: Calculates the annual groundwater recharge and average groundwater recharge using the method of Dyck & Chardbellas (1963) as presented in Hoelting (2013) p.244f. The script requires a precipitation time series with 10-minutes time step. Precipiation data can be downloaded at [WeatherDB](https://apps.hydro.uni-freiburg.de/de/weatherdb/get_ts/).
- `write_stress_tests_meteo.py`: Manipulate meteorological Data of DWD stations using seasonal magnitudes and repeat specific events
- `write_crop_rotations.py`: Write crop rotations using Data provided by University Hohenheim (`crop_rotations.csv` and `crop_rotations_catchcrop.csv`)
- `subregions_crop_rotations.csv`: Contains possible crop rotations for each subregion
- `output/no-irrigation/`: contains the calculations of `calculate_gw_recharge_dyck-chardabellas-1963.py`
- `calculate_nitrate_leaching_thuenen.py`: Calculates the load of annual nitrate leaching and average nitrate for a given crop rotation using the Thuenen method i.e. 30% of the applied nitrogen fertiliser. The nitrate leaching is calculated for three different fertilisation intensities (low, medium and high). The script requires an annual time series of the summer and winter crops (see `/input/crop_rotation_scenarios/` for more examples).
- `output/nitrate/thuenen/`: contains the calculations of `calculate_nitrate_leaching_thuenen.py`

Available soil types:
- sandy soil type
- silty soil type
- clayey soil type
- Further soil types can be added to `parameters.csv`

Available irrigation demand rules:
- no_irrigation: No irrigation is applied
- crop-specific: Irrigation demand is specifically calculated for each crop (default)

## Meteorological stress tests
Magnitudes of the meteorological variables and the season are defined in `*_stress_magnitude.csv`. We used information from [RheiKlim](https://apps.hydro.uni-freiburg.de/de/RheiKlim/map) (see also `A4_Ann_Trends_*.pdf`). The following meteorological stress tests are available:
- spring drought of 2020 three times in a row (starting in year 2018) in current climate --> `/input/stress_test_meteo/spring-drought/duration3_magnitude0/`
- spring drought of 2020 three times in a row (starting in year 2018) in far future climate (2070 - 2100) --> `/input/stress_test_meteo/spring-drought/duration3_magnitude2/`
- summer drought of 2018 three times in a row (starting in year 2016) in current climate --> `/input/stress_test_meteo/summer-drought/duration3_magnitude0/`
- summer drought of 2018 three times in a row (starting in year 2016) in far future climate (2070 - 2100) --> `/input/stress_test_meteo/summer-drought/duration3_magnitude2/`
- far future climate (2070 - 2100) --> `/input/stress_test_meteo/long-term/duration0_magnitude2/`

durationx: event is x years repeated.
magnitude1: using seasonal delta values of [RheiKlim](https://apps.hydro.uni-freiburg.de/de/RheiKlim/) for the near future (2040 - 2069)
magnitude2: using seasonale delta values of [RheiKlim](https://apps.hydro.uni-freiburg.de/de/RheiKlim/) for the far future (2070 - 2099)

Crop rotations are repeated after three to four years. The available crop rotations for each subregion (i.e. 30 km radius around DWD station) are listed in `subregions_crop_rotations.csv`.

Below the implementaion of 4 agricultural management scenarios is described. The simulations are run for available crop rotations using current meteorological conditions (base) and different meteorological stress tests.
### no-irrigation
No irrigation is applied i.e. irrigation demand is calculated without irrigation and subsoil compaction is not considered.

- `svat_crop.py`: Setup of the RoGeR model
- `merge_output.py`: Merges the RoGeR model output into a single *.nc-file
- `merge_output.sh`: Runs `merge_output.py` as computing job on BinAC2
- `write_simulations_to_csv.py`: Writes simulations *.nc-file to *.csv-file
- `write_job_script.py`: Writes shell script to generate the simulations for the available soil types, irrigation scenarios and crop roations
- `write_job_script_slurm.py`: Writes shell scripts to run the simulations on BinAC2 computing cluster
- `submit_jobs.sh`: Submit the simulations as job scripts on BinAC2
- `run_roger.sh`: Runs the RoGeR model to generate the simulations

### no-irrigation_soil-compaction
No irrigation is applied i.e. irrigation demand is calculated without irrigation. Additionally, soil compaction by agricultural wheel trafficking is considered. Soil compaction is implemented by reducing the saturated hydraulic conductivity and soil air capacity of the subsoil.

- `svat_crop.py`: Setup of the RoGeR model
- `merge_output.py`: Merges the RoGeR model output into a single *.nc-file
- `merge_output.sh`: Runs `merge_output.py` as computing job on BinAC2
- `write_simulations_to_csv.py`: Writes simulations *.nc-file to *.csv-file
- `write_simulations_to_csv.sh`: Runs `write_simulations_to_csv.py` as computing job on BinAC2
- `write_job_script.py`: Writes shell script to generate the simulations for the available soil types, irrigation scenarios and crop roations
- `write_job_script_slurm.py`: Writes shell scripts to run the simulations on BinAC2 computing cluster
- `submit_jobs.sh`: Submit the simulations as job scripts on BinAC2
- `run_roger.sh`: Runs the RoGeR model to generate the simulations

### irrigation
30 mm per day are irrigated if crop specific irrgation demand occurs and subsoil compaction is not considered.

- `svat_crop.py`: Setup of the RoGeR model
- `merge_output.py`: Merges the RoGeR model output into a single *.nc-file
- `merge_output.sh`: Runs `merge_output.py` as computing job on BinAC2
- `write_simulations_to_csv.py`: Writes simulations *.nc-file to *.csv-file
- `write_simulations_to_csv.sh`: Runs `write_simulations_to_csv.py` as computing job on BinAC2
- `write_job_script.py`: Writes shell script to generate the simulations for the available soil types, irrigation scenarios and crop roations
- `write_job_script_slurm.py`: Writes shell scripts to run the simulations on BinAC2 computing cluster
- `submit_jobs.sh`: Submit the simulations as job scripts on BinAC2
- `run_roger.sh`: Runs the RoGeR model to generate the simulations

### irrigation_soil-compaction
30 mm per day are irrigated according to four irrigation demand rules. Additionally, soil compaction by agricultural wheel trafficking is considered. Soil compaction is implemented by reducing the saturated hydraulic conductivity and soil air capacity of the subsoil.

- `svat_crop.py`: Setup of the RoGeR model
- `merge_output.py`: Merges the RoGeR  model output into a single *.nc-file
- `merge_output.sh`: Runs `merge_output.py` as computing job on BinAC2
- `write_simulations_to_csv.py`: Writes simulations *.nc-file to *.csv-file
- `write_simulations_to_csv.sh`: Runs `write_simulations_to_csv.py` as computing job on BinAC2
- `write_job_script.py`: Writes shell script to generate the simulations for the available soil types, irrigation scenarios and crop roations
- `write_job_script_slurm.py`: Writes shell scripts to run the simulations on BinAC2 computing cluster
- `submit_jobs.sh`: Submit the simulations as job scripts on BinAC2
- `run_roger.sh`: Runs the RoGeR model to generate the simulations

### nitrate
Nitrate leaching is simulated considering all combinations: no irrigation and no soil compaction, no irrigation and soil compaction, irrigation and no soil compaction, irrigation and soil compaction.

- `parameters_sas_nitrate.nc`: contains the SAS and nitrate parameters of RoGeR for three different soil types
- `svat_crop_nitrate.py`: Setup of the RoGeR-SAS model to simulate the nitrate transport 
- `merge_output.py`: Merges the model output into a single *.nc-file
- `write_simulations_to_csv.py`: Writes simulations to *.csv-file
- `write_simulations_to_csv.sh`: Runs `write_simulations_to_csv.py` as computing job on BinAC2
- `write_job_script.py`: Writes shell script to generate the simulations for the available soil types, irrigation scenarios and crop roations
- `write_job_script_slurm.py`: Writes shell scripts to run the simulations on BinAC2 computing cluster
- `submit_jobs.sh`: Submit the simulations as job scripts on BinAC2
- `run_roger.sh`: Runs the RoGeR model to generate the simulations
- `submit_jobs.sh`: Submit the simulations as job scripts on BinAC2

## Workflow
! Windows user may change from `/` to `\` in the provided *.sh-files. Please check beforehand. !

1. Install RoGeR and the required Python libraries using Anaconda `conda env create -f conda-environment.yml`
2. After successfull installation activate the conda environment `roger`
3. Open the terminal and move to the project directory.
4. Set the model parameters in `parameters.csv` (Skip this step if you agree using the provided parameters)
5. Set the considered irrigation demand rules and crop rotation scenarios in `config.yml`
6. Run `python write_parameters_to_netcdf.py` to write `parameters.csv` to NetCDF format
7. Run `python write_crop_rotations.py` to write `subregions_crop_rotations.csv` and crop rotation input files for RoGeR
8. Run `python write_stress_tests_meteo.py` to write meteorological data applying stress test
9. Run `python calculate_area_of_representative_soil_types.py` to calculate areas which can be used for weighted averaging

### Computation on BinAC2

Move in `cd no-irrigation_soil-compaction`, `cd no-irrigation/`, `cd irrigation_soil-compaction` or `cd irrigation`:
1. Run `python write_job_scripts_slurm.py` to write the computing jobs
2. Run `./submit_jobs.sh` to submit the jobs to the queue
3. After all jobs are finalised, run `merge_output.sh` to merge the RoGeR output files
4. Run `write_simulations_to_csv.sh` to convert NetCDF format to csv format (optionally)
5. Analyse the data (e.g. `python impact_analysis.py`)

Move in `cd nitrate`
6. Run `python write_job_scripts_slurm.py` to write the computing jobs
7. Run `./submit_jobs.sh` to submit the jobs to the queue
8. After all jobs are finalised, run `merge_output.sh` to merge the RoGeR output files
9. Run `write_simulations_to_csv.sh` to convert NetCDF format to csv format (optionally)
10. Analyse the data

### Computation on local computer

1. Run `python write_job_scripts.py` to write the computing jobs
2. Run `./submit_jobs.sh` to start the job queue
3. After all jobs are finalised, run `merge_output.sh` to merge the RoGeR output files
4. Run `write_simulations_to_csv.sh` to convert NetCDF format to csv format (optionally)
5. Analyse the data (e.g. `python impact_analysis.py`)