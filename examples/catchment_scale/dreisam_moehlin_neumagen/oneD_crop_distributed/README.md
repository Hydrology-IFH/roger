# Spatially-distributed ONED-model of the Dreisam-Moehlin-Neumagen catchment

## Files
- `update_lu_id_of_roger_parameters.py`: Use areas of "Gemeinsamer Antrag" dataset to update lu_id in RoGeR parameter file
- `modify_parameters.py`: Fix inconsistencies in the RoGeR parameter file (e.g. zero soil depth)
- `write_crop_rotations.py`: `crops_2018-2022.nc` is used to write the RoGeR crop rotation file
- `write_crop_rotations_yellow_mustard.py`: `crops_2018-2022.nc` is used to write the RoGeR crop rotation file including yellow mustard as catch crop where possible (i.e. before summer crop)
- `write_stress_tests_meteo.py`: Write the meteorological stress test scenarios. The stress tests consider a combination of increasing stress duration and increasing stress magnitude. The magnitudes are defined in `*_stress_magnitude.csv` and are derived from `RheiKlim_Ann_Trends_DWD_*.pdf`. These figures are available at [RheiKlim](https://apps.hydro.uni-freiburg.de/de/RheiKlim/)
- `write_job_scripts_slurm.py`: Write SLURM job scripts to run the RoGeR simulations on BinAC2 computing cluster
- `crops_2018-2022.nc`: Crops of the period 2018-2022 encoded as RoGeR lu_id. Data was derived from "Gemeinsamer Antrag" dataset
- `*_stress_magnitude.csv`: Magnitudes to increase meteorological stress of the considered period. Magnitudes of spring precipitation correspond with magnitudes of the summer precipitation. The reason for this is that magnitudes of spring precipitation are too little or cause an increase of precipitation.
- `input/stress_tests_meteo/RheiKlim_Ann_Trends_DWD_*.pdf`: Figures to derive magnitudes of the meteorological stress tests.

## Workflow on BinAC2 using NVIDIA A100 GPU
1. Use [install_roger_gpu_on_binac2.sh](https://github.com/Hydrology-IFH/roger/blob/main/install_roger_gpu_on_binac2.sh) to install GPU-ready anaconda environment
2. `python update_lu_id_of_roger_parameters.py`
3. `python modify_parameters.py`
4. `python write_crop_rotations.py`
5. `python write_crop_rotations_yellow_mustard.py`
6. Use `RheiKlim_Ann_Trends_DWD_*.pdf` to define values in `*_stress_magnitude.csv` then run `python write_stress_tests_meteo.py`
7. `python write_job_scripts_slurm_gpu.py`
8. `./submit_gpu_jobs.sh`
9. `python write_data_job_scripts_slurm.py`
10. `./submit_data_jobs.sh`

## Stress-Test scenarios

### Climate stress
summer-drought:
- duration3-magnitude0: Summer drought of 2018 is repeated and occurs in 2016, 2017 and 2018 in current climate
- duration3-magnitude2: Summer drought of 2018 is repeated and occurs in 2016, 2017 and 2018 in future climate

long-term:
- duration0-magnitude2: Far future climate (2070 - 2099)

durationx: event is x years repeated.
magnitude1: using seasonal delta values of [RheiKlim](https://apps.hydro.uni-freiburg.de/de/RheiKlim/) for the near future (2040 - 2069)
magnitude2: using seasonale delta values of [RheiKlim](https://apps.hydro.uni-freiburg.de/de/RheiKlim/) for the far future (2070 - 2099)

### Agricultural management
- no-irrigation: no irrigation is applied on agricultural areas
- irrigation: irrigation is applied on agricultural areas
- no-yellow-mustard: No catch crop is considered before summer crops.
- yellow-mustard: Yellow mustard is cultivated before summer crops.
- soil-compaction: Soil compaction on agricultural areas is considered by decreasing air capacity and hydraulic conductivity.

## Files in output/
File names contain a combination of the stress test scenarios. For example:
- roger_long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction.nc: RoGeR-1D simulation result of the period 2013-2023 with a future climate, agricultural irrigation, soil compaction of agricultural areas.

If soil-compaction does not occur in the file name, the stress test scenario is not applied.


Data of `input/`, `output/`, and parameter files are stored on FUHYS018 in `StressRes_RoGeR-ModFlow/` since GitHub is not meant to be a large data storage facility. Please contact [Jürgen Strub](juergen.strub@hydrology.uni-freiburg.de) or [Markus Weiler](markus.weiler@hydrology.uni-freiburg.de) to access the data.