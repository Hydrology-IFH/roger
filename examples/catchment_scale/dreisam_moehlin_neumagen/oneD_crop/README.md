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
- `RheiKlim_Ann_Trends_DWD_*.pdf`: Figures to derive magnitudes of the meteorological stress tests.

## Workflow
1. `python update_lu_id_of_roger_parameters.py`
2. `python modify_parameters.py`
3. `python write_crop_rotations.py`
4. `python write_crop_rotations_yellow_mustard.py`
6. Use `RheiKlim_Ann_Trends_DWD_*.pdf` to define values in `*_stress_magnitude.csv` then run `python write_stress_tests_meteo.py`
7. `python write_job_scripts_slurm.py`