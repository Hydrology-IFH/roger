#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate roger
python write_parameters_to_netcdf.py
python svat_crop.py -b numpy -d cpu --irrigation-scenario 35-ufc --crop-rotation-scenario grain-corn_winter-wheat_winter-rape
python svat_crop.py -b numpy -d cpu --irrigation-scenario 35-ufc --crop-rotation-scenario grain-corn_winter-wheat_winter-rape_yellow-mustard
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn_winter-wheat_winter-rape
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn_winter-wheat_winter-rape_yellow-mustard
python merge_output.py
python simulations_to_csv.py