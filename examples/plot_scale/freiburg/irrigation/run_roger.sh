#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate roger
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn_yellow-mustard
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn_winter-wheat_winter-rape
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn_winter-wheat_winter-rape_yellow-mustard
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario winter-wheat
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario winter-wheat_winter-rape
python merge_output.py
python write_simulations_to_csv.py