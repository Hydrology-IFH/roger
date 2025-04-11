#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate roger
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario grain-corn
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario summer-barley
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario winter-wheat
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario potato
python merge_output.py
python write_simulations_to_csv.py