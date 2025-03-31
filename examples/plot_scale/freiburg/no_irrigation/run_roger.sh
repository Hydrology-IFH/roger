#!/bin/bash
# eval "$(conda shell.bash hook)"
# conda activate roger
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario grain-corn
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario silage-corn
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario summer-barley
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario clover
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario faba-bean
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario potato-early
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario sugar-beet
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario vegetables
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario strawberry
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario asparagus
# python merge_output.py
# python write_simulations_to_csv.py