#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate roger
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario sugar-beet
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario sugar-beet_yellow-mustard
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario vegetables
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario strawberry
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario winter-wheat_clover
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario summer-wheat_winter-wheat
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario winter-wheat_clover_silage-corn
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario winter-wheat_sugar-beet_silage-corn
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario summer-wheat_winter-wheat_silage-corn
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario summer-wheat_winter-wheat_winter-rape
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario sugar-beet_winter-wheat_winter-barley
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn_winter-wheat_winter-barley
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn_winter-wheat_clover
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario summer-wheat_winter-wheat_yellow-mustard
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario winter-wheat_sugar-beet_silage-corn_yellow-mustard
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario summer-wheat_winter-wheat_silage-corn_yellow-mustard
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario summer-wheat_winter-wheat_winter-rape_yellow-mustard
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario sugar-beet_winter-wheat_winter-barley_yellow-mustard
python svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn_winter-wheat_winter-barley_yellow-mustard
python merge_output.py
python write_simulations_to_csv.py