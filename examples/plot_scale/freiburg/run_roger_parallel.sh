#!/bin/bash

python no_irrigation/svat_crop.py -b numpy -d cpu --crop-rotation-scenario grain-corn_winter-wheat_winter-rape
python no_irrigation/svat_crop.py -b numpy -d cpu --crop-rotation-scenario grain-corn_winter-wheat_winter-rape_yellow-mustard

python irrigation/svat_crop.py -b numpy -d cpu --irrigation-scenario 35-ufc --crop-rotation-scenario grain-corn_winter-wheat_winter-rape
python irrigation/svat_crop.py -b numpy -d cpu --irrigation-scenario 35-ufc --crop-rotation-scenario grain-corn_winter-wheat_winter-rape_yellow-mustard
python irrigation/svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn_winter-wheat_winter-rape
python irrigation/svat_crop.py -b numpy -d cpu --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn_winter-wheat_winter-rape_yellow-mustard