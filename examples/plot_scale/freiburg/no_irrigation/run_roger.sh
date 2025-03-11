#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate roger
python write_parameters_to_netcdf.pypython svat_crop.py -b numpy -d cpu --crop-rotation-scenario winter-wheat_clover
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario winter-wheat_silage-corn
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario summer-wheat_winter-wheat
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario summer-wheat_clover_winter-wheat
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario winter-wheat_clover_silage-corn
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario winter-wheat_sugar-beet_silage-corn
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario summer-wheat_winter-wheat_silage-corn
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario summer-wheat_winter-wheat_winter-rape
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario winter-wheat_winter-rape
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario winter-wheat_soybean_winter-rape
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario sugar-beet_winter-wheat_winter-barley
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario grain-corn_winter-wheat_winter-rape
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario grain-corn_winter-wheat_winter-barley
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario grain-corn_winter-wheat_clover
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario winter-wheat_silage-corn_yellow-mustard
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario summer-wheat_winter-wheat_yellow-mustard
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario winter-wheat_sugar-beet_silage-corn_yellow-mustard
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario summer-wheat_winter-wheat_silage-corn_yellow-mustard
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario summer-wheat_winter-wheat_winter-rape_yellow-mustard
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario sugar-beet_winter-wheat_winter-barley_yellow-mustard
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario grain-corn_winter-wheat_winter-rape_yellow-mustard
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario grain-corn_winter-wheat_winter-barley_yellow-mustard
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario miscanthus
python svat_crop.py -b numpy -d cpu --crop-rotation-scenario bare-grass
python merge_output.pypython simulations_to_csv.py