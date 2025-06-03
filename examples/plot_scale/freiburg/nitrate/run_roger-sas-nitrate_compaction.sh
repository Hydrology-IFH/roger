#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate roger
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario summer-barley
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario winter-wheat
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario potato
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn_winter-wheat_winter-rape
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn_winter-wheat_winter-rape_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario winter-wheat_winter-rape
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario winter-wheat_silage-corn
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario winter-wheat_silage-corn_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario winter-wheat_soybean_winter-rape
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario silage-corn
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario silage-corn_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario grain-corn
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario summer-barley
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario winter-wheat
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario potato
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario grain-corn_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario grain-corn_winter-wheat_winter-rape
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario grain-corn_winter-wheat_winter-rape_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario winter-wheat_winter-rape
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario winter-wheat_silage-corn
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario winter-wheat_silage-corn_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario winter-wheat_soybean_winter-rape
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario silage-corn
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario silage-corn_yellow-mustard
python merge_output.py
python write_simulations_to_csv.py