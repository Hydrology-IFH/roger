#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate roger
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn_winter-wheat_winter-barley
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn_winter-wheat_clover
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario summer-wheat_winter-wheat_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario winter-wheat_sugar-beet_silage-corn_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario summer-wheat_winter-wheat_silage-corn_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario summer-wheat_winter-wheat_winter-rape_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario sugar-beet_winter-wheat_winter-barley_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn_winter-wheat_winter-barley_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario grain-corn_winter-wheat_winter-barley
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario grain-corn_winter-wheat_clover
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario summer-wheat_winter-wheat_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario winter-wheat_sugar-beet_silage-corn_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario summer-wheat_winter-wheat_silage-corn_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario summer-wheat_winter-wheat_winter-rape_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario sugar-beet_winter-wheat_winter-barley_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no_irrigation --crop-rotation-scenario grain-corn_winter-wheat_winter-barley_yellow-mustard
python merge_output.py
python write_simulations_to_csv.py