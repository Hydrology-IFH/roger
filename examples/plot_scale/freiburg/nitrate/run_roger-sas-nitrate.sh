#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate roger
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario crop-specific --crop-rotation-scenario sugar-beet
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario crop-specific --crop-rotation-scenario sugar-beet_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario crop-specific --crop-rotation-scenario vegetables
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario crop-specific --crop-rotation-scenario strawberry
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario crop-specific --crop-rotation-scenario winter-wheat_clover
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario crop-specific --crop-rotation-scenario summer-wheat_winter-wheat
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario crop-specific --crop-rotation-scenario summer-wheat_clover_winter-wheat
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario sugar-beet
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario sugar-beet_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario vegetables
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario strawberry
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario winter-wheat_clover
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario summer-wheat_winter-wheat
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario summer-wheat_clover_winter-wheat
python merge_output.py
python write_simulations_to_csv.py