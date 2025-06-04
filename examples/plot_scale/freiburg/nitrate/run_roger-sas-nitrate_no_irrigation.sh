#!/bin/bash
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario sugar-beet
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario sugar-beet_yellow-mustard
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario vegetables
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario strawberry
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario winter-wheat_clover
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario summer-wheat_winter-wheat
python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario summer-wheat_clover_winter-wheat
