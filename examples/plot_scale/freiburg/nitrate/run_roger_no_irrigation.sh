#!/bin/bash
python svat_crop_nitrate.py -b numpy -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario grain-corn
python svat_crop_nitrate.py -b numpy -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario grain-corn_yellow-mustard
python svat_crop_nitrate.py -b numpy -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario grain-corn_winter-wheat_winter-rape
python svat_crop_nitrate.py -b numpy -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario grain-corn_winter-wheat_winter-rape_yellow-mustard
python svat_crop_nitrate.py -b numpy -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario winter-wheat
python svat_crop_nitrate.py -b numpy -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no_irrigation --crop-rotation-scenario winter-wheat_winter-rape
