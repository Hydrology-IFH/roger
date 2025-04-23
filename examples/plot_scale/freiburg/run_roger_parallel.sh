#!/bin/bash

cd no-irrigation
nohup ./run_roger.sh &

cd ../no-irrigation_soil-compaction
nohup ./run_roger.sh &

cd ../irrigation
nohup ./run_roger_crop-specific.sh &
nohup ./run_roger_20-ufc.sh &
nohup ./run_roger_35-ufc.sh &
nohup ./run_roger_50-ufc.sh &

cd ../irrigation_soil-compaction
nohup ./run_roger_crop-specific.sh &
nohup ./run_roger_20-ufc.sh &
nohup ./run_roger_35-ufc.sh &
nohup ./run_roger_50-ufc.sh &