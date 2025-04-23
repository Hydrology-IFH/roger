#!/bin/bash

nohup ./nitrate/run_roger_no_irrigation.sh &
nohup ./nitrate/run_roger_no_irrigation_compaction.sh &
# nohup ./nitrate/run_roger_20-ufc.sh &
# nohup ./nitrate/run_roger_35-ufc.sh &
# nohup ./nitrate/run_roger_50-ufc.sh &
nohup ./nitrate/run_roger_crop-specific.sh &
# nohup ./nitrate/run_roger_20-ufc_compaction.sh &
# nohup ./nitrate/run_roger_35-ufc_compaction.sh &
# nohup ./nitrate/run_roger_50-ufc_compaction.sh &
nohup ./nitrate/run_roger_crop-specific_compaction.sh &