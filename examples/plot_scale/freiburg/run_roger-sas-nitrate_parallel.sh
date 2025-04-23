#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate roger

cd nitrate
nohup ./run_roger-sas-nitrate_no_irrigation.sh &
nohup ./run_roger-sas-nitrate_no_irrigation_compaction.sh &
# nohup ./run_roger-sas-nitrate_20-ufc.sh &
# nohup ./run_roger-sas-nitrate_35-ufc.sh &
# nohup ./run_roger-sas-nitrate_50-ufc.sh &
nohup ./run_roger-sas-nitrate_crop-specific.sh &
# nohup ./run_roger-sas-nitrate_20-ufc_compaction.sh &
# nohup ./run_roger-sas-nitrate_35-ufc_compaction.sh &
# nohup ./run_roger-sas-nitrate_50-ufc_compaction.sh &
nohup ./run_roger-sas-nitrate_crop-specific_compaction.sh &