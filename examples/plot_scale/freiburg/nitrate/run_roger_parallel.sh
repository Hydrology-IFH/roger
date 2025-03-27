#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate roger
nohup ./run_roger_no_irrigation.sh &
nohup ./run_roger_35-ufc.sh &
nohup ./run_roger_45-ufc.sh &
nohup ./run_roger_50-ufc.sh &
nohup ./run_roger_80-ufc.sh &
nohup ./run_roger_crop-specific.sh &

python merge_output.py
python write_simulations_to_csv.py