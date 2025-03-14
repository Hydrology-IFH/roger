#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate roger
./run_roger_30-ufc.sh &
./run_roger_45-ufc.sh &
./run_roger_50-ufc.sh &
./run_roger_80-ufc.sh &
./run_roger_crop-specific.sh &

python merge_output.py
python write_simulations_to_csv.py