#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate roger

python no-irrigation/write_simulations_to_csv.py
python irrigation/write_simulations_to_csv.py 
python no-irrigation_soil-compaction/write_simulations_to_csv.py 
python irrigation_soil-compaction/write_simulations_to_csv.py 