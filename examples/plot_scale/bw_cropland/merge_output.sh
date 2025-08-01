#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate roger
python no-irrigation/merge_output.py
python irrigation/merge_output.py 
python no-irrigation_soil-compaction/merge_output.py 
python irrigation_soil-compaction/merge_output.py 