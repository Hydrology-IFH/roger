#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate roger

cd nitrate
python write_job_scripts.py
./run_roger-sas-nitrate.sh