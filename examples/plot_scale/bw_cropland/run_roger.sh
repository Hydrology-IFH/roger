#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate roger

python /no-irrigation/write_job_scripts.py
python /no-irrigation_soil-compaction/write_job_scripts.py
python /irrigation/write_job_scripts.py
python /irrigation_soil-compaction/write_job_scripts.py
./no_irrigation/run_roger.sh
./no_irrigation_soil-compaction/run_roger.sh
./irrigation/run_roger.sh
./irrigation_soil-compaction/run_roger.sh
