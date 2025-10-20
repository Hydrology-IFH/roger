#!/bin/bash

cd irrigation
python write_job_scripts_slurm.py
cd ..
cd no-irrigation
python write_job_scripts_slurm.py
cd ..
cd irrigation_soil-compaction
python write_job_scripts_slurm.py
cd ..
cd no-irrigation_soil-compaction
python write_job_scripts_slurm.py
cd ..