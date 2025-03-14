#!/bin/bash
python /no_irrigation/write_job_scripts.py
python /irrigation/write_job_scripts.py
./no_irrigation/run_roger.sh
./irrigation/run_roger.sh
