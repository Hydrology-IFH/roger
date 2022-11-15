#!/bin/sh

INFILES=$(find $PWD/var_size/oneD -type f -maxdepth 1 -name "timing_files_*.json")
python get_timing.py --benchmark-type var_size --infolder oneD -- $INFILES

INFILES=$(find $PWD/var_proc/oneD -type f -maxdepth 1 -name "timing_files_*.json")
python get_timing.py --benchmark-type var_proc --infolder oneD -- $INFILES

INFILES=$(find $PWD/var_size/svat -type f -maxdepth 1 -name "timing_files_*.json")
python get_timing.py --benchmark-type var_size --infolder svat -- $INFILES

INFILES=$(find $PWD/var_proc/svat -type f -maxdepth 1 -name "timing_files_*.json")
python get_timing.py --benchmark-type var_proc --infolder svat -- $INFILES
