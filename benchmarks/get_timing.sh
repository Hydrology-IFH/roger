#!/bin/sh

INFILES=$(find $PWD/var_size/oneD -type f -maxdepth 1 -name "timing_files_*.json")
python get_timing.py --benchmark-type var_size -- $INFILES

INFILES=$(find $PWD/var_proc/oneD -type f -maxdepth 1 -name "timing_files_*.json")
python get_timing.py --benchmark-type var_proc -- $INFILES
