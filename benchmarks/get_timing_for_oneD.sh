#!/bin/sh

INFILES=$(find $PWD/var_size/oneD/$HABENCH -type f -maxdepth 1 -name "timing_files_*.json")
python get_timing.py --benchmark-type var_size --infolder $PWD/var_size/oneD/$HABENCH -- $INFILES

INFILES=$(find $PWD/var_proc/oneD/$HABENCH -type f -maxdepth 1 -name "timing_files_*.json")
python get_timing.py --benchmark-type var_proc --infolder $PWD/var_proc/oneD/$HABENCH -- $INFILES
