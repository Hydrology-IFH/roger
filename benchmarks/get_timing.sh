#!/bin/sh

# INFILES=$(find $PWD/var_size/oneD/cluster -type f -maxdepth 1 -name "timing_files_*.json")
# python get_timing.py --benchmark-type var_size --infolder oneD -- $INFILES
#
# INFILES=$(find $PWD/var_proc/oneD/cluster -type f -maxdepth 1 -name "timing_files_*.json")
# python get_timing.py --benchmark-type var_proc --infolder oneD -- $INFILES

INFILES=$(find $PWD/var_size/svat/cluster -type f -maxdepth 1 -name "timing_files_*.json")
python get_timing.py --benchmark-type var_size --infolder $PWD/var_size/svat/cluster -- $INFILES

INFILES=$(find $PWD/var_proc/svat/cluster -type f -maxdepth 1 -name "timing_files_*.json")
python get_timing.py --benchmark-type var_proc --infolder $PWD/var_proc/svat/cluster -- $INFILES

INFILES=$(find $PWD/var_size/svat_oxygen18/cluster -type f -maxdepth 1 -name "timing_files_*.json")
python get_timing.py --benchmark-type var_size --infolder $PWD/var_size/svat_oxygen18/cluster -- $INFILES

INFILES=$(find $PWD/var_proc/svat_oxygen18/cluster -type f -maxdepth 1 -name "timing_files_*.json")
python get_timing.py --benchmark-type var_proc --infolder $PWD/var_proc/svat_oxygen18/cluster -- $INFILES
