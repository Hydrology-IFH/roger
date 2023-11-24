#!/bin/sh

INFILES=$(find $PWD/var_size/svat_oxygen18/$HABENCH -type f -maxdepth 1 -name "timing_files_*.json")
python get_timing.py --benchmark-type var_size --infolder $PWD/var_size/svat_oxygen18/$HABENCH -- $INFILES

INFILES=$(find $PWD/var_proc/svat_oxygen18/$HABENCH -type f -maxdepth 1 -name "timing_files_*.json")
python get_timing.py --benchmark-type var_proc --infolder $PWD/var_proc/svat_oxygen18/$HABENCH -- $INFILES
