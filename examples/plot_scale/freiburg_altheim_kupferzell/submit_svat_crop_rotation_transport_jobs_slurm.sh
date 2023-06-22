#!/bin/bash

cd ~/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat_transport

FILES="$PWD/svat_transport_*_crop_rotation_*_cpumpi_slurm.sh"
for f in $FILES
do
  sbatch --partition=single $f
done

