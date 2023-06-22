#!/bin/bash

cd ~/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat_transport

FILES="$PWD/svat_transport_*_crop_rotation_*_cpu_slurm.sh"
for f in $FILES
do
  qsub -q long $f
done

