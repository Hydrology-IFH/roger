#!/bin/bash

cd ~/roger/examples/plot_scale/reckenholz/svat_sensitivity
FILES="$PWD/*_sa_moab.sh"
for f in $FILES
do
  qsub -q short $f
done
