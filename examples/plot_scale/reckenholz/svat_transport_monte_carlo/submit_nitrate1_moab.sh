#!/bin/bash

cd ~/roger/examples/plot_scale/reckenholz/svat_transport_monte_carlo
FILES="$PWD/nitrate1_*_mc_moab.sh"
for f in $FILES
do
  qsub -q short ./$f
done
