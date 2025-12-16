#!/bin/bash

cd ~/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat

FILES="$PWD/svat_*_grass_*_moab.sh"
for f in $FILES
do
  qsub -q short $f
done

