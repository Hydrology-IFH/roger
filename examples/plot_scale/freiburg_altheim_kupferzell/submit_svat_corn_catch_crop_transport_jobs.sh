#!/bin/bash

cd ~/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat_transport

FILES="$PWD/svat_transport_*_corn_catch_crop_*.sh"
for f in $FILES
do
  qsub -q gpu $f
done

