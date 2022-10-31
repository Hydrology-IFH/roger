#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_transport_sensitivity
FILES="$PWD/oxygen18_determinsistic_*_sa.sh"
for f in $FILES
do
  qsub -q short $f
done
