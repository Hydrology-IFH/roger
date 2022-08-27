#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_transport_sensitivity
FILES="$PWD/oxygen18_*_sa_moab.sh"
for f in $FILES
do
  qsub -q short $f
  sleep 300
done
