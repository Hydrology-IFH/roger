#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_transport
FILES="$PWD/oxygen18_*_moab.sh"
for f in $FILES
do
  qsub -q short $f
  sleep 300
done
