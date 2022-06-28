#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_transport_sensitivity
FILES="$PWD/oxygen18_*.sh"
for f in $FILES
do
  sbatch ./$f
done
