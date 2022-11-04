#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_transport
FILES="$PWD/oxygen18_deterministic_*_moab.sh"
for f in $FILES
do
  qsub -q long $f
done

cd ~/roger/examples/plot_scale/rietholzbach/svat_transport
FILES="$PWD/oxygen18_RK4_*_moab.sh"
for f in $FILES
do
  qsub -q long $f
done
