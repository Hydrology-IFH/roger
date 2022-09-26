#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_transport_monte_carlo
FILES="$PWD/oxygen18_Euler_*_moab.sh"
for f in $FILES
do
  qsub -q short $f
done

cd ~/roger/examples/plot_scale/rietholzbach/svat_transport_monte_carlo
FILES="$PWD/oxygen18_RK4_*_moab.sh"
for f in $FILES
do
  qsub -q short $f
done
