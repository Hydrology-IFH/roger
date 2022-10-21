#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_transport_monte_carlo
FILES="$PWD/oxygen18_determinsistic_*_mc.sh"
for f in $FILES
do
  qsub -q short $f
done

# cd ~/roger/examples/plot_scale/rietholzbach/svat_transport_monte_carlo
# FILES="$PWD/oxygen18_Euler_*_mc.sh"
# for f in $FILES
# do
#   qsub -q short $f
# done
#
# cd ~/roger/examples/plot_scale/rietholzbach/svat_transport_monte_carlo
# FILES="$PWD/oxygen18_RK4_*_mc.sh"
# for f in $FILES
# do
#   qsub -q short $f
# done
