#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_transport_monte_carlo
FILES="$PWD/oxygen18_determinsistic_svat_ad_mc_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_determinsistic_svat_adt_mc_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done
