#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_oxygen18_partial
FILES="$PWD/oxygen18_deterministic_svat_*_partial_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_RK4_svat_*_partial_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done
