#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_transport_sensitvity
FILES="$PWD/oxygen18_deterministic_svat_ad_sa_*_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_adt_sa_*_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done
