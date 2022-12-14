#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_oxygen18_sensitivity
FILES="$PWD/oxygen18_deterministic_svat_cm_sa_*_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_pi_sa_*_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_adp_sa_*_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_adpt_sa_*_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done
