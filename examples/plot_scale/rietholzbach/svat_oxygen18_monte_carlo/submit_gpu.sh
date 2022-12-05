#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_oxygen18_monte_carlo

qsub -q gpu oxygen18_deterministic_svat_cm_mc_gpu.sh

qsub -q gpu oxygen18_deterministic_svat_pi_mc_gpu.sh

FILES="$PWD/oxygen18_deterministic_svat_ad_mc_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_adt_mc_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_pf_mc_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_pfad_mc_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_pfadt_mc_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_op_mc_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_pow_mc_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_powt_mc_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_tvt_mc_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_tv_mc_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done
