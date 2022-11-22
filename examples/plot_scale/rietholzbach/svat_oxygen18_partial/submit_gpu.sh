#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_oxygen18_partial
FILES="$PWD/oxygen18_deterministic_svat_ad_partial_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_adt_partial_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_pf_partial_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_pfad_partial_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_op_partial_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_pow_partial_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_tvt_partial_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_tv_partial_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done
