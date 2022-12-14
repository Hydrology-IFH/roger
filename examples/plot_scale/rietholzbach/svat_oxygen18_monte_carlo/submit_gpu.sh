#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_oxygen18_monte_carlo

qsub -q gpu oxygen18_deterministic_svat_cm_mc_gpu.sh

qsub -q gpu oxygen18_deterministic_svat_pi_mc_gpu.sh

FILES="$PWD/oxygen18_deterministic_svat_adp_mc_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_adpt_mc_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_pfp_mc_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_opp_mc_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_adk_mc_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done

FILES="$PWD/oxygen18_deterministic_svat_adkt_mc_*_gpu.sh"
for f in $FILES
do
  qsub -q gpu $f
done
