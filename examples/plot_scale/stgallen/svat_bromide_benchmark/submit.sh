#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_transport_bromide_benchmark
FILES="$PWD/bromide_deterministic_svat_cm_*.sh"
for f in $FILES
do
  qsub -q short $f
done

FILES="$PWD/bromide_deterministic_svat_pi_*.sh"
for f in $FILES
do
  qsub -q short $f
done

FILES="$PWD/bromide_deterministic_svat_adp_*.sh"
for f in $FILES
do
  qsub -q short $f
done

FILES="$PWD/bromide_deterministic_svat_adpt_*.sh"
for f in $FILES
do
  qsub -q short $f
done
