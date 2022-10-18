#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_transport_monte_carlo
for i in {1..25}
do
  qsub -q gpu "$PWD/oxygen18_determinsistic_svat_ad_mc_gpu.sh"
done

for i in {1..25}
do
  qsub -q gpu "$PWD/oxygen18_determinsistic_svat_adt_mc_gpu.sh"
done
