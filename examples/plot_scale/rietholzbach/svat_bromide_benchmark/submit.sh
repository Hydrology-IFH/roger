#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_transport_bromide_benchmark
FILES="$PWD/bromide_deterministic_svat_*_*.sh"
for f in $FILES
do
  qsub -q short $f
done
