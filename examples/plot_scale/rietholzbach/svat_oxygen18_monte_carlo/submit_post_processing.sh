#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_oxygen18_monte_carlo

qsub -q short post_processing_cm.sh
qsub -q short post_processing_pi.sh
qsub -q short post_processing_adp.sh
qsub -q short post_processing_adpt.sh
qsub -q short post_processing_pfp.sh
qsub -q short post_processing_opp.sh
qsub -q short post_processing_adk.sh
qsub -q short post_processing_adkt.sh
