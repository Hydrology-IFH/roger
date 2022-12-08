#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_oxygen18_monte_carlo

qsub -q short post_processing_cm.sh
qsub -q short post_processing_pi.sh
qsub -q short post_processing_ad.sh
qsub -q short post_processing_adt.sh
qsub -q short post_processing_pf.sh
qsub -q short post_processing_op.sh
qsub -q short post_processing_pow.sh
qsub -q short post_processing_powt.sh
qsub -q short post_processing_pfad.sh
qsub -q short post_processing_pfadt.sh
qsub -q short post_processing_tvt.sh
qsub -q short post_processing_tv.sh
