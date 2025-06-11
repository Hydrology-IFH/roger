#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate roger

python svat.py -tms complete-mixing
python svat.py -tms piston
python svat.py -tms advection-dispersion-power
python svat.py -tms time-variant_advection-dispersion-power