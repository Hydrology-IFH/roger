#!/bin/bash

python svat_transport.py -tms complete-mixing
python svat_transport.py -tms piston
python svat_transport.py -tms advection-dispersion
python svat_transport.py -tms time-variant_advection-dispersion
python svat_transport.py -tms time-variant
