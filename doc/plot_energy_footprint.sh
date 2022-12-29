#!/bin/sh

python plot_energy_footprint.py --xaxis size --file $PWD/../benchmarks/var_size/svat/cluster/carbon_footprint.csv --unit Wh --name svat_iteration --nitt 97
python plot_energy_footprint.py --xaxis size --file $PWD/../benchmarks/var_size/svat/cluster/carbon_footprint.csv --name svat_year --nitt 97 --rescale 10000
python plot_energy_footprint.py --xaxis size --file $PWD/../benchmarks/var_size/svat_oxygen18/cluster/carbon_footprint.csv --unit Wh --name svat_oxygen18_iteration --nitt 20
python plot_energy_footprint.py --xaxis size --file $PWD/../benchmarks/var_size/svat_oxygen18/cluster/carbon_footprint.csv --name svat_oxygen18_year --nitt 20 --rescale 365
