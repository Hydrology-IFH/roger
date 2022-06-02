#!/bin/zsh
# run the RoGeR modeling experiment at Rietholzbach lysimeter
export PWDR=$PWD
# run monte carlo
cd ${PWDR}/svat_monte_carlo
OMP_NUM_THREADS=1 mpirun -n 4 python svat.py
# cd ${PWDR}
# # run sobol' sequence
# cd ${PWDR}/svat_sensitivity
# OMP_NUM_THREADS=1 mpirun -n 4 python svat.py
# cd ${PWDR}
# # run monte carlo for transport model
# cd ${PWDR}/svat_transport_monte_carlo
# OMP_NUM_THREADS=1 mpirun -n 4 python svat_transport.py
# cd ${PWDR}
# # run sobol' sequence for transport model
# cd ${PWDR}/svat_transport_sensitivity
# OMP_NUM_THREADS=1 mpirun -n 4 python svat_transport.py
# cd ${PWDR}
# # run reverse sensitivity analysis for transport model
# cd ${PWDR}/svat_transport_sensitivity_reverse
# OMP_NUM_THREADS=1 mpirun -n 4 python svat_transport.py
# cd ${PWDR}
# # run reverse monte carlo for transport model
# cd ${PWDR}/svat_transport_monte_carlo_reverse
# OMP_NUM_THREADS=1 mpirun -n 4 python svat_transport.py
# cd ${PWDR}
# # run bromide benchmark transport model
# cd ${PWDR}/svat_transport_bromide_benchmark
# python svat_transport.py
# cd ${PWDR}
