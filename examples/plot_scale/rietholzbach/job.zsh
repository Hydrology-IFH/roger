#!/bin/zsh
# run the RoGeR modeling experiment at Rietholzbach lysimeter
# run monte carlo
OMP_NUM_THREADS=1 mpirun -n 4 python ${PWD}/svat_monte_carlo/svat.py
# run sobol' sequence
OMP_NUM_THREADS=1 mpirun -n 4 python ${PWD}/svat_sensitivity/svat.py
# run monte carlo for transport model
OMP_NUM_THREADS=1 mpirun -n 4 python ${PWD}/svat_transport_monte_carlo/svat_transport.py
# run sobol' sequence for transport model
OMP_NUM_THREADS=1 mpirun -n 4 python ${PWD}/svat_transport_sensitivity/svat_transport.py
# run reverse sensitivity analysis for transport model
OMP_NUM_THREADS=1 mpirun -n 4 python ${PWD}/svat_transport_sensitivity_reverse/svat_transport.py
# run reverse monte carlo for transport model
OMP_NUM_THREADS=1 mpirun -n 4 python ${PWD}/svat_transport_monte_carlo_reverse/svat_transport.py
# run bromide benchmark transport model
python ${PWD}/svat_transport_bromide_benchmark/svat_transport.py
