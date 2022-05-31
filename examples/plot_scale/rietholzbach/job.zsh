#!/bin/zsh
# run the RoGeR modeling experiment at Rietholzbach lysimeter
# run monte carlo
cd "$PWD"/svat_monte_carlo
OMP_NUM_THREADS=1 mpirun -n 4 python svat.py
cd ..
# run sobol' sequence
cd "$PWD"/svat_sensitivity
OMP_NUM_THREADS=1 mpirun -n 4 python svat.py
cd ..
# run monte carlo for transport model
cd "$PWD"/svat_transport_monte_carlo
OMP_NUM_THREADS=1 mpirun -n 4 python svat_transport.py
cd ..
# run sobol' sequence for transport model
cd "$PWD"/svat_transport_sensitivity
OMP_NUM_THREADS=1 mpirun -n 4 python svat_transport.py
cd ..
# run bromide benchmark transport model
cd "$PWD"/svat_transport_bromide_benchmark
python svat_transport.py
cd ..
# run reverse sensitivity analysis for transport model
cd "$PWD"/svat_transport_sensitivity_reverse
OMP_NUM_THREADS=1 mpirun -n 4 python svat_transport.py
cd ..
# run reverse monte carlo for transport model
cd "$PWD"/svat_transport_monte_carlo_reverse
OMP_NUM_THREADS=1 mpirun -n 4 python svat_transport.py
exit
