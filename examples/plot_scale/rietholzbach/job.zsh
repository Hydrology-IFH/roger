#!/bin/zsh
# run monte carlo
cd "$PWD"/svat_monte_carlo
OMP_NUM_THREADS=1 mpirun -n 4 python svat.py
# run sobol' sequence
cd "$PWD"/svat_sensitivity
OMP_NUM_THREADS=1 mpirun -n 4 python svat.py
cd "$PWD"/svat_monte_carlo
python post_processing.py
cd "$PWD"/svat_sensitivity
python post_processing.py
# run monte carlo for transport model
cd "$PWD"/svat_transport_monte_carlo
OMP_NUM_THREADS=1 mpirun -n 4 python svat_transport.py
python post_processing.py
# run sobol' sequence for transport model
cd "$PWD"/svat_transport_sensitivity
OMP_NUM_THREADS=1 mpirun -n 4 python svat_transport.py
python post_processing.py
# run bromide benchmark transport model
cd "$PWD"/svat_transport_bromide_benchmark
python svat_transport.py
python post_processing.py
# run reverse sensitivity analysis for transport model
cd "$PWD"/svat_transport_sensitivity_reverse
OMP_NUM_THREADS=1 mpirun -n 4 python svat_transport.py
python post_processing.py
# run reverse monte carlo for transport model
cd "$PWD"/svat_transport_monte_carlo_reverse
OMP_NUM_THREADS=1 mpirun -n 4 python svat_transport.py
python post_processing.py
