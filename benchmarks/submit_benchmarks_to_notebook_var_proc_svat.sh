#!/bin/bash

# fixed problem size and varying number of CPU cores
python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 2 --only svat_benchmark.py --debug --local --outfile $PWD/var_proc/svat/macbookpro14_2023/benchmark2_21112023.json
python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 4 --only svat_benchmark.py --debug --local --outfile $PWD/var_proc/svat/macbookpro14_2023/benchmark4_21112023.json
python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 6 --only svat_benchmark.py --debug --local --outfile $PWD/var_proc/svat/macbookpro14_2023/benchmark6_21112023.json
python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 8 --only svat_benchmark.py --debug --local --outfile $PWD/var_proc/svat/macbookpro14_2023/benchmark8_21112023.json
