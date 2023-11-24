#!/bin/bash

# fixed problem size and varying number of CPU cores
python run_benchmarks.py --sizes 24000. --backends numpy-mpi --backends jax-mpi --nproc 4 --pmem 45000 --only SVATOXYGEN18_benchmark.py --timesteps 20 --debug
python run_benchmarks.py --sizes 24000. --backends numpy-mpi --backends jax-mpi --nproc 10 --pmem 18000 --only SVATOXYGEN18_benchmark.py --timesteps 20 --debug
python run_benchmarks.py --sizes 24000. --backends numpy-mpi --backends jax-mpi --nproc 20 --pmem 9000 --only SVATOXYGEN18_benchmark.py --timesteps 20 --debug
python run_benchmarks.py --sizes 24000. --backends numpy-mpi --backends jax-mpi --nproc 40 --pmem 4500 --only SVATOXYGEN18_benchmark.py --timesteps 20 --debug
python run_benchmarks.py --sizes 24000. --backends numpy-mpi --backends jax-mpi --nproc 80 --pmem 4500 --only SVATOXYGEN18_benchmark.py --timesteps 20 --debug
python run_benchmarks.py --sizes 24000. --backends numpy-mpi --backends jax-mpi --nproc 120 --pmem 4500 --only SVATOXYGEN18_benchmark.py --timesteps 20 --debug
