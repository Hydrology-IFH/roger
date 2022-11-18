#!/bin/bash

# fixed problem size and varying number of CPU cores
python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 4 --pmem 32000 --only oneD_benchmark.py --debug
python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 8 --pmem 16000 --only oneD_benchmark.py --debug
python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 16 --pmem 8000 --only oneD_benchmark.py --debug
python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 28 --pmem 4000 --only oneD_benchmark.py --debug
python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 56 --pmem 4000 --only oneD_benchmark.py --debug
python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 112 --pmem 4000 --only oneD_benchmark.py --debug
python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 224 --pmem 4000 --only oneD_benchmark.py --debug
