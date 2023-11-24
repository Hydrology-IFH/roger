#!/bin/bash

# fixed problem size and varying number of CPU cores
python run_benchmarks.py --sizes 1000. --backends numpy-mpi --backends jax-mpi --nproc 2 --pmem 12000 --only SVATOXYGEN18_benchmark.py --debug --local --outfile $PWD/var_proc/svat_oxygen18/macbookpro14_2023/benchmark2_21112023.json
python run_benchmarks.py --sizes 1000. --backends numpy-mpi --backends jax-mpi --nproc 4 --pmem 6000 --only SVATOXYGEN18_benchmark.py --debug --local --outfile $PWD/var_proc/svat_oxygen18/macbookpro14_2023/benchmark4_21112023.json
python run_benchmarks.py --sizes 1000. --backends numpy-mpi --backends jax-mpi --nproc 6 --pmem 4000 --only SVATOXYGEN18_benchmark.py --debug --local --outfile $PWD/var_proc/svat_oxygen18/macbookpro14_2023/benchmark6_21112023.json
python run_benchmarks.py --sizes 1000. --backends numpy-mpi --backends jax-mpi --nproc 8 --pmem 3000 --only SVATOXYGEN18_benchmark.py --debug --local --outfile $PWD/var_proc/svat_oxygen18/macbookpro14_2023/benchmark8_21112023.json
