#!/bin/bash

# activate conda environment roger if not already activated
# conda activate roger-mpi

mpirun --bind-to core --map-by core -report-bindings python oneD_event.py -b numpy -d cpu -n 4 4