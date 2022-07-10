#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate roger
python oneD_event.py
