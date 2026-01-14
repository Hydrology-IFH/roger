#!/bin/bash

sbatch -p gpu oneD_crop_base_soil-compaction.sh
sbatch -p gpu oneD_crop_base.sh
