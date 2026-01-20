#!/bin/bash

sbatch -p gpu oneD_crop_base_soil-compaction.sh
sbatch -p gpu oneD_crop_base.sh
sbatch -p gpu oneD_crop_spring-summer-drought_magnitude2_duration3_soil-compaction.sh
sbatch -p gpu oneD_crop_spring-summer-drought_magnitude2_duration3.sh
