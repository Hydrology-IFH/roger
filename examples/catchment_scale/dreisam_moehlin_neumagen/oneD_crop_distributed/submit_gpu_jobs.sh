#!/bin/bash

sbatch -p gpu oneD_crop_base_no-irrigation_soil-compaction_no-yellow-mustard.sh
sbatch -p gpu oneD_crop_base_no-irrigation_no-soil-compaction_no-yellow-mustard.sh
sbatch -p gpu oneD_crop_spring-drought_magnitude2_duration3_no-irrigation_soil-compaction_no-yellow-mustard.sh
sbatch -p gpu oneD_crop_spring-drought_magnitude2_duration3_no-irrigation_no-soil-compaction_no-yellow-mustard.sh
