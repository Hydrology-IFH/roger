#!/bin/bash

sbatch -p gpu oneD_crop_base_soil-compaction.sh
sbatch -p gpu oneD_crop_base_soil-compaction_irrigation.sh
sbatch -p gpu oneD_crop_summer-drought_magnitude0_duration3_soil-compaction.sh
sbatch -p gpu oneD_crop_summer-drought_magnitude0_duration3_soil-compaction_irrigation.sh
sbatch -p gpu oneD_crop_summer-drought_magnitude2_duration0_soil-compaction.sh
sbatch -p gpu oneD_crop_summer-drought_magnitude2_duration0_soil-compaction_irrigation.sh
sbatch -p gpu oneD_crop_summer-drought_magnitude2_duration3_soil-compaction.sh
sbatch -p gpu oneD_crop_summer-drought_magnitude2_duration3_soil-compaction_irrigation.sh
sbatch -p gpu oneD_crop_spring-drought_magnitude0_duration3_soil-compaction.sh
sbatch -p gpu oneD_crop_spring-drought_magnitude0_duration3_soil-compaction_irrigation.sh
sbatch -p gpu oneD_crop_spring-drought_magnitude2_duration0_soil-compaction.sh
sbatch -p gpu oneD_crop_spring-drought_magnitude2_duration0_soil-compaction_irrigation.sh
sbatch -p gpu oneD_crop_spring-drought_magnitude2_duration3_soil-compaction.sh
sbatch -p gpu oneD_crop_spring-drought_magnitude2_duration3_soil-compaction_irrigation.sh
sbatch -p gpu oneD_crop_spring-summer-drought_magnitude0_duration3_soil-compaction.sh
sbatch -p gpu oneD_crop_spring-summer-drought_magnitude0_duration3_soil-compaction_irrigation.sh
sbatch -p gpu oneD_crop_spring-summer-drought_magnitude2_duration3_soil-compaction.sh
sbatch -p gpu oneD_crop_spring-summer-drought_magnitude2_duration3_soil-compaction_irrigation.sh