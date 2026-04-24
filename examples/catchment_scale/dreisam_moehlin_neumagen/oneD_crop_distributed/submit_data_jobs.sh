#!/bin/bash

sbatch -p compute write_roger_data_base_soil-compaction.sh
sbatch -p compute write_roger_data_base_soil-compaction_irrigation.sh
sbatch -p compute write_roger_data_summer-drought_magnitude0_duration3_soil-compaction.sh
sbatch -p compute write_roger_data_summer-drought_magnitude0_duration3_soil-compaction_irrigation.sh
sbatch -p compute write_roger_data_long-term_magnitude2_duration0_soil-compaction.sh
sbatch -p compute write_roger_data_long-term_magnitude2_duration0_soil-compaction_irrigation.sh
sbatch -p compute write_roger_data_base_soil-compaction_grain-corn-only.sh