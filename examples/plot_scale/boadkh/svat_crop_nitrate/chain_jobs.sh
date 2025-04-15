#!/bin/bash
# use the following to run in background: nohup ./chain_jobs.sh &
# sbatch --partition cpu_il svat_crop_nitrate_bretten_bare-grass_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_bare-grass_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_bare-grass_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_grain-corn_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_grain-corn_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_grain-corn_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_grain-corn_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_grain-corn_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_grain-corn_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_grain-corn_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_grain-corn_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_grain-corn_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_grain-corn_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_grain-corn_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_grain-corn_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_grain-corn_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_grain-corn_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_grain-corn_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_miscanthus_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_miscanthus_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_miscanthus_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_sugar-beet_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_sugar-beet_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_sugar-beet_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_sugar-beet_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_sugar-beet_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_sugar-beet_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_clover_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_clover_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_clover_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_winter-wheat_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_winter-wheat_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_summer-wheat_winter-wheat_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_clover_high_Nfert_slurm.sh
# sleep 14h
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_clover_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_clover_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_clover_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_soybean_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_soybean_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_soybean_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_sugar-beet_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_sugar-beet_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_sugar-beet_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_sugar-beet_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_sugar-beet_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_sugar-beet_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bretten_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_bare-grass_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_bare-grass_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_bare-grass_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_grain-corn_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_grain-corn_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_grain-corn_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_grain-corn_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_grain-corn_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_grain-corn_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_grain-corn_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_grain-corn_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_grain-corn_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_grain-corn_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_grain-corn_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_grain-corn_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_grain-corn_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_grain-corn_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_grain-corn_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_miscanthus_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_miscanthus_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_miscanthus_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_sugar-beet_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_sugar-beet_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_sugar-beet_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_sugar-beet_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_sugar-beet_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sleep 14h
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_sugar-beet_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_clover_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_clover_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_clover_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_winter-wheat_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_winter-wheat_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_summer-wheat_winter-wheat_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_clover_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_clover_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_clover_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_soybean_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_soybean_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_soybean_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_sugar-beet_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_sugar-beet_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_sugar-beet_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_sugar-beet_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_sugar-beet_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_sugar-beet_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_bruchsal-heidelsheim_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_bare-grass_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_bare-grass_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_bare-grass_medium_Nfert_slurm.sh
# sleep 14h
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_grain-corn_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_grain-corn_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_grain-corn_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_grain-corn_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_grain-corn_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_grain-corn_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_grain-corn_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_grain-corn_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_grain-corn_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_grain-corn_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_grain-corn_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_grain-corn_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_grain-corn_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_grain-corn_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_grain-corn_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_miscanthus_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_miscanthus_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_miscanthus_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_sugar-beet_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_sugar-beet_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_sugar-beet_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_sugar-beet_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_sugar-beet_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_sugar-beet_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_clover_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_clover_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_clover_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_winter-wheat_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_winter-wheat_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_summer-wheat_winter-wheat_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_clover_silage-corn_high_Nfert_slurm.sh
# sleep 14h
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_clover_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_clover_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_soybean_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_soybean_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_soybean_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_sugar-beet_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_sugar-beet_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_sugar-beet_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_sugar-beet_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_sugar-beet_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_sugar-beet_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_ehingen-kirchen_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_bare-grass_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_bare-grass_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_bare-grass_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_grain-corn_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_grain-corn_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_grain-corn_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_grain-corn_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_grain-corn_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_grain-corn_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_grain-corn_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_grain-corn_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_grain-corn_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_grain-corn_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_grain-corn_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_grain-corn_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_grain-corn_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_grain-corn_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_grain-corn_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_miscanthus_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_miscanthus_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_miscanthus_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_sugar-beet_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_sugar-beet_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_sugar-beet_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_sugar-beet_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_sugar-beet_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_sugar-beet_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_clover_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_clover_winter-wheat_low_Nfert_slurm.sh
# sleep 14h
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_clover_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_winter-wheat_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_winter-wheat_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_summer-wheat_winter-wheat_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_clover_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_clover_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_clover_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_soybean_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_soybean_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_soybean_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_sugar-beet_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_sugar-beet_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_sugar-beet_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_sugar-beet_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_sugar-beet_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_sugar-beet_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_eppingen-elsenz_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_bare-grass_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_bare-grass_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_bare-grass_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_grain-corn_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_grain-corn_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_grain-corn_winter-wheat_clover_medium_Nfert_slurm.sh
# sleep 14h
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_grain-corn_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_grain-corn_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_grain-corn_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_grain-corn_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_grain-corn_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_grain-corn_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_grain-corn_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_grain-corn_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_grain-corn_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_grain-corn_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_grain-corn_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_grain-corn_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_miscanthus_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_miscanthus_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_miscanthus_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_sugar-beet_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_sugar-beet_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_sugar-beet_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_sugar-beet_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_sugar-beet_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_sugar-beet_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_clover_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_clover_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_clover_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_summer-wheat_winter-wheat_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_clover_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_clover_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_clover_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sleep 14h
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_soybean_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_soybean_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_soybean_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_sugar-beet_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_sugar-beet_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_sugar-beet_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_sugar-beet_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_sugar-beet_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_sugar-beet_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_freiburg_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_bare-grass_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_bare-grass_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_bare-grass_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_grain-corn_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_grain-corn_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_grain-corn_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_grain-corn_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_grain-corn_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_grain-corn_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_grain-corn_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_grain-corn_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_grain-corn_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_grain-corn_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_grain-corn_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_grain-corn_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_grain-corn_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_grain-corn_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_grain-corn_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_miscanthus_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_miscanthus_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_miscanthus_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_sugar-beet_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_sugar-beet_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_sugar-beet_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_sugar-beet_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_sugar-beet_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_sugar-beet_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_clover_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_clover_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_clover_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_winter-wheat_low_Nfert_slurm.sh
# sleep 14h
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_winter-wheat_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_winter-wheat_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_summer-wheat_winter-wheat_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_clover_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_clover_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_clover_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_soybean_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_soybean_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_soybean_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_sugar-beet_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_sugar-beet_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_sugar-beet_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_sugar-beet_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_sugar-beet_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_sugar-beet_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_gottmadingen_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_bare-grass_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_bare-grass_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_bare-grass_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_grain-corn_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_grain-corn_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_grain-corn_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_grain-corn_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_grain-corn_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_grain-corn_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sleep 14h
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_grain-corn_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_grain-corn_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_grain-corn_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_grain-corn_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_grain-corn_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_grain-corn_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_grain-corn_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_grain-corn_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_grain-corn_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_miscanthus_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_miscanthus_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_miscanthus_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_sugar-beet_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_sugar-beet_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_sugar-beet_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_sugar-beet_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_sugar-beet_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_sugar-beet_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_clover_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_clover_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_clover_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_winter-wheat_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_winter-wheat_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_summer-wheat_winter-wheat_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_clover_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_clover_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_clover_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sleep 14h
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_soybean_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_soybean_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_soybean_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_sugar-beet_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_sugar-beet_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_sugar-beet_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_sugar-beet_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_sugar-beet_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_sugar-beet_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_hayingen_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_bare-grass_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_bare-grass_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_bare-grass_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_grain-corn_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_grain-corn_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_grain-corn_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_grain-corn_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_grain-corn_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_grain-corn_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_grain-corn_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_grain-corn_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_grain-corn_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_grain-corn_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_grain-corn_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_grain-corn_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_grain-corn_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_grain-corn_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_grain-corn_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_miscanthus_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_miscanthus_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_miscanthus_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_sugar-beet_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_sugar-beet_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_sugar-beet_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_sugar-beet_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_sugar-beet_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_sugar-beet_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_clover_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_clover_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_clover_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sleep 14h
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_winter-wheat_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_winter-wheat_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_summer-wheat_winter-wheat_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_clover_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_clover_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_clover_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_soybean_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_soybean_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_soybean_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_sugar-beet_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_sugar-beet_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_sugar-beet_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_sugar-beet_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_sugar-beet_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_sugar-beet_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_kupferzell_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_bare-grass_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_bare-grass_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_bare-grass_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_grain-corn_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_grain-corn_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_grain-corn_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_grain-corn_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_grain-corn_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_grain-corn_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_grain-corn_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_grain-corn_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_grain-corn_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sleep 14h
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_grain-corn_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_grain-corn_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_grain-corn_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_grain-corn_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_grain-corn_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_grain-corn_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_miscanthus_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_miscanthus_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_miscanthus_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_sugar-beet_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_sugar-beet_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_sugar-beet_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_sugar-beet_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_sugar-beet_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_sugar-beet_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_clover_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_clover_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_clover_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_winter-wheat_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_winter-wheat_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_summer-wheat_winter-wheat_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_clover_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_clover_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_clover_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_soybean_winter-rape_high_Nfert_slurm.sh
# sleep 14h
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_soybean_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_soybean_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_sugar-beet_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_sugar-beet_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_sugar-beet_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_sugar-beet_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_sugar-beet_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_sugar-beet_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_merklingen_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_bare-grass_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_bare-grass_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_bare-grass_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_grain-corn_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_grain-corn_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_grain-corn_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_grain-corn_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_grain-corn_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_grain-corn_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_grain-corn_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_grain-corn_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_grain-corn_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_grain-corn_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_grain-corn_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_grain-corn_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_grain-corn_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_grain-corn_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_grain-corn_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_miscanthus_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_miscanthus_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_miscanthus_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_sugar-beet_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_sugar-beet_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_sugar-beet_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_sugar-beet_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_sugar-beet_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_sugar-beet_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_clover_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_clover_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_clover_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sleep 14h
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_winter-wheat_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_winter-wheat_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_summer-wheat_winter-wheat_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_clover_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_clover_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_clover_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_soybean_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_soybean_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_soybean_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_sugar-beet_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_sugar-beet_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_sugar-beet_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_sugar-beet_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_sugar-beet_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_sugar-beet_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_muellheim_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_bare-grass_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_bare-grass_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_bare-grass_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_grain-corn_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_grain-corn_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_grain-corn_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_grain-corn_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_grain-corn_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_grain-corn_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_grain-corn_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_grain-corn_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_grain-corn_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_grain-corn_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_grain-corn_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_grain-corn_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sleep 14h
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_grain-corn_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_grain-corn_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_grain-corn_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_miscanthus_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_miscanthus_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_miscanthus_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_sugar-beet_winter-wheat_winter-barley_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_sugar-beet_winter-wheat_winter-barley_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_sugar-beet_winter-wheat_winter-barley_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_sugar-beet_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_sugar-beet_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_sugar-beet_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_clover_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_clover_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_clover_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_winter-wheat_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_winter-wheat_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_winter-wheat_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_winter-wheat_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_winter-wheat_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_winter-wheat_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_winter-wheat_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_winter-wheat_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_summer-wheat_winter-wheat_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_clover_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_clover_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_clover_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_clover_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_clover_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_clover_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_silage-corn_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_silage-corn_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_silage-corn_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_soybean_winter-rape_high_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_soybean_winter-rape_low_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_soybean_winter-rape_medium_Nfert_slurm.sh
# sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_sugar-beet_silage-corn_high_Nfert_slurm.sh
# sleep 14h
sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_sugar-beet_silage-corn_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_sugar-beet_silage-corn_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_sugar-beet_silage-corn_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_sugar-beet_silage-corn_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_sugar-beet_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_winter-rape_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_winter-rape_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_oehringen_winter-wheat_winter-rape_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_bare-grass_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_bare-grass_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_bare-grass_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_grain-corn_winter-wheat_clover_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_grain-corn_winter-wheat_clover_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_grain-corn_winter-wheat_clover_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_grain-corn_winter-wheat_winter-barley_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_grain-corn_winter-wheat_winter-barley_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_grain-corn_winter-wheat_winter-barley_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_grain-corn_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_grain-corn_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_grain-corn_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_grain-corn_winter-wheat_winter-rape_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_grain-corn_winter-wheat_winter-rape_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_grain-corn_winter-wheat_winter-rape_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_grain-corn_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_grain-corn_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_grain-corn_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_miscanthus_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_miscanthus_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_miscanthus_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_sugar-beet_winter-wheat_winter-barley_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_sugar-beet_winter-wheat_winter-barley_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_sugar-beet_winter-wheat_winter-barley_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_sugar-beet_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_sugar-beet_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_sugar-beet_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_clover_winter-wheat_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_clover_winter-wheat_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_clover_winter-wheat_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_winter-wheat_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_winter-wheat_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_winter-wheat_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_winter-wheat_silage-corn_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_winter-wheat_silage-corn_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_winter-wheat_silage-corn_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_winter-wheat_winter-rape_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_winter-wheat_winter-rape_low_Nfert_slurm.sh
sleep 14h
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_winter-wheat_winter-rape_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_winter-wheat_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_winter-wheat_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_summer-wheat_winter-wheat_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_clover_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_clover_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_clover_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_clover_silage-corn_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_clover_silage-corn_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_clover_silage-corn_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_silage-corn_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_silage-corn_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_silage-corn_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_soybean_winter-rape_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_soybean_winter-rape_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_soybean_winter-rape_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_sugar-beet_silage-corn_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_sugar-beet_silage-corn_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_sugar-beet_silage-corn_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_sugar-beet_silage-corn_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_sugar-beet_silage-corn_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_sugar-beet_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_winter-rape_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_winter-rape_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_stockach_winter-wheat_winter-rape_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_bare-grass_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_bare-grass_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_bare-grass_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_grain-corn_winter-wheat_clover_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_grain-corn_winter-wheat_clover_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_grain-corn_winter-wheat_clover_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_grain-corn_winter-wheat_winter-barley_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_grain-corn_winter-wheat_winter-barley_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_grain-corn_winter-wheat_winter-barley_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_grain-corn_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_grain-corn_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_grain-corn_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_grain-corn_winter-wheat_winter-rape_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_grain-corn_winter-wheat_winter-rape_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_grain-corn_winter-wheat_winter-rape_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_grain-corn_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_grain-corn_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_grain-corn_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
sleep 14h
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_miscanthus_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_miscanthus_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_miscanthus_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_sugar-beet_winter-wheat_winter-barley_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_sugar-beet_winter-wheat_winter-barley_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_sugar-beet_winter-wheat_winter-barley_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_sugar-beet_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_sugar-beet_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_sugar-beet_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_clover_winter-wheat_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_clover_winter-wheat_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_clover_winter-wheat_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_winter-wheat_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_winter-wheat_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_winter-wheat_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_winter-wheat_silage-corn_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_winter-wheat_silage-corn_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_winter-wheat_silage-corn_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_winter-wheat_winter-rape_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_winter-wheat_winter-rape_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_winter-wheat_winter-rape_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_winter-wheat_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_winter-wheat_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_summer-wheat_winter-wheat_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_clover_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_clover_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_clover_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_clover_silage-corn_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_clover_silage-corn_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_clover_silage-corn_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_silage-corn_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_silage-corn_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_silage-corn_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_soybean_winter-rape_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_soybean_winter-rape_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_soybean_winter-rape_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_sugar-beet_silage-corn_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_sugar-beet_silage-corn_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_sugar-beet_silage-corn_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_sugar-beet_silage-corn_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_sugar-beet_silage-corn_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_sugar-beet_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_winter-rape_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_winter-rape_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_vellberg-kleinaltdorf_winter-wheat_winter-rape_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_bare-grass_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_bare-grass_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_bare-grass_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_grain-corn_winter-wheat_clover_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_grain-corn_winter-wheat_clover_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_grain-corn_winter-wheat_clover_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_grain-corn_winter-wheat_winter-barley_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_grain-corn_winter-wheat_winter-barley_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_grain-corn_winter-wheat_winter-barley_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_grain-corn_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_grain-corn_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_grain-corn_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_grain-corn_winter-wheat_winter-rape_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_grain-corn_winter-wheat_winter-rape_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_grain-corn_winter-wheat_winter-rape_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_grain-corn_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_grain-corn_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_grain-corn_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_miscanthus_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_miscanthus_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_miscanthus_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_sugar-beet_winter-wheat_winter-barley_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_sugar-beet_winter-wheat_winter-barley_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_sugar-beet_winter-wheat_winter-barley_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_sugar-beet_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_sugar-beet_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_sugar-beet_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_clover_winter-wheat_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_clover_winter-wheat_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_clover_winter-wheat_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_winter-wheat_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_winter-wheat_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_winter-wheat_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_winter-wheat_silage-corn_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_winter-wheat_silage-corn_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_winter-wheat_silage-corn_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_winter-wheat_winter-rape_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_winter-wheat_winter-rape_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_winter-wheat_winter-rape_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
sleep 14h
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_winter-wheat_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_winter-wheat_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_summer-wheat_winter-wheat_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_clover_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_clover_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_clover_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_clover_silage-corn_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_clover_silage-corn_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_clover_silage-corn_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_silage-corn_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_silage-corn_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_silage-corn_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_soybean_winter-rape_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_soybean_winter-rape_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_soybean_winter-rape_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_sugar-beet_silage-corn_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_sugar-beet_silage-corn_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_sugar-beet_silage-corn_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_sugar-beet_silage-corn_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_sugar-beet_silage-corn_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_sugar-beet_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_winter-rape_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_winter-rape_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_weingarten_winter-wheat_winter-rape_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_bare-grass_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_bare-grass_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_bare-grass_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_grain-corn_winter-wheat_clover_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_grain-corn_winter-wheat_clover_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_grain-corn_winter-wheat_clover_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_grain-corn_winter-wheat_winter-barley_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_grain-corn_winter-wheat_winter-barley_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_grain-corn_winter-wheat_winter-barley_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_grain-corn_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_grain-corn_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_grain-corn_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_grain-corn_winter-wheat_winter-rape_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_grain-corn_winter-wheat_winter-rape_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_grain-corn_winter-wheat_winter-rape_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_grain-corn_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_grain-corn_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_grain-corn_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_miscanthus_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_miscanthus_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_miscanthus_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_sugar-beet_winter-wheat_winter-barley_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_sugar-beet_winter-wheat_winter-barley_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_sugar-beet_winter-wheat_winter-barley_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_sugar-beet_winter-wheat_winter-barley_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_sugar-beet_winter-wheat_winter-barley_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_sugar-beet_winter-wheat_winter-barley_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_clover_winter-wheat_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_clover_winter-wheat_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_clover_winter-wheat_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_winter-wheat_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_winter-wheat_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_winter-wheat_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_winter-wheat_silage-corn_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_winter-wheat_silage-corn_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_winter-wheat_silage-corn_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_winter-wheat_winter-rape_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_winter-wheat_winter-rape_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_winter-wheat_winter-rape_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_winter-wheat_winter-rape_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_winter-wheat_winter-rape_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_winter-wheat_winter-rape_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_winter-wheat_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_winter-wheat_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_summer-wheat_winter-wheat_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_clover_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_clover_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_clover_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_clover_silage-corn_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_clover_silage-corn_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_clover_silage-corn_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_silage-corn_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_silage-corn_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_silage-corn_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_silage-corn_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_silage-corn_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_soybean_winter-rape_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_soybean_winter-rape_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_soybean_winter-rape_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_sugar-beet_silage-corn_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_sugar-beet_silage-corn_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_sugar-beet_silage-corn_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_sugar-beet_silage-corn_yellow-mustard_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_sugar-beet_silage-corn_yellow-mustard_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_sugar-beet_silage-corn_yellow-mustard_medium_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_winter-rape_high_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_winter-rape_low_Nfert_slurm.sh
sbatch --partition cpu_il svat_crop_nitrate_lahr_winter-wheat_winter-rape_medium_Nfert_slurm.sh