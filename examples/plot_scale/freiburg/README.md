# Crop irrigation management at district level of Freiburg, Germany

- `figures/`: contains figures
- `output/`: contains the simulations
- `input/`: contains meteorological data (`PREC.txt`, `PET.txt`, `TA.txt`) of the DWD station Freiburg and the defined crop rotations (`crop_rotation_scenarios/`) from 2000 to 2024

Available soil types:
- sandy soil type
- silty soil type
- clayey soil type

Available irrigation rules:
- 30-ufc: Irrigation demand is calculated if soil water content is less than 30% of usable field capacity
- 45-ufc: Irrigation demand is calculated if soil water content is less than 45% of usable field capacity
- 50-ufc: Irrigation demand is calculated if soil water content is less than 50% of usable field capacity
- 80-ufc: Irrigation demand is calculated if soil water content is less than 80% of usable field capacity
- crop-specific: Irrigation demand is specifically calculated for each crop

Crop rotations are repeated after three to four years. The following crop rotations are available:
- winter-wheat_clover: winter wheat, clover, clover, winter wheat
- winter-wheat_silage-corn: winter wheat, silage corn, winter wheat, silage corn
- summer-wheat_winter-wheat: summer wheat, winter wheat, summer wheat, winter wheat
- summer-wheat_clover_winter-wheat: summer wheat, clover, clover, winter wheat
- winter-wheat_clover_silage-corn: winter wheat, clover, clover, silage corn
- winter-wheat_sugar-beet_silage-corn: winter wheat, sugar beet, winter wheat, silage corn
- summer-wheat_winter-wheat_silage-corn: summer wheat, winter wheat, silage corn
- summer-wheat_winter-wheat_winter-rape: summer wheat, winter wheat, winter wheat, winter rape seed
- winter-wheat_winter-rape: winter wheat, winter wheat, winter wheat, winter rape seed
- winter-wheat_soybean_winter-rape: summer wheat, soy bean, winter wheat, winter rape seed
- sugar-beet_winter-wheat_winter-barley: sugar beet, winter wheat, winter wheat, winter barley
- grain-corn_winter-wheat_winter-rape: grain corn, winter wheat, winter rape, winter wheat
- grain-corn_winter-wheat_winter-barley: grain corn, winter wheat, winter barley
- grain-corn_winter-wheat_clover: grain corn, winter wheat, clover, clover
- winter-wheat_silage-corn_yellow-mustard: winter wheat, silage corn, winter wheat, silage corn including yellow mustard before winter crop
- summer-wheat_winter-wheat_yellow-mustard: summer wheat, winter wheat, summer wheat, winter wheat including yellow mustard before winter crop
- winter-wheat_sugar-beet_silage-corn_yellow-mustard: winter wheat, sugar beet, winter wheat, silage corn including yellow mustard before winter crop
- summer-wheat_winter-wheat_silage-corn_yellow-mustard: summer wheat, winter wheat, silage corn including yellow mustard before winter crop
- summer-wheat_winter-wheat_winter-rape_yellow-mustard: summer wheat, winter wheat, winter wheat, winter rape seed including yellow mustard before winter crop
- sugar-beet_winter-wheat_winter-barley_yellow-mustard: sugar beet, winter wheat, winter wheat, winter barley including yellow mustard before winter crop
- grain-corn_winter-wheat_winter-rape_yellow-mustard: grain corn, winter wheat, winter rape, winter wheat including yellow mustard before winter crop
- grain-corn_winter-wheat_winter-barley_yellow-mustard: grain corn, winter wheat, winter barley including yellow mustard before winter crop
- miscanthus: perennial miscanthus
- bare-grass: bare land turning into grass

### no_irrigation
No irrigation is applied i.e. irrigation demand is calculated without irrigation.

- `svat_crop.py`: Setup of the RoGeR model
- `merge_output.py`: Merges the model output into a single *.nc-file
- `write_simulations_to_csv.py`: Writes simulations to *.csv-file
- `write_job_script.py`: Writes shell script to generate the simulations for the available soil types, irrigation scenarios and crop roations
- `run_roger.sh`: Runs the RoGeR model to generate the simulations
- `parameters.csv`: contains the soil hydraulic paramters of RoGeR for three different soil types (z_soil: soil depth in mm; dmpv: density of vertical macropores in 1/$m^2$; lmpv: length of vertical macropores in mm; theta_ac: air capacity in -; theta_ufc: usable field capacity in -; theta_pwp: permanent wilting point in -; ks: saturated hydraulic conductivity in mm/h; kf: hydraulic conductivity of bedrock in mm/h; soil_fertility: soil fertility; clay: clay content in -)

### irrigation
Irrigation is applied according to five irrigation rules

- `svat_crop.py`: Setup of the RoGeR model
- `merge_output.py`: Merges the model output into a single *.nc-file
- `write_simulations_to_csv.py`: Writes simulations to *.csv-file
- `write_job_script.py`: Writes shell script to generate the simulations for the available soil types, irrigation scenarios and crop roations
- `run_roger.sh`: Runs the RoGeR model to generate the simulations
- `parameters.csv`: contains the soil hydraulic paramters of RoGeR for three different soil types (z_soil: soil depth in mm; dmpv: density of vertical macropores in 1/$m^2$; lmpv: length of vertical macropores in mm; theta_ac: air capacity in -; theta_ufc: usable field capacity in -; theta_pwp: permanent wilting point in -; ks: saturated hydraulic conductivity in mm/h; kf: hydraulic conductivity of bedrock in mm/h; soil_fertility: soil fertility; clay: clay content in -)
- `write_parameters_to_netcdf.py`: Writes soil hydraulic paramters of RoGeR to *.nc-file

### no_irrigation_nitrate
Nitrate leaching is simulated without irrigation.

### irrigation_nitrate
Nitrate leaching is simulated with irrigation.

## Workflow
! Windows user may change from `/` to `\` in the provided *.sh-files. Please check beforehand. !

1. Install RoGeR and the required Python libraries using Anaconda
2. After successfull installation activate the conda environment `roger`
3. Open the terminal and move to the directory
4. Run `./run_roger.sh`
5. Simulations will be saved to the `output/` directory