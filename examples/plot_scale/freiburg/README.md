# Crop irrigation management at district level of Freiburg, Germany

This modelling experiment investigates the impact of crop irrigation on groundwater recharge and nitrate leaching. The impact is systematically investigated using a combination of soil types, irrigation demand rules and crop rotation scenarios.

- `figures/`: contains figures
- `output/`: contains the simulations of RoGeR
- `simulation.csv`: contains the time series of the variables simulated by RoGeR (precip: precipitation in mm/day; irrig: irrigation in mm/day; canopy_cover: crop canopy cover in -; z_root: crop root depth in mm; theta_rz: soil water content of the upper soil layer in -; theta_irrig: irrigation threshold of soil water content of the upper soil layer in -; theta_fc: soil water content at field capacity in -; transp: transpiration in mm/day; perc: percolation in mm/day; irrigation_demand: soil water deficit in mm; root_ventilation: available soil air for root ventilation in %; lu_id: land use identifier or RoGeR; crop_type: name of the crop)
- `input/`: contains meteorological data (`PREC.txt`, `PET.txt`, `TA.txt`) of the DWD station Freiburg and the defined crop rotations (`crop_rotation_scenarios/`) from 2000 to 2024
- `crop_water_stress.csv`: contains fraction of usable field capacity when crop water stress starts. IMPORTANT: Do not modify the file.
- `parameters.csv`: contains the soil hydraulic parameters of RoGeR for three different soil types (z_soil: soil depth in mm; dmpv: density of vertical macropores in 1/$m^2$; lmpv: length of vertical macropores in mm; theta_ac: air capacity in -; theta_ufc: usable field capacity in -; theta_pwp: permanent wilting point in -; ks: saturated hydraulic conductivity in mm/h; kf: hydraulic conductivity of bedrock in mm/h; soil_fertility: soil fertility; clay: clay content in -)
- `config.yml`: configuration file. Irrigation rules and crop rotation scenarios should be defined from the list below.

Available soil types:
- sandy soil type
- silty soil type
- clayey soil type
- Further soil types can be added to `parameters.csv`

Available irrigation demand rules:
- no_irrigation: No irrigation is applied
- 20-ufc: Irrigation demand is calculated if soil water content is less than 20% of usable field capacity
- 35-ufc: Irrigation demand is calculated if soil water content is less than 35% of usable field capacity
- 50-ufc: Irrigation demand is calculated if soil water content is less than 50% of usable field capacity
- crop-specific: Irrigation demand is specifically calculated for each crop

Crop rotations are repeated after three to four years. The following crop rotations are available:
- grain-corn
- grain-corn_yellow-mustard: yellow mustard, grain corn
- silage-corn
- silage-corn_yellow-mustard: yellow mustard, silage corn
- summer-barley
- summer-barley_yellow-mustard: yellow mustard, summer-barley
- clover
- winter-wheat
- winter-barley
- winter-rape
- faba-bean
- potato-early
- sugar-beet
- sugar-beet_yellow-mustard: yellow mustard, sugar beet
- vegetables
- strawberry
- asparagus
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
- bare-grass: bare land turning into grass land

### no_irrigation
No irrigation is applied i.e. irrigation demand is calculated without irrigation.

- `svat_crop.py`: Setup of the RoGeR model
- `merge_output.py`: Merges the model output into a single *.nc-file
- `write_simulations_to_csv.py`: Writes simulations to *.csv-file
- `write_job_script.py`: Writes shell script to generate the simulations for the available soil types, irrigation scenarios and crop roations
- `run_roger.sh`: Runs the RoGeR model to generate the simulations

### irrigation
30 mm per day are irrigated according to five irrigation demand rules.

- `svat_crop.py`: Setup of the RoGeR model
- `merge_output.py`: Merges the model output into a single *.nc-file
- `write_simulations_to_csv.py`: Writes simulations to *.csv-file
- `write_job_script.py`: Writes shell script to generate the simulations for the available soil types, irrigation scenarios and crop roations
- `run_roger.sh`: Runs the RoGeR model to generate the simulations

### nitrate
Nitrate leaching is simulated with irrigation and without irrigation.

- `parameters_sas_nitrate.nc`: contains the SAS and nitrate parameters of RoGeR for three different soil types
- `svat_crop_nitrate.py`: Setup of the RoGeR-SAS model to simulate the nitrate transport 
- `merge_output.py`: Merges the model output into a single *.nc-file
- `write_simulations_to_csv.py`: Writes simulations to *.csv-file
- `write_job_script.py`: Writes shell script to generate the simulations for the available soil types, irrigation scenarios and crop roations
- `run_roger.sh`: Runs the RoGeR model to generate the simulations

## Workflow
! Windows user may change from `/` to `\` in the provided *.sh-files. Please check beforehand. !

1. Install RoGeR and the required Python libraries using Anaconda `conda env create -f conda-environment.yml`
2. After successfull installation activate the conda environment `roger`
3. Open the terminal and move to the project directory.
4. Set the model parameters in `parameters.csv` (Skip this step if you agree using the provided parameters)
5. Set the considered irrigation demand rules and crop rotation scenarios in `config.yml`
6. Run `python write_parameters_to_netcdf.py` to write `parameters.csv` to NetCDF format
7. Run `./run_roger.sh`
8. Simulations will be saved to the `output/` directory
9. Run `./run_roger_nitrate.sh`
10. Simulations will be saved to the `output/` directory