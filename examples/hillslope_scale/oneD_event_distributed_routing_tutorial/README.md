# Tutorial for the spatially-distributed 1D-event-model

Lateral transfer between grid cells occurs with a single flow direction. The flow direction is based on the topography.


Brief overview of the files:

- `input`: Contains input data
- `output`: Contains output data
- `figures`: Contains figures
- `config.yml`: File to set settings and output variables
- `parameters.csv`: File that contains the model parameters
- `param_bounds.yml`: File to set settings and output variables
- `write_parameters.py`: Generates random model parameters based on given parameter boundaries (`param_bounds.yml`) and writes model parameter file
- `oneD_event.py`: 1D-event-model setup
- `merge_output.py`: Merges output into a single NetCDF file
- `netcdf_to_csv.py`: Writes output to csv files
- `make_figures_and_tables.py`: Produces figures and tables

# Date requirements

The following information is required to run the model. 

## Meteorological input data
The required meteorological input data is loaded from the input folder. The input folder should contain a tab-delimited text file
for precipitation (`PREC.txt`; 10 minutes time steps), air temperature (`TA.txt`; daily time steps) and potential evapotranspiration (`PET.txt`; daily time steps).

Format of `PREC.txt` (PREC in mm/10 minutes):
| YYYY  | MM    | DD    | hh    | mm    | PREC  |
| ------| ------| ------| ------| ------| ------|
| 2023  | 1     | 1     | 0     | 0     | 0     |
| 2023  | 1     | 1     | 0     | 10    | 0.3   |        
| 2023  | 1     | 1     | 0     | 20    | 1.0   |
| 2023  | 1     | 1     | 0     | 30    | 0.5   |        
| 2023  | 1     | 1     | 0     | 40    | 0.4   |
| 2023  | 1     | 1     | 0     | 50    | 0.7   |        
| 2023  | 1     | 1     | 1     | 0     | 0.6   |
| ...   | ...   | ...   | ...   | ...   | ...   |

Format of`TA.txt` (TA in degC):
| YYYY  | MM    | DD    | hh    | mm    | TA    |
| ------| ------| ------| ------| ------| ------|
| 2023  | 1     | 1     | 0     | 0     | 2     |
| 2023  | 1     | 1     | 0     | 0     | 3     |        
| ...   | ...   | ...   | ...   | ...   | ...   |

Format of `PET.txt` (PET in mm/day):
| YYYY  | MM    | DD    | hh    | mm    | PET   |
| ------| ------| ------| ------| ------| ------|
| 2023  | 1     | 1     | 0     | 0     | 2     |
| 2023  | 1     | 1     | 0     | 0     | 2.1   |        
| ...   | ...   | ...   | ...   | ...   | ...   |


where YYYY is the year, MM is the month, DD is the day, hh is the hour and mm is the minute.

## Model parameters
The model parameters (`parameters.csv`) are loaded from the same directory as `oneD.py`.

Random model parameters can be generated running the following script:
```
python write_parameters.py
```

Format of `parameters.csv`:
| lu_id | z_soil   | slope  | dmph | dmpv  | lmpv  | theta_pwp | theta_ufc | theta_ac | ks  | kf   |
| ------| ---------| -------| -----| ------| ------| ----------| ----------| ---------| ----| -----|
| 8     | 1000     | 0.05   | 25   | 25    | 200   | 0.2       | 0.11      | 0.09     | 5   | 2500 |  
| 8     | 1000     | 0.06   | 20   | 30    | 300   | 0.18      | 0.1       | 0.08     | 6   | 2500 |
| ...   | ...      | ...    | ...  | ...   | ...   | ...       | ...       | ...      | ... | ...  |

where *lu_id* is the land cover, *z_soil* is the soil depth (mm), *slope* is the surface slope (-), *dmph* is the density of horizontal macropores (1/$m^2$), *dmpv* is the density of vertical macropores (1/$m^2$), *lmpv* is the length of vertical macropores (mm), *theta_pwp* is soil water content of the permanent wilting point (-), *theta_ufc* is soil water content of the usable field capacity (-), *theta_ac* is soil water content of the air capacity (-), *ks* is the saturated hydraulic conductivity (mm/hour) and *kf* is the hydraulic conductivity of the bedrock (mm/hour).

## Model settings
Name of model experiment and spatial discretization are defined in `config.yml`.

## Model output
The variables written to the output files are defined in `config.yml`. Available variables
are listed [here](https://roger.readthedocs.io/en/latest/reference/variables.html#available-variables). Generally, storage variables
should be defined for `OUTPUT_COLLECT` and flux variables for `OUTPUT_RATE`.

# Workflow

The following workflow briefly describes the model application:

1. Prepare the meteorological input data (see Meteorological input data).

2. Generates the model parameters for the simulation:
```
python write_parameters.py
```

3. If required data is ready, the following script runs the simulation:

```
python oneD.py
```

4. After calculation is done, the simulation results can be merged into a single NetCDF-file:
```
python merge_output.py
```

5. (Optional) The following script converts the model output from NetCDF to csv.
```
python netcdf_to_csv.py
```

6. Figures and tables can be produced with the following script:
```
python make_figures_and_tables.py
``````