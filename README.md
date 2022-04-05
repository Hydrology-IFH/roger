# RoGeR

## How to cite
In case you use `roger` in other software or scientific publications,
please reference this package. It is published and has a DOI. It can be cited
as:
  ...

## Full Documentation
The full documentation can be found at: https://roger.readthedocs.io

## License
Which license?
© 2021, Robin Schwemmle (<robin.schwemmle@hydrology.uni-freiburg.de>)

## Description
`roger` is a hydrologic model written in Python.

## TODO
- implement distributed meteorological input (e.g. spatially distributed precipitation)
- implement runoff and channel routing (e.g. kinematic wave or hydraulic approach)
- implement RoGeR-Urban
- implement distributed model with run-on infiltration
- implement online-coupled solute transport
- use coarser spatial and temporal resolution for computation of
groundwater-related processes
- implement baseflow in the groundwater routine. requires surface water depth.
- add equations to docstrings
- write documentation for readthedocs (see `docs/`)
- write unit tests (see `tests/`) and prepare continous integration (see `.travis.yml`)
- use numba (or cython) to speed up computation time

## Installation and usage on Mac
Some instructions how to use the model from the repository. Please ensure that
Anaconda is installed (i.e. Python distribution). You can install it from
https://www.anaconda.com/products/individual.

The repository is still in development. Please update your local repository
regularly by using the `git pull` command.

GIT:

**First step:**
```bash
git clone https://github.com/schwemro/roger.git
cd roger
conda env create -f conda-environment.yml
```
**Second step:**
Activate the anaconda environment and launch Spyder
```bash
conda activate roger
spyder
```

## Installation and usage on Windows
Some instructions how to use the model from the repository. Please ensure that
Anaconda is installed (i.e. Python distribution). You can install it from
https://www.anaconda.com/products/individual.

The repository is still in development. Please update your local repository
regularly by using the `git pull` command.

GIT:

**First step:**
```bash
git clone https://github.com/schwemro/roger.git
cd roger
conda env create -f conda-environment.yml
```
**Second step:**
Activate the anaconda environment and launch Spyder
```bash
activate roger
spyder
```

**Third step:**
Run a test case. Navigate to `roger/setups/xxx`.

## Instructions for model setup and running the hydrologic model
Navigate through the `setups/` to learn more about the model setup and running
the model. More instructions follow soon (e.g. some Jupyter notebooks).

- `config_hm.yml`: configuration file used for setting up the hydrologic model
- `parameters.csv`: contains model parameters for each grid cell
- `initvals.csv`: contains initial values for each grid cell
- `ìnput`: contains precipitation data with 10 mins intervals (`P.txt`), air
temperature (`TA.txt`) and potential evapotranspiration data (`ET.txt`) with
daily intervals. Optionally, Groundwater head as lower boundary condition
(`Z_GW.txt`) and crop rotation (`crop_rotation.csv`)
- `hm.py`: Python file to setup and run the hydrologic model
- `output.log`: log-file of simulation

## Instructions for model setup and running the transport model
Navigate through the `examples/` to learn more about the model setup and running
the model. More instructions follow soon (e.g. some Jupyter notebooks).

- `config_hm.yml`: configuration file used for setting up the hydrologic model
- `config_tm.yml`: configuration file used for setting up the transport model
- `parameters.csv`: contains model parameters for each grid cell
- `initvals.csv`: contains initial values for each grid cell
- `ìnput`: contains precipitation data with 10 mins intervals (`P.txt`), air
temperature (`TA.txt`) and potential evapotranspiration data (`ET.txt`) with
daily intervals. Optionally, Groundwater head as lower boundary condition
(`Z_GW.txt`), crop rotation (`crop_rotation.csv`) and solute input can be
provided (Chloride: `Cl.txt`, Bromide: `Br.txt`, Oxygen-18: `d18O.txt`,
Deuterium: `d2H.txt`, Nitrate: `NO3.txt`)
- `hm.py`: Python file to setup and run the hydrologic model
- `tm.py`: Python file to setup and run the transport model
- `output.log`: log-file of simulation

The transport model is coupled to hydrologic model. First, you have to run the
hydrologic model before you can run the transport model.

### Configuration file
#### Experiment configurations
- `experiment_name`: name of your modelling experiment

#### Input configurations
- `prec_corr`: precipitation correction according to Richter (1995).
b1 = open location; b2 = slightly protected; b3 = moderately protected;
b4 = strongly protected
- `tres`: temporal resolution of potential evapotranspiration and air temperature

#### Model configurations
- `model`: model type
- `crop_phenology`: crop phenologic approach
- `crop_rotation`: if True provide file with crop rotation
- `nrows`: number of grid rows
- `ncols`: number of grid columns
- `cell_size`: spatial resolution (in square meter)

##### Model type
- `1D`: vertical and lateral hydrologic processes
- `1D-GW`: vertical and lateral hydrologic processes including groundwater
- `1D-ST`: offline solute transport with vertical and lateral hydrologic processes
- `1D-GW-ST`: offline solute transport with vertical and lateral hydrologic processes including groundwater
- `2D`: distributed model
- `2D-GW`: distributed model including groundwater
- `2D-ST`: distributed model with online solute transport
- `2D-ST-GW`: distributed model with online solute transport including groundwater
- `SVAT`: vertical hydrologic processes
- `SVAT-FILM`: vertical hydrologic processes with film flow infiltration
- `SVAT-ST`: offline solute transport with vertical hydrologic processes
- `CROP`: crop phenologic model
- `SNOW`: snow model
- `INTCP`: interception model
- `INF`: infiltration model for a single rainfall event
- `INF-FILM`: film flow infiltration model for a single rainfall event
- `PERC`: percolation model
- `CPR`: capillary rise model
- `SSQ`: subsurface runoff model
- `GW`: groundwater model

#### Output configurations
- `result_dir`: name of output directory
- `output_variables`: list the name of the hydrologic variables which are exported
- `output_variables_st`: list the name of the solute transport variables which are exported

#### Hydrologic model variables
- `p`: precipitation (in mm)
- `ta`: air temperature (in degC)
- `rain_ground`: rainfall at ground surface (in mm)
- `snow_ground`: snowfall at ground surface (in mm)
- `et_act`: actual evapotranspiration (in mm)
- `evap_int_top`: evaporation from upper interception layer (in mm)
- `evap_int_ground`: evaporation from lower interception layer (in mm)
- `evap_dep`: evaporation from surface depression storage (in mm)
- `et_soil`: soil evapotranspiration (in mm)
- `transp`: transpiration (in mm)
- `evap_soil`: soil evaporation (in mm)
- `inf`: infiltration (in mm)
- `inf_rz`: infiltration into root zone (in mm)
- `inf_mat_rz`: matrix infiltration into root zone (in mm)
- `inf_mp_rz`: macropore infiltration into root zone (in mm)
- `inf_sc_rz`: shrinkage crack infiltration into root zone (in mm)
- `inf_ss`: preferential infiltration into subsoil (in mm)
- `q_sur`: surface runoff (in mm)
- `q_rz`: vertical root zone drainage (in mm)
- `q_ss`: vertical subsoil drainage (in mm)
- `cpr_rz`: uplift from subsoil to root zone (in mm)
- `cpr_ss`: capillary into root zone (in mm)
- `S_snow`: snow cover (in mm)
- `dS_snow`: storage change of snow cover (in mm)
- `S_sur`: surface storage (in mm)
- `dS_sur`: storage change of surface storage (in mm)
- `S`: soil storage (in mm)
- `dS`: storage change of soil storage (in mm)
- `S_rz`: root zone storage (in mm)
- `dS_rz`: storage change of root zone storage (in mm)
- `S_ss`: subsoil storage (in mm)
- `dS_ss`: storage change of subsoil storage (in mm)
- `theta`: soil water content (-)
- `theta_rz`: soil water content of root zone (-)
- `theta_ss`: soil water content of subsoil (-)
- `k_stress_evap`: water stress coefficient for soil evaporation (-)

#### Crop phenologic variables
- `ground_cover_crop`: fraction of crop ground cover/crop canopy cover (-)
- `basal_crop_coeff`: basal crop coefficient (-)
- `veg_height`: plant height (in mm)
- `z_root_crop`: crop root depth (in mm)

### Set the hydrologic model parameters
Model parameters are stored in the file `parameters.csv`. In the following,
the model parameters are briefly described:
- `No`: number of grid cell
- `lu_id`: land use/cover ID (e.g. 8=grassland or 5=agriculture)
- `trees`: tree type (e.g. 0=no trees)
- `urban`: urban type (e.g. 0=not urban)
- `sealing`: degree of surface sealing (in %)
- `slope`: slope of land surface (in m/m)
- `S_tot_dep`: total storage of surface depression (in mm)
- `z_soil`: soil depth (in mm)
- `dmpv`: density of vertical macropores (in 1/m<sup>2</sup>)
- `lmpv`: length of vertical macropores (in mm)
- `dmph`: density of horizontal macropores (in 1/m<sup>2</sup>)
- `theta_ac`: air capacity of soil (-)
- `theta_ufc`: usable field capacity of soil (-)
- `theta_pwp`: permnanent wilting point of soil (-)
- `ks`: saturated hydraulic conductivity of soil (-)
- `kf`: hydraulic conductivity of bedrock (-)

### Set the initial values
Initial values are stored in the file `initvals.csv`. In the following,
the storages are briefly described:
- `No`: number of grid cell
- `S_int_top`: upper interception storage (in mm)
- `swe_top`: snow water equivalent in upper interception storage (in mm)
- `S_int_ground`: lower interception storage (in mm)
- `swe_ground`: snow water equivalent in lower interception storage (in mm)
- `S_dep`: surface depression storage (in mm)
- `S_snow`: depth of snow cover (in mm)
- `swe`: snow water equivalent of snow cover (in mm)
- `theta_rz`: soil water content of root zone (-)
- `theta_ss`: soil water content of subsoil (-)
- `z_root`: root depth (in mm). only required for crops!

### Crop phenology
Including crop phenology requires simulation start at 1st January. For winter
crops (e.g. winter wheat) initial value of `z_root` (e.g. 800 mm) is
required. Crop phenology is considered for grid cells with lu_id = 5xx. Root
growth is limited to 95 % of total soil depth.

- `doy`: linear growth based on julian dates
- `gdd`: exponential growth (gradual degree days) based on growing degree days
- `dyn`: dynamic growth (gradual degree days + water stress) based on growing degree days
- `doy_gdd`: exponential growth with fixed julian dates for sowing and harvesting
- `doy_dyn`: dynamic growth with fixed julian dates for sowing and harvesting

### Crop rotation
Crop rotation is defined in `crop_rotation.csv`. For each year (i.e. year of
sowing) summer and winter crops can be defined. No crop is defined by value
-9999. Winter crops can follow on summer crops. However, summer crops cannot
follow on winter crops except on winter green manure (winter catch crop).
Values for the year before the simulation period starts are required to set
the initial values.

### Description of land use (lu_id)
- `0`: sealed surface
- `5`: arable land
- `501`: bean
- `502`: amaranth
- `503`: other commercial crops
- `504`: artichoke
- `505`: berry
- `506`: ornamental plant
- `507`: nettle
- `508`: buckwheat
- `509`: pea
- `510`: strawberry
- `511`: esparcet
- `512`: sunflower
- `513`: vegetables
- `514`: flax
- `515`: early potatoes
- `516`: fodder root crops
- `517`: fodder legumes
- `518`: hemp
- `519`: home garden
- `520`: hop
- `521`: legumes
- `522`: intensive fruit-growing
- `523`: potato
- `524`: clover
- `525`: grain corn
- `526`: herbs
- `527`: false flax
- `528`: lentil
- `529`: lupine
- `530`: lucerne
- `531`: summer phacelia
- `532`: flat pea
- `533`: grape
- `534`: grape school
- `535`: rhubarb
- `536`: beetroot
- `537`: nuts
- `538`: summer mustard
- `539`: silage corn
- `540`: silphium
- `541`: soybean
- `542`: summer barley
- `543`: summer wheat
- `544`: summer oat
- `545`: summer rape
- `546`: summer triticale
- `547`: sunflower
- `548`: other fruit-growing
- `549`: sorghum
- `550`: asparagus
- `551`: orchards
- `552`: sweet potato
- `553`: tobacco
- `554`: helianthus
- `555`: vetch
- `556`: winter barley
- `557`: winter wheat
- `558`: winter oat
- `559`: winter rape
- `560`: winter triticale
- `561`: chicory
- `562`: sweet corn
- `563`: sugar beet
- `564`: winter green manure (Oct)
- `565`: summer grass
- `566`: winter grass
- `567`: clover
- `568`: winter phacelia
- `569`: winter green manure (Aug)
- `570`: winter green manure (Sep)
- `571`: summer grass (growing only)
- `572`: winter grass (growing only)
- `573`: summer grass (continued)
- `574`: winter grass (continued)
- `598`: no crop
- `599`: bare
- `6`: vineyard
- `7`: fruits
- `8`: grass
- `9`: complex parcel
- `10`: deciduous forest
- `11`: mixed forest
- `12`: coniferous forest
- `13`: wetland
- `14`: lake
- `15`: forest (unknown tree species)
- `16`: urban tree
- `20`: river
- `31`: gravel rooftop
- `32`: grass rooftop extensive
- `33`: grass rooftop intensive
- `41`: gravel
- `50`: percolation plant
- `98`: grass intensive
- `100`: urban
- `999`: no value
