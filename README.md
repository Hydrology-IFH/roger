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
