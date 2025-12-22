from pathlib import Path
import xarray as xr
import geoxarray
import numpy as onp
import pandas as pd
import roger.lookuptables as lut

summer_crops = lut.SUMMER_CROPS.tolist()
winter_crops = lut.WINTER_CROPS.tolist()

base_path = Path(__file__).parent

params_file = base_path / "input" / "parameters_roger_25m__.nc"
ds_params = xr.open_dataset(params_file)
xcoords = ds_params.x.values
ycoords = ds_params.y.values
mask = ds_params['maskCatch'].values
cond_catch = mask == 1
lu_ids = onp.copy(ds_params['lanu'].values)
lu_ids[~cond_catch] = -9999  # set non-catchment area to -9999

# load the netcdf file
file = base_path / "input" / "crops_2018-2022.nc"
ds_cr_2018_2022 = xr.open_dataset(file)
spatial_ref = ds_cr_2018_2022.spatial_ref
lu_ids_2018_2022 = ds_cr_2018_2022['Nutzcode'].values
cond = onp.isnan(lu_ids_2018_2022)
lu_ids_2018_2022[cond] = -9999  # set nan to
lu_ids_2018_2022 = lu_ids_2018_2022.astype(onp.int16)

cond = (lu_ids_2018_2022 == 6).any(axis=0)
lu_ids[cond] = 6
cond = (lu_ids_2018_2022 == 7).any(axis=0)
lu_ids[cond] = 7
ll_ga_ids = onp.arange(501, 598).tolist() + [599, 8, 81, 82]
cond_ga = onp.isin(lu_ids_2018_2022, ll_ga_ids).all(axis=0)
cond5 = (lu_ids == 5)

cropland_gaps = onp.zeros(lu_ids.shape, dtype=float)
cropland_gaps[:, :] = onp.nan
cropland_gaps[~cond_ga & cond5] = 1.0

cropland_ga = onp.zeros(lu_ids.shape, dtype=float)
cropland_ga[:, :] = onp.nan
cropland_5 = onp.zeros(lu_ids.shape, dtype=float)
cropland_5[:, :] = onp.nan
cropland_5[cond5] = 1.0
cropland_ga[cond_ga] = 1.0

# update land use ids in cropland gaps with crop rotations from 2018-2022
lu_ids_updated = lu_ids.copy()
lu_ids_updated[~cond_catch] = ds_params['lanu'].values[~cond_catch]
lu_ids_updated[cond_ga] = 5
cond5_ = (lu_ids_updated == 5)
lu_ids_updated[~cond_ga & cond5_] = 8

ds_params['lanu'].values = lu_ids_updated
file = base_path / "input" / "parameters_roger_25m_.nc"
ds_params.to_netcdf(file)
