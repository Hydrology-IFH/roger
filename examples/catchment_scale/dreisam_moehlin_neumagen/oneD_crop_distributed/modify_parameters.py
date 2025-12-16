from pathlib import Path
import xarray as xr
import geoxarray
import numpy as onp

base_path = Path(__file__).parent

params_file = base_path / "input" / "parameters_roger_25m_.nc"
ds_params = xr.open_dataset(params_file)
xcoords = ds_params.x.values
ycoords = ds_params.y.values
mask = ds_params['maskCatch'].values
cond_catch = onp.isfinite(ds_params['TP'].values) & (mask == 1)
mask_catch = onp.zeros(ds_params['TP'].shape, dtype=onp.int16)
mask_catch = onp.where(cond_catch, 1, 0)
ds_params['maskCatch'].values = mask_catch
cond_catch = mask_catch == 1
cond_gaps = (ds_params['lanu'].values == -1) & cond_catch
lu_ids = onp.copy(ds_params['lanu'].values)
lu_ids[cond_gaps] = 20
lu_ids[lu_ids < 0] = -9999
ds_params['lanu'].values = lu_ids
cond_zsoil0 = (ds_params['GRUND'].values <= 0) & cond_catch
z_soil = onp.copy(ds_params['GRUND'].values)
z_soil[cond_zsoil0] = 25.0
ds_params['GRUND'].values = z_soil
cond_kf0 = (ds_params['TP'].values <= 0) & cond_catch
kf = onp.copy(ds_params['TP'].values)
kf[cond_kf0] = 0.1
ds_params['TP'].values = kf
cond_lk0 = (ds_params['LK'].values <= 0) & cond_catch
lk = onp.copy(ds_params['LK'].values)
lk[cond_lk0] = 5.
ds_params['LK'].values = lk
zgw = onp.copy(ds_params['gwfa_gew'].values)
zgw[cond_zsoil0] = z_soil[cond_zsoil0] + 10.0
zgw[zgw < 0] = z_soil[zgw < 0] + 10.0
zgw[zgw <= z_soil] = z_soil[zgw <= z_soil] + 10.0
ds_params['gwfa_gew'].values = zgw

file = base_path / "parameters_roger.nc"
ds_params.to_netcdf(file)
ds_params.close()

