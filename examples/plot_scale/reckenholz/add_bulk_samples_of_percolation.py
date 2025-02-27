from pathlib import Path
import xarray as xr
import h5netcdf
from cftime import num2date
import pandas as pd
import numpy as onp

lys_experiments = ["lys2", "lys3", "lys8"]
for lys_experiment in lys_experiments:
    # load observations (measured data)
    path_obs = Path(__file__).parent / "observations" / "reckenholz_lysimeter.nc"
    ds_obs = xr.open_dataset(path_obs, engine="h5netcdf", group=lys_experiment)

    # assign date
    days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

    idx = ds_obs.date.values  # time index
    df_idx_bs = pd.DataFrame(index=date_obs, columns=['sol'])
    df_idx_bs.loc[:, 'sol'] = ds_obs['NO3_PERC'].isel(x=0, y=0).values
    idx_bs = df_idx_bs['sol'].dropna().index

    # calculate bulk samples of percolation
    sample_no = pd.DataFrame(index=idx_bs, columns=['sample_no'])
    sample_no['sample_no'] = range(len(sample_no.index))
    df_perc_bs = pd.DataFrame(index=idx, columns=['perc'])
    df_perc_bs.loc[:, 'perc'] = ds_obs['PERC'].isel(x=0, y=0).values
    df_perc_bs = df_perc_bs.join(sample_no)
    df_perc_bs.loc[:, 'sample_no'] = df_perc_bs.loc[:, 'sample_no'].bfill(limit=14)
    perc_obs_sum = df_perc_bs.groupby(['sample_no']).sum().loc[:, 'perc']
    sample_no['perc_bs'] = perc_obs_sum.values
    df_perc_bs = df_perc_bs.join(sample_no['perc_bs'])

    # add simulated bulk samples to the dataset
    ds_obs.close()
    del(ds_obs)
    with h5netcdf.File(path_obs, "a", decode_vlen_strings=False) as f:
        try:
            v = f[f"{lys_experiment}"].create_variable("PERC_bs", ("x", "y", "Time"), float, compression="gzip", compression_opts=1)
            v[0, 0, :] = df_perc_bs['perc_bs'].values.astype(float)
            v.attrs.update(long_name="Bulk samples of percolation", units="mm")
        except ValueError:
            var_obj = f[f"{lys_experiment}"].variables.get("PERC_bs")
            var_obj[0, 0, :] = df_perc_bs['perc_bs'].values.astype(float)
            var_obj.attrs.update(long_name="Bulk samples of percolation", units="mm")
