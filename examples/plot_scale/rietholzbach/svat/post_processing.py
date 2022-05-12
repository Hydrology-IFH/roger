import shutil
import glob
from pathlib import Path
import datetime
from cftime import num2date
import h5netcdf
import xarray as xr
import pandas as pd
import numpy as onp

import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs

base_path = Path(__file__).parent
# merge model output into single file
path = str(base_path / "SVAT.*.nc")
diag_files = glob.glob(path)
states_hm_file = base_path / "states_hm.nc"
with h5netcdf.File(states_hm_file, 'w', decode_vlen_strings=False) as f:
    f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title='RoGeR model results at Rietholzbach Lysimeter site',
        institution='University of Freiburg, Chair of Hydrology',
        references='',
        comment='SVAT model with free drainage'
    )
    for dfs in diag_files:
        with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
            # set dimensions with a dictionary
            if not f.dimensions:
                dict_dim = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'Time': len(df.variables['Time'])}
                f.dimensions = dict_dim
                v = f.create_variable('x', ('x',), float)
                v.attrs['long_name'] = 'Zonal coordinate'
                v.attrs['units'] = 'meters'
                v[:] = onp.arange(dict_dim["x"])
                v = f.create_variable('y', ('y',), float)
                v.attrs['long_name'] = 'Meridonial coordinate'
                v.attrs['units'] = 'meters'
                v[:] = onp.arange(dict_dim["y"])
                v = f.create_variable('Time', ('Time',), float)
                var_obj = df.variables.get('Time')
                with h5netcdf.File(base_path / "input" / 'forcing.nc', "r", decode_vlen_strings=False) as infile:
                    time_origin = infile.variables['Time'].attrs['time_origin']
                v.attrs.update(time_origin=time_origin,
                                units=var_obj.attrs["units"])
                v[:] = onp.array(var_obj)
            for key in list(df.variables.keys()):
                var_obj = df.variables.get(key)
                if key not in list(f.dimensions.keys()) and var_obj.ndim == 3:
                    v = f.create_variable(key, ('x', 'y', 'Time'), float)
                    vals = onp.array(var_obj)
                    v[:, :, :] = vals.swapaxes(0, 2)
                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                    units=var_obj.attrs["units"])

# move hydrologic states to directory of transport model
base_path_tm = base_path.parent / "svat_transport"
states_hm_file1 = base_path_tm / "states_hm.nc"
shutil.copy(states_hm_file, states_hm_file1)

# load simulation
ds_sim = xr.open_dataset(states_hm_file, engine="h5netcdf")

# load observations (measured data)
path_obs = Path("/Users/robinschwemmle/Desktop/PhD/data/plot/rietholzbach/rietholzbach_lysimeter.nc")
ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")

# plot observed and simulated time series
base_path_figs = base_path / "figures"

# assign date
days_sim = (ds_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
date_sim = num2date(days_sim, units=f"days since {ds_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
ds_sim = ds_sim.assign_coords(date=("Time", date_sim))
ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

# compare simulation and observation
vars_obs = ['AET', 'PERC']
vars_sim = ['aet', 'q_ss']
for var_obs, var_sim in zip(vars_obs, vars_sim):
    obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
    df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
    df_obs.loc[:, 'obs'] = obs_vals
    sim_vals = ds_sim[var_sim].isel(x=0, y=0).values
    # join observations on simulations
    df_eval = eval_utils.join_obs_on_sim(date_sim, sim_vals, df_obs)
    # plot observed and simulated time series
    fig = eval_utils.plot_obs_sim(df_eval, labs._Y_LABS_DAILY[var_sim], fmt_x='date')
    file_str = '%s.pdf' % (var_sim)
    path_fig = base_path_figs / file_str
    fig.savefig(path_fig, dpi=250)
    # plot cumulated observed and simulated time series
    fig = eval_utils.plot_obs_sim_cum(df_eval, labs._Y_LABS_CUM[var_sim], fmt_x='date', x_lab='Time [year]')
    file_str = '%s_cum.pdf' % (var_sim)
    path_fig = base_path_figs / file_str
    fig.savefig(path_fig, dpi=250)
    fig = eval_utils.plot_obs_sim_cum_year_facet(df_eval, labs._Y_LABS_CUM[var_sim], x_lab='Time\n[day-month-hydyear]')
    file_str = '%s_cum_year_facet.pdf' % (var_sim)
    path_fig = base_path_figs / file_str
    fig.savefig(path_fig, dpi=250)

# plot simulations
vars_sim = ['S_snow']
for var_sim in vars_sim:
    sim_vals = ds_sim[var_sim].isel(x=0, y=0).values
    df_sim = pd.DataFrame(index=date_sim, columns=['sim'])
    df_sim.loc[:, 'sim'] = sim_vals
    # plot observed and simulated time series
    fig = eval_utils.plot_sim(df_sim, labs._Y_LABS_DAILY[var_sim], fmt_x='date')
    file_str = '%s_sim.pdf' % (var_sim)
    path_fig = base_path_figs / file_str
    fig.savefig(path_fig, dpi=250)
