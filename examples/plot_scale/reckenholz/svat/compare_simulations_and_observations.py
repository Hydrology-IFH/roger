import os
from pathlib import Path
from cftime import num2date
import xarray as xr
import pandas as pd
import numpy as onp

import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs

base_path = Path(__file__).parent

lys_experiments = ["lys8", "lys3", "lys2"]
for lys_experiment in lys_experiments:
    # load simulation
    states_hm_file = base_path.parent / "output" / "svat" / f"SVAT_{lys_experiment}.nc"
    ds_sim = xr.open_dataset(states_hm_file, engine="h5netcdf")

    # load observations (measured data)
    path_obs = Path("/Users/robinschwemmle/Desktop/PhD/data/plot/reckenholz/reckenholz_lysimeter.nc")
    ds_obs = xr.open_dataset(path_obs, engine="h5netcdf", group=lys_experiment)

    # plot observed and simulated time series
    base_path_figs = base_path.parent / "figures" / "svat"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    time_origin = ds_sim['Time'].attrs['time_origin']
    days_sim = (ds_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim = num2date(days_sim, units=f"days since {ds_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim = ds_sim.assign_coords(date=("Time", date_sim))
    ds_obs = ds_obs.assign_coords(date=("Time", date_obs))
    vars_obs = ['PERC']
    vars_sim = ['q_ss']
    for var_obs, var_sim in zip(vars_obs, vars_sim):
        obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
        df_obs.loc[:, 'obs'] = obs_vals
        sim_vals = ds_sim[var_sim].isel(x=0, y=0).values
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(date_sim, sim_vals, df_obs)
        # plot observed and simulated time series
        fig = eval_utils.plot_obs_sim(df_eval, y_lab=labs._Y_LABS_DAILY[var_sim], x_lab='Time [year]')
        file_str = '%s_%s.pdf' % (var_sim, lys_experiment)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
        # plot cumulated observed and simulated time series
        df_eval = df_eval.dropna()
        fig = eval_utils.plot_obs_sim_cum(df_eval, y_lab=labs._Y_LABS_CUM[var_sim], x_lab='Time [year]')
        file_str = '%s_cum_%s.pdf' % (var_sim, lys_experiment)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
        fig = eval_utils.plot_obs_sim_cum_year_facet(df_eval, y_lab=labs._Y_LABS_CUM[var_sim], x_lab='Time\n[day-month-hydyear]')
        file_str = '%s_cum_year_facet_%s.pdf' % (var_sim, lys_experiment)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)

    vars_obs = ['THETA']
    vars_sim = ['theta']
    for var_obs, var_sim in zip(vars_obs, vars_sim):
        obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
        df_obs.loc[:, 'obs'] = obs_vals
        sim_vals = onp.nanmean(ds_sim[var_sim].isel(x=0, y=0).values, axis=-1)
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(date_sim, sim_vals, df_obs)
        # plot observed and simulated time series
        fig = eval_utils.plot_obs_sim(df_eval, y_lab=labs._Y_LABS_DAILY[var_sim], x_lab='Time [year]')
        file_str = '%s_%s.pdf' % (var_sim, lys_experiment)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
