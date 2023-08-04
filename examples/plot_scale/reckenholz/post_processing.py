from pathlib import Path
import os
import glob
import datetime
import h5netcdf
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import matplotlib.pyplot as plt
import click
import seaborn as sns
sns.set_style('ticks', {'xtick.major.size': 8, 'ytick.major.size': 8})
sns.set_context("paper", font_scale=1.5)
import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs


@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    # load observations (measured data)
    lys_experiments = ["lys2", "lys3", "lys4", "lys8", "lys9", "lys2_bromide", "lys8_bromide", "lys9"]
    for lys_experiment in lys_experiments:
        path_obs = Path(__file__).parent / "observations" / "reckenholz_lysimeter.nc"
        ds_obs = xr.open_dataset(path_obs, engine="h5netcdf", group=lys_experiment)
        # assign date
        days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_obs = ds_obs.assign_coords(date=("Time", date_obs))
        df_obs = pd.DataFrame(index=date_obs)

        # load best monte carlo run
        states_hm_file = base_path / "svat_monte_carlo" / "states_hm.nc"
        ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf", group=lys_experiment)
        # assign date
        days_sim_hm = (ds_sim_hm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date_sim_hm = num2date(days_sim_hm, units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_sim_hm = ds_sim_hm.assign_coords(date=("Time", date_sim_hm))

        # compare simulation and observation
        vars_obs = ['PERC', 'dWEIGHT']
        vars_sim = ['q_ss', 'dS']
        for var_obs, var_sim in zip(vars_obs, vars_sim):
            obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
            df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
            df_obs.loc[:, 'obs'] = obs_vals
            sim_vals = ds_sim_hm[var_sim].isel(x=0, y=0).values
            # join observations on simulations
            df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
            df_eval = df_eval.iloc[3:, :]
            # plot observed and simulated time series
            fig = eval_utils.plot_obs_sim(df_eval, labs._Y_LABS_DAILY[var_sim])
            file_str = '%s_%s.pdf' % (var_sim, lys_experiment)
            path_fig = base_path_figs / file_str
            fig.savefig(path_fig, dpi=250)
            # plot cumulated observed and simulated time series
            fig = eval_utils.plot_obs_sim_cum(df_eval, labs._Y_LABS_CUM[var_sim], x_lab='Time [year]')
            file_str = '%s_cum_%s.pdf' % (var_sim, lys_experiment)
            path_fig = base_path_figs / file_str
            fig.savefig(path_fig, dpi=250)
            fig = eval_utils.plot_obs_sim_cum_year_facet(df_eval, labs._Y_LABS_CUM[var_sim], x_lab='Time\n[day-month-hydyear]')
            file_str = '%s_cum_year_facet_%s.pdf' % (var_sim, lys_experiment)
            path_fig = base_path_figs / file_str
            fig.savefig(path_fig, dpi=250)

        vars_obs = ['PREC']
        vars_sim = ['prec']
        for var_obs, var_sim in zip(vars_obs, vars_sim):
            obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
            df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
            df_obs.loc[:, 'obs'] = obs_vals
            # plot observed time series
            fig = eval_utils.plot_sim(df_obs, labs._Y_LABS_DAILY[var_sim])
            file_str = '%s.pdf' % (var_sim)
            path_fig = base_path_figs / file_str
            fig.savefig(path_fig, dpi=250)
            # plot cumulated observed time series
            fig = eval_utils.plot_sim_cum(df_obs, labs._Y_LABS_CUM[var_sim], x_lab='Time [year]')
            file_str = '%s_cum.pdf' % (var_sim)
            path_fig = base_path_figs / file_str
            fig.savefig(path_fig, dpi=250)
            fig = eval_utils.plot_sim_cum_year_facet(df_obs, labs._Y_LABS_CUM[var_sim], x_lab='Time\n[day-month-hydyear]')
            file_str = '%s_cum_year_facet.pdf' % (var_sim)
            path_fig = base_path_figs / file_str
            fig.savefig(path_fig, dpi=250)

        vars_obs = ['TA']
        vars_sim = ['ta']
        for var_obs, var_sim in zip(vars_obs, vars_sim):
            obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
            df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
            df_obs.loc[:, 'obs'] = obs_vals
            # plot observed time series
            fig = eval_utils.plot_sim(df_obs, labs._Y_LABS_DAILY[var_sim])
            file_str = '%s.pdf' % (var_obs)
            path_fig = base_path_figs / file_str
            fig.savefig(path_fig, dpi=250)
    return


if __name__ == "__main__":
    main()
