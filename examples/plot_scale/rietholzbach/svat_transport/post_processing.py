from pathlib import Path
import os
import glob
import datetime
import h5netcdf
import xarray as xr
from cftime import num2date, date2num
import pandas as pd
import numpy as onp
import click
import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs
import matplotlib as mpl
import seaborn as sns
mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
mpl.rcParams['font.size'] = 7
mpl.rcParams['axes.titlesize'] = 9
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7
mpl.rcParams['legend.fontsize'] = 7
mpl.rcParams['legend.title_fontsize'] = 8
sns.set_style("ticks")


@click.option("--sas-solver", type=click.Choice(['RK4', 'Euler', 'deterministic']), default='deterministic')
@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir, sas_solver):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent
    age_max = "age_max_11"
    # directory of results
    base_path_results = base_path / "results"
    if not os.path.exists(base_path_results):
        os.mkdir(base_path_results)
    base_path_results = base_path / "results" / sas_solver
    if not os.path.exists(base_path_results):
        os.mkdir(base_path_results)
    base_path_results = base_path / "results" / sas_solver / age_max
    if not os.path.exists(base_path_results):
        os.mkdir(base_path_results)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)
    base_path_figs = base_path / "figures" / sas_solver
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)
    base_path_figs = base_path / "figures" / sas_solver / age_max
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    transport_models = ['complete-mixing', 'piston', 'advection-dispersion', 'time-variant advection-dispersion']

    # merge model output into a single file
    for tm in transport_models:
        tm1 = tm.replace(" ", "_")
        path = str(base_path / sas_solver / age_max / f"SVATTRANSPORT_{tm1}_{sas_solver}.*.nc")
        diag_files = glob.glob(path)
        states_tm_file = base_path / sas_solver / age_max / f"states_{tm1}.nc"
        if not os.path.exists(states_tm_file):
            click.echo(f'Merge output files of {tm} into {states_tm_file.as_posix()}')
            with h5netcdf.File(states_tm_file, 'w', decode_vlen_strings=False) as f:
                f.attrs.update(
                    date_created=datetime.datetime.today().isoformat(),
                    title=f'RoGeR {tm} transport model results at Rietholzbach lysimeter site',
                    institution='University of Freiburg, Chair of Hydrology',
                    references='',
                    comment=f'SVAT {tm} transport model with free drainage'
                )
                # collect dimensions
                for dfs in diag_files:
                    with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                        f.attrs.update(
                            roger_version=df.attrs['roger_version']
                        )
                        # set dimensions with a dictionary
                        if not dfs.split('/')[-1].split('.')[1] == 'constant':
                            dict_dim = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'Time': len(df.variables['Time']), 'ages': len(df.variables['ages']), 'nages': len(df.variables['nages']), 'n_sas_params': len(df.variables['n_sas_params'])}
                            time = onp.array(df.variables.get('Time'))
                        if not f.dimensions:
                            f.dimensions = dict_dim
                            v = f.create_variable('x', ('x',), float, compression="gzip", compression_opts=1)
                            v.attrs['long_name'] = 'Number of model run'
                            v.attrs['units'] = ''
                            v[:] = onp.arange(dict_dim["x"])
                            v = f.create_variable('y', ('y',), float, compression="gzip", compression_opts=1)
                            v.attrs['long_name'] = ''
                            v.attrs['units'] = ''
                            v[:] = onp.arange(dict_dim["y"])
                            v = f.create_variable('ages', ('ages',), float, compression="gzip", compression_opts=1)
                            v.attrs['long_name'] = 'Water ages'
                            v.attrs['units'] = 'days'
                            v[:] = onp.arange(1, dict_dim["ages"]+1)
                            v = f.create_variable('nages', ('nages',), float, compression="gzip", compression_opts=1)
                            v.attrs['long_name'] = 'Water ages (cumulated)'
                            v.attrs['units'] = 'days'
                            v[:] = onp.arange(0, dict_dim["nages"])
                            v = f.create_variable('n_sas_params', ('n_sas_params',), float, compression="gzip", compression_opts=1)
                            v.attrs['long_name'] = 'Number of SAS parameters'
                            v.attrs['units'] = ''
                            v[:] = onp.arange(0, dict_dim["n_sas_params"])
                            v = f.create_variable('Time', ('Time',), float, compression="gzip", compression_opts=1)
                            var_obj = df.variables.get('Time')
                            v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                           units=var_obj.attrs["units"])
                            v[:] = time
                        for var_sim in list(df.variables.keys()):
                            var_obj = df.variables.get(var_sim)
                            if var_sim not in list(dict_dim.keys()) and ('Time', 'y', 'x') == var_obj.dimensions:
                                v = f.create_variable(var_sim, ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                                vals = onp.array(var_obj)
                                v[:, :, :] = vals.swapaxes(0, 2)
                                v.attrs.update(long_name=var_obj.attrs["long_name"],
                                               units=var_obj.attrs["units"])
                            elif var_sim not in list(dict_dim.keys()) and ('Time', 'n_sas_params', 'y', 'x') == var_obj.dimensions:
                                v = f.create_variable(var_sim, ('x', 'y', 'n_sas_params'), float, compression="gzip", compression_opts=1)
                                vals = onp.array(var_obj)
                                vals = vals.swapaxes(0, 3)
                                vals = vals.swapaxes(1, 2)
                                v[:, :, :] = vals[:, :, :, 0]
                                v.attrs.update(long_name=var_obj.attrs["long_name"],
                                               units=var_obj.attrs["units"])
                            elif var_sim not in list(dict_dim.keys()) and ('Time', 'ages', 'y', 'x') == var_obj.dimensions:
                                v = f.create_variable(var_sim, ('x', 'y', 'Time', 'ages'), float, compression="gzip", compression_opts=1)
                                vals = onp.array(var_obj)
                                vals = vals.swapaxes(0, 3)
                                vals = vals.swapaxes(1, 2)
                                vals = vals.swapaxes(2, 3)
                                v[:, :, :, :] = vals
                                v.attrs.update(long_name=var_obj.attrs["long_name"],
                                               units=var_obj.attrs["units"])
                            elif var_sim not in list(dict_dim.keys()) and ('Time', 'nages', 'y', 'x') == var_obj.dimensions:
                                v = f.create_variable(var_sim, ('x', 'y', 'Time', 'nages'), float, compression="gzip", compression_opts=1)
                                vals = onp.array(var_obj)
                                vals = vals.swapaxes(0, 3)
                                vals = vals.swapaxes(1, 2)
                                vals = vals.swapaxes(2, 3)
                                v[:, :, :, :] = vals
                                v.attrs.update(long_name=var_obj.attrs["long_name"],
                                               units=var_obj.attrs["units"])

    # load hydrologic simulation
    states_hm_file = base_path / "states_hm1_bootstrap.nc"
    ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")
    days_sim_hm = (ds_sim_hm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    time_origin = ds_sim_hm['Time'].attrs['time_origin']
    date_sim_hm = num2date(days_sim_hm, units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))
    ds_sim_hm.Time.attrs['time_origin'] = time_origin

    # load observations (measured data)
    path_obs = base_path.parent / "observations" / "rietholzbach_lysimeter.nc"
    ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")
    days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_obs = ds_obs.assign_coords(Time=("Time", date_obs))

    # load transport simulation
    for tm in transport_models:
        click.echo(f'Plot results of {tm}')
        tm1 = tm.replace(" ", "_")
        states_tm_file = base_path / sas_solver / age_max / f"states_{tm1}.nc"
        ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf", decode_times=False)
        date_sim_tm = num2date(ds_sim_tm['Time'].values, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))

        # compare observations and simulations
        nrows = ds_sim_tm.dims['x']
        idx = ds_sim_tm.Time.values  # time index
        # figure to compare simulations with observations
        fig, ax = plt.subplots(figsize=(14, 3.5))
        # DataFrame with sampled model parameters and the corresponding metrics
        df_params_metrics = pd.DataFrame(index=range(nrows))
        df_params_metrics.loc[:, 'dmpv'] = ds_sim_hm["dmpv"].values.flatten()
        df_params_metrics.loc[:, 'lmpv'] = ds_sim_hm["lmpv"].values.flatten()
        df_params_metrics.loc[:, 'theta_ac'] = ds_sim_hm["theta_ac"].values.flatten()
        df_params_metrics.loc[:, 'theta_ufc'] = ds_sim_hm["theta_ufc"].values.flatten()
        df_params_metrics.loc[:, 'theta_pwp'] = ds_sim_hm["theta_pwp"].values.flatten()
        df_params_metrics.loc[:, 'ks'] = ds_sim_hm["ks"].values.flatten()
        # loop over simulations
        d18O_perc_bs = onp.zeros((nrows, 1, len(idx)))
        for nrow in range(nrows):
            df_idx_bs = pd.DataFrame(index=date_obs, columns=['sol'])
            df_idx_bs.loc[:, 'sol'] = ds_obs['d18O_PERC'].isel(x=0, y=0).values
            idx_bs = df_idx_bs['sol'].dropna().index
            # calculate simulated oxygen-18 bulk sample
            df_perc_18O_obs = pd.DataFrame(index=date_obs, columns=['perc_obs', 'd18O_perc_obs'])
            df_perc_18O_obs.loc[:, 'perc_obs'] = ds_obs['PERC'].isel(x=0, y=0).values
            df_perc_18O_obs.loc[:, 'd18O_perc_obs'] = ds_obs['d18O_PERC'].isel(x=0, y=0).values
            sample_no = pd.DataFrame(index=idx_bs, columns=['sample_no'])
            sample_no = sample_no.loc['1997':'2007']
            sample_no['sample_no'] = range(len(sample_no.index))
            df_perc_18O_sim = pd.DataFrame(index=date_sim_tm, columns=['perc_sim', 'd18O_perc_sim'])
            df_perc_18O_sim['perc_sim'] = ds_sim_hm['q_ss'].isel(x=nrow, y=0).values
            df_perc_18O_sim['d18O_perc_sim'] = ds_sim_tm['C_iso_q_ss'].isel(x=nrow, y=0).values
            df_perc_18O_sim = df_perc_18O_sim.join(sample_no)
            df_perc_18O_sim.loc[:, 'sample_no'] = df_perc_18O_sim.loc[:, 'sample_no'].fillna(method='bfill', limit=14)
            perc_sum = df_perc_18O_sim.groupby(['sample_no']).sum().loc[:, 'perc_sim']
            sample_no['perc_sum'] = perc_sum.values
            df_perc_18O_sim = df_perc_18O_sim.join(sample_no['perc_sum'])
            df_perc_18O_sim.loc[:, 'perc_sum'] = df_perc_18O_sim.loc[:, 'perc_sum'].fillna(method='bfill', limit=14)
            df_perc_18O_sim['weight'] = df_perc_18O_sim['perc_sim'] / df_perc_18O_sim['perc_sum']
            df_perc_18O_sim['d18O_weight'] = df_perc_18O_sim['d18O_perc_sim'] * df_perc_18O_sim['weight']
            d18O_sample = df_perc_18O_sim.groupby(['sample_no']).sum().loc[:, 'd18O_weight']
            sample_no['d18O_sample'] = d18O_sample.values
            df_perc_18O_sim = df_perc_18O_sim.join(sample_no['d18O_sample'])
            cond = (df_perc_18O_sim['d18O_sample'] == 0)
            df_perc_18O_sim.loc[cond, 'd18O_sample'] = onp.NaN
            d18O_perc_bs[nrow, 0, :] = df_perc_18O_sim.loc[:, 'd18O_sample'].values
            # calculate observed oxygen-18 bulk sample
            df_perc_18O_obs.loc[:, 'd18O_perc_bs'] = df_perc_18O_obs['d18O_perc_obs'].fillna(method='bfill', limit=14)

            perc_sample_sum_obs = df_perc_18O_sim.join(df_perc_18O_obs).groupby(['sample_no']).sum().loc[:, 'perc_obs']
            sample_no['perc_obs_sum'] = perc_sample_sum_obs.values
            df_perc_18O_sim = df_perc_18O_sim.join(sample_no['perc_obs_sum'])
            df_perc_18O_sim.loc[:, 'perc_obs_sum'] = df_perc_18O_sim.loc[:, 'perc_obs_sum'].fillna(method='bfill', limit=14)

            # join observations on simulations
            obs_vals_bs = ds_obs['d18O_PERC'].isel(x=0, y=0).values
            sim_vals_bs = d18O_perc_bs[nrow, 0, :]
            sim_vals = ds_sim_tm['C_iso_q_ss'].isel(x=nrow, y=0).values
            df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
            df_obs.loc[:, 'obs'] = obs_vals_bs
            df_eval = eval_utils.join_obs_on_sim(date_sim_tm, sim_vals_bs, df_obs)
            df_eval = df_eval.dropna()
            # calculate metrics
            var_sim = 'C_iso_q_ss'
            obs_vals = df_eval.loc[:, 'obs'].values
            sim_vals = df_eval.loc[:, 'sim'].values
            key_kge = f'KGE_{var_sim}'
            df_params_metrics.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals, sim_vals)
            key_kge_alpha = f'KGE_alpha_{var_sim}'
            df_params_metrics.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals, sim_vals)
            key_kge_beta = f'KGE_beta_{var_sim}'
            df_params_metrics.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals, sim_vals)
            key_r = f'r_{var_sim}'
            df_params_metrics.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals, sim_vals)
            # plot observed and simulated d18O in percolation
            ax.plot(ds_sim_tm.Time.values, ds_sim_tm['C_iso_q_ss'].isel(x=nrow, y=0).values, color='red', zorder=2)
            ax.scatter(df_eval.index, df_eval.iloc[:, 0], color='red', s=4, zorder=1)
            ax.scatter(df_eval.index, df_eval.iloc[:, 1], color='blue', s=4, zorder=3)

        # write figure to .png
        ax.set_ylabel(r'$\delta^{18}$O [‰]')
        ax.set_xlabel('Time [year]')
        ax.set_xlim((ds_sim_tm.Time.values[0], ds_sim_tm.Time.values[-1]))
        ax.set_ylim((-20, -5))
        fig.tight_layout()
        file = base_path_figs / f"d18O_perc_sim_obs_{tm1}.png"
        fig.savefig(file, dpi=250)
        plt.close('all')

        # write to .txt
        file = base_path_results / f"params_metrics_{tm1}.txt"
        df_params_metrics.to_csv(file, header=True, index=False, sep="\t")

        # select best simulation
        idx_best = df_params_metrics['KGE_C_iso_q_ss'].idxmax()

        # dotty plots
        df_metrics = df_params_metrics.loc[:, ['KGE_C_iso_q_ss']]
        df_params = df_params_metrics.loc[:, ['dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks']]
        nrow = len(df_metrics.columns)
        ncol = len(df_params.columns)
        fig, ax1 = plt.subplots(nrow, ncol, sharey=True, figsize=(ncol*1.2, 1.2))
        ax = ax1.reshape(nrow, ncol)
        for i in range(nrow):
            for j in range(ncol):
                y = df_metrics.iloc[:, i]
                x = df_params.iloc[:, j]
                ax[i, j].scatter(x, y, s=4, c='grey', alpha=0.5)
                ax[i, j].set_xlabel('')
                ax[i, j].set_ylabel('')
                ax[i, j].set_ylim(-1, 1)
                # best model run
                y_best = df_metrics.iloc[idx_best, i]
                x_best = df_params.iloc[idx_best, j]
                ax[i, j].scatter(x_best, y_best, s=12, c='red', alpha=0.8)
        for j in range(ncol):
            xlabel = labs._LABS[df_params.columns[j]]
            ax[-1, j].set_xlabel(xlabel)

        ax[0, 0].set_ylabel(r'$KGE_{\delta^{18}O_{PERC}}$ [-]')
        fig.tight_layout()
        file = base_path_figs / f"dotty_plots_{tm1}.png"
        fig.savefig(file, dpi=250)
        plt.close('all')

        var_sim = 'q_ss'
        # calculate upper interquartile travel time for each time step
        df_age = pd.DataFrame(index=idx[2:], columns=['MTT', 'TT_50', 'TT25', 'TT75', 'MRT', 'RT_50', 'RT25', 'RT75'])
        df_age.loc[:, 'MTT'] = ds_sim_tm["ttavg_q_ss"].isel(x=idx_best, y=0).values[2:]
        df_age.loc[:, 'TT_50'] = ds_sim_tm["tt50_q_ss"].isel(x=idx_best, y=0).values[2:]
        df_age.loc[:, 'TT25'] = ds_sim_tm["tt25_q_ss"].isel(x=idx_best, y=0).values[2:]
        df_age.loc[:, 'TT75'] = ds_sim_tm["tt75_q_ss"].isel(x=idx_best, y=0).values[2:]
        df_age.loc[:, 'MRT'] = ds_sim_tm["rtavg_s"].isel(x=idx_best, y=0).values[2:]
        df_age.loc[:, 'RT_50'] = ds_sim_tm["rt50_s"].isel(x=idx_best, y=0).values[2:]
        df_age.loc[:, 'RT25'] = ds_sim_tm["rt25_s"].isel(x=idx_best, y=0).values[2:]
        df_age.loc[:, 'RT75'] = ds_sim_tm["rt75_s"].isel(x=idx_best, y=0).values[2:]
        df_age.loc[:, var_sim] = ds_sim_hm[var_sim].isel(x=idx_best, y=0).values[2:]

        # mean and median travel time over entire simulation period
        df_age_mean = pd.DataFrame(index=['avg'], columns=['MTT', 'TT_50', 'MRT', 'RT_50'])
        df_age_mean.loc['avg', 'MTT'] = onp.nanmean(df_age['MTT'].values)
        df_age_mean.loc['avg', 'TT_50'] = onp.nanmean(df_age['TT_50'].values)
        df_age_mean.loc['avg', 'MRT'] = onp.nanmean(df_age['MRT'].values)
        df_age_mean.loc['avg', 'RT_50'] = onp.nanmean(df_age['RT_50'].values)
        file_str = 'age_mean_%s_%s.csv' % (var_sim, tm1)
        path_csv = base_path_figs / file_str
        df_age_mean.to_csv(path_csv, header=True, index=True, sep="\t")

        # plot mean and median travel time
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 3))
        axes[0].plot(df_age.index, df_age['MTT'], ls='--', lw=2, color='magenta')
        axes[0].plot(df_age.index, df_age['TT_50'], ls=':', lw=2, color='purple')
        axes[0].fill_between(df_age.index, df_age['TT25'], df_age['TT75'], color='purple',
                             edgecolor=None, alpha=0.2)
        tt_50 = str(int(df_age_mean.loc['avg', 'TT_50']))
        tt_mean = str(int(df_age_mean.loc['avg', 'MTT']))
        axes[0].text(0.9, 0.93, r'$\overline{TT}_{50}$: %s days' % (tt_50), size=6, horizontalalignment='left',
                     verticalalignment='center', transform=axes[0].transAxes)
        axes[0].text(0.9, 0.83, r'$\overline{TT}$: %s days' % (tt_mean), size=6, horizontalalignment='left',
                     verticalalignment='center', transform=axes[0].transAxes)
        axes[0].set_ylabel('age\n[days]')
        axes[0].set_ylim(0,)
        axes[0].set_xlim((df_age.index[0], df_age.index[-1]))
        axes[1].bar(df_age.index, df_age[var_sim], width=-1, align='edge', edgecolor='grey')
        axes[1].set_ylim(0,)
        axes[1].invert_yaxis()
        axes[1].set_xlim((df_age.index[0], df_age.index[-1]))
        axes[1].set_ylabel('Percolation\n[mm $day^{-1}$]')
        axes[1].set_xlabel(r'Time [year]')
        fig.tight_layout()
        file_str = 'mean_median_tt_%s_%s.pdf' % (var_sim, tm1)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=250)

        # plot mean and median travel time and residence time
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 3))
        axes[0].plot(df_age.index, df_age['MRT'], ls='--', lw=2, color='magenta')
        axes[0].plot(df_age.index, df_age['RT_50'], ls=':', lw=2, color='purple')
        axes[0].fill_between(df_age.index, df_age['RT25'], df_age['RT75'], color='purple',
                             edgecolor=None, alpha=0.2)
        rt_50 = str(int(df_age_mean.loc['avg', 'RT_50']))
        rt_mean = str(int(df_age_mean.loc['avg', 'MRT']))
        axes[0].text(0.9, 0.93, r'$\overline{RT}_{50}$: %s days' % (rt_50), size=6, horizontalalignment='left',
                     verticalalignment='center', transform=axes[0].transAxes)
        axes[0].text(0.9, 0.83, r'$\overline{RT}$: %s days' % (rt_mean), size=6, horizontalalignment='left',
                     verticalalignment='center', transform=axes[0].transAxes)
        axes[0].set_ylabel('age\n[days]')
        axes[0].set_ylim(0,)
        axes[0].set_xlim((df_age.index[0], df_age.index[-1]))
        axes[1].plot(df_age.index, df_age['MTT'], ls='--', lw=2, color='magenta')
        axes[1].plot(df_age.index, df_age['TT_50'], ls=':', lw=2, color='purple')
        axes[1].fill_between(df_age.index, df_age['TT25'], df_age['TT75'], color='purple',
                             edgecolor=None, alpha=0.2)
        tt_50 = str(int(df_age_mean.loc['avg', 'TT_50']))
        tt_mean = str(int(df_age_mean.loc['avg', 'MTT']))
        axes[1].text(0.9, 0.93, r'$\overline{TT}_{50}$: %s days' % (tt_50), size=6, horizontalalignment='left',
                     verticalalignment='center', transform=axes[1].transAxes)
        axes[1].text(0.9, 0.83, r'$\overline{TT}$: %s days' % (tt_mean), size=6, horizontalalignment='left',
                     verticalalignment='center', transform=axes[1].transAxes)
        axes[1].set_ylabel('age\n[days]')
        axes[1].set_ylim(0,)
        axes[1].set_xlim((df_age.index[0], df_age.index[-1]))
        axes[1].set_xlabel(r'Time [year]')
        fig.tight_layout()
        file_str = 'mean_median_rt_tt_%s_%s.pdf' % (var_sim, tm1)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=250)

        # plot observed and simulated d18O in percolation
        fig, ax = plt.subplots(figsize=(6, 1.5))
        # join observations on simulations
        obs_vals_bs = ds_obs['d18O_PERC'].isel(x=0, y=0).values
        sim_vals_bs = d18O_perc_bs[idx_best, 0, :]
        sim_vals = ds_sim_tm['C_iso_q_ss'].isel(x=idx_best, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
        df_obs.loc[:, 'obs'] = obs_vals_bs
        df_eval = eval_utils.join_obs_on_sim(date_sim_tm, sim_vals_bs, df_obs)
        df_eval = df_eval.dropna()
        # plot observed and simulated d18O in percolation
        ax.plot(ds_sim_tm.Time.values, ds_sim_tm['C_iso_q_ss'].isel(x=idx_best, y=0).values, color='red', zorder=2)
        ax.scatter(df_eval.index, df_eval.iloc[:, 1], color='blue', s=4, zorder=3)
        # write figure to .png
        ax.set_ylabel(r'$\delta^{18}$O [‰]')
        ax.set_xlabel('Time [year]')
        ax.set_xlim((ds_sim_tm.Time.values[0], ds_sim_tm.Time.values[-1]))
        ax.set_ylim((-20, -5))
        fig.tight_layout()
        file = base_path_figs / f"d18O_perc_sim_obs_{tm1}_best.png"
        fig.savefig(file, dpi=250)
        plt.close('all')

        # plot numerical errors
        sd_dS_num_error = '{:.2e}'.format(onp.std(ds_sim_tm['dS_num_error'].isel(x=idx_best, y=0).values))
        max_dS_num_error = '{:.2e}'.format(onp.max(ds_sim_tm['dS_num_error'].isel(x=idx_best, y=0).values))
        sd_dC_num_error = '{:.2e}'.format(onp.std(ds_sim_tm['dC_num_error'].isel(x=idx_best, y=0).values))
        max_dC_num_error = '{:.2e}'.format(onp.max(ds_sim_tm['dC_num_error'].isel(x=idx_best, y=0).values))
        fig, axes = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(6, 3))
        axes[0].plot(ds_sim_tm.Time.values, ds_sim_tm['dS_num_error'].isel(x=idx_best, y=0).values, ls='-', lw=1, color='black')
        axes[0].set_ylabel('Bias\n[mm]')
        axes[0].set_ylim(0,)
        axes[0].set_xlim((ds_sim_tm.Time.values[0], ds_sim_tm.Time.values[-1]))
        axes[0].text(0.75, 0.93, r'Error SD: %s' % (sd_dS_num_error), size=6, horizontalalignment='left',
                     verticalalignment='center', transform=axes[0].transAxes)
        axes[0].text(0.75, 0.83, r'Error Max: %s' % (max_dS_num_error), size=6, horizontalalignment='left',
                     verticalalignment='center', transform=axes[0].transAxes)
        axes[1].plot(ds_sim_tm.Time.values, ds_sim_tm['dC_num_error'].isel(x=idx_best, y=0).values, ls='-', lw=1, color='black')
        axes[1].set_ylabel('Bias\n[mg/l]')
        axes[1].set_ylim(0,)
        axes[1].set_xlim((ds_sim_tm.Time.values[0], ds_sim_tm.Time.values[-1]))
        axes[1].text(0.75, 0.93, r'Error SD: %s' % (sd_dC_num_error), size=6, horizontalalignment='left',
                     verticalalignment='center', transform=axes[1].transAxes)
        axes[1].text(0.75, 0.83, r'Error Max: %s' % (max_dC_num_error), size=6, horizontalalignment='left',
                     verticalalignment='center', transform=axes[1].transAxes)
        axes[1].set_xlabel(r'Time [year]')
        fig.tight_layout()
        file_str = 'num_errors_%s.pdf' % (tm1)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=250)

        # # write states of best hydrologic simulation corresponding to best transport simulation
        # ds_sim_hm_best = ds_sim_hm.loc[dict(x=idx_best)]
        # ds_sim_hm_best.attrs['title'] = f'Best hydrologic simulation corresponding to best {tm} oxygen-18 simulation'
        # days = date2num(ds_sim_hm_best["Time"].values.astype('M8[ms]').astype('O'), units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}", calendar='standard')
        # ds_sim_hm_best = ds_sim_hm_best.assign_coords(Time=("Time", days))
        # ds_sim_hm_best.Time.attrs['units'] = "days"
        # ds_sim_hm_best.Time.attrs['time_origin'] = ds_sim_hm['Time'].attrs['time_origin']
        # file = base_path / f"states_hm_best_for_{tm1}.nc"
        # ds_sim_hm_best.to_netcdf(file, engine="h5netcdf")

        # # write simulated bulk sample to output file
        # ds_sim_tm = ds_sim_tm.load()  # required to release file lock
        # ds_sim_tm = ds_sim_tm.close()
        # del ds_sim_tm
        # states_tm_file = base_path / sas_solver / age_max / f"states_{tm1}.nc"
        # with h5netcdf.File(states_tm_file, 'a', decode_vlen_strings=False) as f:
        #     try:
        #         v = f.create_variable('d18O_perc_bs', ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
        #         v[:, :, :] = d18O_perc_bs
        #         v.attrs.update(long_name="bulk sample of d18O in percolation",
        #                        units="permil")
        #     except ValueError:
        #         v = f.get('d18O_perc_bs')
        #         v[:, :, :] = d18O_perc_bs
        #         v.attrs.update(long_name="bulk sample of d18O in percolation",
        #                        units="permil")
    return


if __name__ == "__main__":
    main()
