import glob
import os
from pathlib import Path
import datetime
from cftime import num2date
import h5netcdf
import xarray as xr
import pandas as pd
import numpy as onp
import click
import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs
import matplotlib as mpl
import seaborn as sns
mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
sns.set_style("ticks")


@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent
    # directory of results
    base_path_results = base_path / "results"
    if not os.path.exists(base_path_results):
        os.mkdir(base_path_results)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    # merge model output into single file
    states_hm_mc_file = base_path / "states_hm_monte_carlo.nc"
    if not os.path.exists(states_hm_mc_file):
        click.echo(f'Merge output files into {states_hm_mc_file.as_posix()}')
        path = str(base_path / "SVAT.*.nc")
        diag_files = glob.glob(path)
        with h5netcdf.File(states_hm_mc_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='RoGeR Monte Carlo simulations at Rietholzbach Lysimeter site',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment='First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).',
                model_structure='SVAT model with free drainage',
            )
            # collect dimensions
            for dfs in diag_files:
                with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                    f.attrs.update(
                        roger_version=df.attrs['roger_version']
                    )
                    # set dimensions with a dictionary
                    if not dfs.split('/')[-1].split('.')[1] == 'constant':
                        dict_dim = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'Time': len(df.variables['Time'])}
                        time = onp.array(df.variables.get('Time'))
            for dfs in diag_files:
                with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                    if not f.dimensions:
                        f.dimensions = dict_dim
                        v = f.create_variable('x', ('x',), float, compression="gzip", compression_opts=1)
                        v.attrs['long_name'] = 'model run'
                        v.attrs['units'] = ''
                        v[:] = onp.arange(dict_dim["x"])
                        v = f.create_variable('y', ('y',), float, compression="gzip", compression_opts=1)
                        v.attrs['long_name'] = ''
                        v.attrs['units'] = ''
                        v[:] = onp.arange(dict_dim["y"])
                        v = f.create_variable('Time', ('Time',), float, compression="gzip", compression_opts=1)
                        var_obj = df.variables.get('Time')
                        v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                       units=var_obj.attrs["units"])
                        v[:] = time
                    for var_sim in list(df.variables.keys()):
                        var_obj = df.variables.get(var_sim)
                        if var_sim not in list(f.dimensions.keys()) and var_obj.ndim == 3 and var_obj.shape[0] > 2:
                            v = f.create_variable(var_sim, ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                            vals = onp.array(var_obj)
                            v[:, :, :] = vals.swapaxes(0, 2)
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
                        elif var_sim not in list(f.dimensions.keys()) and var_obj.ndim == 3 and var_obj.shape[0] <= 2:
                            v = f.create_variable(var_sim, ('x', 'y'), float, compression="gzip", compression_opts=1)
                            vals = onp.array(var_obj)
                            v[:, :] = vals.swapaxes(0, 2)[:, :, 0]
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])

    # load simulation
    ds_sim = xr.open_dataset(states_hm_mc_file, engine="h5netcdf")
    # assign date
    days_sim = (ds_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim = num2date(days_sim, units=f"days since {ds_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim = ds_sim.assign_coords(date=("Time", date_sim))

    # load observations (measured data)
    path_obs = Path(__file__).parent.parent / "observations" / "rietholzbach_lysimeter.nc"
    ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")
    # assign date
    days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

    # average observed soil water content of previous 5 days
    window = 5
    df_thetap = pd.DataFrame(index=date_obs,
                             columns=['doy', 'theta', 'sc'])
    df_thetap.loc[:, 'doy'] = df_thetap.index.day_of_year
    df_thetap.loc[:, 'theta'] = onp.mean(ds_obs['THETA'].isel(x=0, y=0).values, axis=0)
    df_thetap.loc[df_thetap.index[window-1]:, f'theta_avg{window}'] = df_thetap.loc[:, 'theta'].rolling(window=window).mean().iloc[window-1:].values
    df_thetap.iloc[:window, 2] = onp.nan
    theta_lower = df_thetap.loc[:, f'theta_avg{window}'].quantile(0.1)
    theta_upper = df_thetap.loc[:, f'theta_avg{window}'].quantile(0.9)
    cond1 = (df_thetap[f'theta_avg{window}'] < theta_lower)
    cond2 = (df_thetap[f'theta_avg{window}'] >= theta_lower) & (df_thetap[f'theta_avg{window}'] < theta_upper)
    cond3 = (df_thetap[f'theta_avg{window}'] >= theta_upper)
    df_thetap.loc[cond1, 'sc'] = 1  # dry
    df_thetap.loc[cond2, 'sc'] = 2  # normal
    df_thetap.loc[cond3, 'sc'] = 3  # wet

    # DataFrame with sampled model parameters and the corresponding metrics
    nx = ds_sim.dims['x']  # number of rows
    ny = ds_sim.dims['y']  # number of columns
    file = base_path_results / "params_metrics.txt"
    if not os.path.exists(file):
        click.echo('Calculate metrics ...')
        df_params_metrics = pd.DataFrame(index=range(nx * ny))
        # sampled model parameters
        df_params_metrics.loc[:, 'dmpv'] = ds_sim["dmpv"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'lmpv'] = ds_sim["lmpv"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'theta_eff'] = ds_sim["theta_eff"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'frac_lp'] = ds_sim["frac_lp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'frac_fp'] = ds_sim["frac_fp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'theta_ac'] = ds_sim["theta_ac"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'theta_ufc'] = ds_sim["theta_ufc"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'theta_pwp'] = ds_sim["theta_pwp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'ks'] = ds_sim["ks"].isel(y=0).values.flatten()
        # calculate metrics
        vars_sim = ['aet', 'q_ss', 'dS']
        vars_obs = ['AET', 'PERC', 'dWEIGHT']
        for var_sim, var_obs in zip(vars_sim, vars_obs):
            if var_sim == 'theta':
                obs_vals = onp.mean(ds_obs['THETA'].isel(x=0, y=0).values, axis=0)
            elif var_sim == 'theta_rz':
                obs_vals = onp.mean(ds_obs['THETA'].isel(x=0, y=0).values[:5, :], axis=0)
            elif var_sim == 'theta_ss':
                obs_vals = onp.mean(ds_obs['THETA'].isel(x=0, y=0).values[5:, :], axis=0)
            else:
                obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
            df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
            df_obs.loc[:, 'obs'] = obs_vals
            for nrow in range(nx * ny):
                sim_vals = ds_sim[var_sim].isel(x=nrow, y=0).values
                # join observations on simulations
                df_eval = eval_utils.join_obs_on_sim(date_sim, sim_vals, df_obs)

                if var_sim in ['dS', 'dS_s']:
                    df_eval.loc['2000-01':'2000-06', :] = onp.nan
                    df_eval = df_eval.dropna()
                    obs_vals = df_eval.loc[:, 'obs'].values
                    sim_vals = df_eval.loc[:, 'sim'].values
                    key_kge = 'KGE_' + var_sim
                    df_params_metrics.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals, sim_vals)
                    key_r = 'r_' + var_sim
                    df_params_metrics.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals, sim_vals)

                else:
                    # skip first seven days for warmup
                    df_eval.loc[:'1997-01-07', :] = onp.nan
                    df_eval = df_eval.dropna()
                    obs_vals = df_eval.loc[:, 'obs'].values
                    sim_vals = df_eval.loc[:, 'sim'].values
                    key_kge = 'KGE_' + var_sim
                    df_params_metrics.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals, sim_vals)

                # avoid defragmentation of DataFrame
                click.echo(f'{var_sim}: {nrow}')
                df_params_metrics = df_params_metrics.copy()
        # Calculate multi-objective metric
        df_params_metrics.loc[:, 'E_multi'] = 0.2 * df_params_metrics.loc[:, 'r_dS'] + 0.4 * df_params_metrics.loc[:, 'KGE_aet'] + 0.4 * df_params_metrics.loc[:, 'KGE_q_ss']
        df_params_metrics.loc[:, 'KGE_multi'] = 0.2 * df_params_metrics.loc[:, 'KGE_dS'] + 0.4 * df_params_metrics.loc[:, 'KGE_aet'] + 0.4 * df_params_metrics.loc[:, 'KGE_q_ss']

        # write .txt-file
        file = base_path_results / "params_metrics.txt"
        df_params_metrics.to_csv(file, header=True, index=False, sep="\t")

    else:
        df_params_metrics = pd.read_csv(file, header=0, index_col=False, sep="\t")
        # dotty plots
        click.echo('Dotty plots ...')
        df_metrics = df_params_metrics.loc[:, ['KGE_aet', 'KGE_dS', 'KGE_q_ss', 'KGE_multi']]
        df_params = df_params_metrics.loc[:, ['dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks']]
        nrow = len(df_metrics.columns)
        ncol = len(df_params.columns)
        fig, ax = plt.subplots(nrow, ncol, sharey='row', figsize=(14, 7))
        for i in range(nrow):
            for j in range(ncol):
                y = df_metrics.iloc[:, i]
                x = df_params.iloc[:, j]
                ax[i, j].scatter(x, y, s=4, c='grey', alpha=0.5)
                ax[i, j].set_xlabel('')
                ax[i, j].set_ylabel('')

        for j in range(ncol):
            xlabel = labs._LABS[df_params.columns[j]]
            ax[-1, j].set_xlabel(xlabel)

        ax[0, 0].set_ylabel('$KGE_{ET}$ [-]')
        ax[1, 0].set_ylabel(r'$KGE_{\Delta S}$ [-]')
        ax[2, 0].set_ylabel('$KGE_{PERC}$ [-]')
        ax[3, 0].set_ylabel('$KGE_{multi}$\n [-]')

        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        file = base_path_figs / "dotty_plots_.png"
        fig.savefig(file, dpi=250)
        plt.close('all')

        # select best model run
        idx_best1 = df_params_metrics['KGE_multi'].idxmax()

        # write states of best simulation
        click.echo('Write best simulation ...')
        states_hm_mc_file = base_path / "states_hm_monte_carlo.nc"
        states_hm_file = base_path / "states_hm1.nc"
        with h5netcdf.File(states_hm_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='RoGeR best Monte Carlo simulation at Rietholzbach Lysimeter site',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment='First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).',
                model_structure='SVAT model with free drainage',
            )
            with h5netcdf.File(states_hm_mc_file, 'r', decode_vlen_strings=False) as df:
                f.attrs.update(
                    roger_version=df.attrs['roger_version']
                )
                # set dimensions with a dictionary
                dict_dim = {'x': 1, 'y': 1, 'Time': len(df.variables['Time'])}
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
                    v = f.create_variable('Time', ('Time',), float, compression="gzip", compression_opts=1)
                    var_obj = df.variables.get('Time')
                    v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                   units=var_obj.attrs["units"])
                    v[:] = onp.array(var_obj)
                for var_sim in list(df.variables.keys()):
                    var_obj = df.variables.get(var_sim)
                    if var_sim not in list(f.dimensions.keys()) and ('x', 'y', 'Time') == var_obj.dimensions:
                        v = f.create_variable(var_sim, ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        v[:, :, :] = vals[idx_best1, :, :]
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                        units=var_obj.attrs["units"])
                    elif var_sim not in list(f.dimensions.keys()) and ('x', 'y') == var_obj.dimensions:
                        v = f.create_variable(var_sim, ('x', 'y'), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        v[:, :] = vals[idx_best1, :]
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                        units=var_obj.attrs["units"])

        # select best 10 simulations
        click.echo('Write best 10 simulations ...')
        df_params_metrics1 = df_params_metrics.copy()
        df_params_metrics1.loc[:, 'id'] = range(len(df_params_metrics1.index))
        df_params_metrics1 = df_params_metrics1.sort_values(by=['KGE_multi'], ascending=False)
        idx_best10 = df_params_metrics1.loc[:df_params_metrics1.index[9], 'id'].values.tolist()
        # write states of best model run
        states_hm_mc_file = base_path / "states_hm_monte_carlo.nc"
        states_hm_file = base_path / "states_hm10.nc"
        with h5netcdf.File(states_hm_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='RoGeR best 10 Monte Carlo simulations at Rietholzbach lysimeter site',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment='First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).',
                model_structure='SVAT model with free drainage',
            )
            with h5netcdf.File(states_hm_mc_file, 'r', decode_vlen_strings=False) as df:
                f.attrs.update(
                    roger_version=df.attrs['roger_version']
                )
                # set dimensions with a dictionary
                dict_dim = {'x': len(idx_best10), 'y': 1, 'Time': len(df.variables['Time'])}
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
                    v = f.create_variable('Time', ('Time',), float, compression="gzip", compression_opts=1)
                    var_obj = df.variables.get('Time')
                    v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                    units=var_obj.attrs["units"])
                    v[:] = onp.array(var_obj)
                for var_sim in list(df.variables.keys()):
                    var_obj = df.variables.get(var_sim)
                    if var_sim not in list(f.dimensions.keys()) and ('x', 'y', 'Time') == var_obj.dimensions:
                        v = f.create_variable(var_sim, ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        v[:, :, :] = vals[idx_best10, :, :]
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                        units=var_obj.attrs["units"])
                    elif var_sim not in list(f.dimensions.keys()) and ('x', 'y') == var_obj.dimensions:
                        v = f.create_variable(var_sim, ('x', 'y'), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        v[:, :] = vals[idx_best10, :]
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                        units=var_obj.attrs["units"])

        # select best 100 simulations
        click.echo('Write best 100 simulations ...')
        df_params_metrics1 = df_params_metrics.copy()
        df_params_metrics1.loc[:, 'id'] = range(len(df_params_metrics1.index))
        df_params_metrics1 = df_params_metrics1.sort_values(by=['KGE_multi'], ascending=False)
        idx_best100 = df_params_metrics1.loc[:df_params_metrics1.index[99], 'id'].values.tolist()
        # write states of best model run
        states_hm_mc_file = base_path / "states_hm_monte_carlo.nc"
        states_hm_file = base_path / "states_hm100.nc"
        with h5netcdf.File(states_hm_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='RoGeR best 100 Monte Carlo simulations at Rietholzbach lysimeter site',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment='First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).',
                model_structure='SVAT model with free drainage',
            )
            with h5netcdf.File(states_hm_mc_file, 'r', decode_vlen_strings=False) as df:
                f.attrs.update(
                    roger_version=df.attrs['roger_version']
                )
                # set dimensions with a dictionary
                dict_dim = {'x': len(idx_best100), 'y': 1, 'Time': len(df.variables['Time'])}
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
                    v = f.create_variable('Time', ('Time',), float, compression="gzip", compression_opts=1)
                    var_obj = df.variables.get('Time')
                    v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                    units=var_obj.attrs["units"])
                    v[:] = onp.array(var_obj)
                for var_sim in list(df.variables.keys()):
                    var_obj = df.variables.get(var_sim)
                    if var_sim not in list(f.dimensions.keys()) and ('x', 'y', 'Time') == var_obj.dimensions:
                        v = f.create_variable(var_sim, ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        v[:, :, :] = vals[idx_best100, :, :]
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                        units=var_obj.attrs["units"])
                    elif var_sim not in list(f.dimensions.keys()) and ('x', 'y') == var_obj.dimensions:
                        v = f.create_variable(var_sim, ('x', 'y'), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        v[:, :] = vals[idx_best100, :]
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                        units=var_obj.attrs["units"])
    return


if __name__ == "__main__":
    main()
