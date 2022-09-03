from pathlib import Path
import os
import glob
import shutil
import datetime
import h5netcdf
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import click
import roger
import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs


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

    # merge results into single file
    if not os.path.exists(base_path / "states_tm_monte_carlo.nc"):
        tm_structures = ['complete-mixing', 'piston',
                         'advection-dispersion',
                         'time-variant advection-dispersion']
        for tm_structure in tm_structures:
            tms = tm_structure.replace(" ", "_")
            path = str(base_path / f"SVATTRANSPORT_{tms}.*.nc")
            diag_files = glob.glob(path)
            states_tm_file = base_path / "states_tm_monte_carlo.nc"
            with h5netcdf.File(states_tm_file, 'a', decode_vlen_strings=False) as f:
                if tm_structure not in list(f.groups.keys()):
                    f.create_group(tm_structure)
                f.attrs.update(
                    date_created=datetime.datetime.today().isoformat(),
                    title='RoGeR transport model Monte Carlo simulations at Rietholzbach lysimeter site',
                    institution='University of Freiburg, Chair of Hydrology',
                    references='',
                    comment='',
                    model_structure='SVAT transport model with free drainage',
                    roger_version=f'{roger.__version__}'
                )
                # collect dimensions
                for dfs in diag_files:
                    with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                        # set dimensions with a dictionary
                        if not dfs.split('/')[-1].split('.')[1] == 'constant':
                            dict_dim = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'Time': len(df.variables['Time']), 'ages': len(df.variables['ages']), 'nages': len(df.variables['nages']), 'n_sas_params': len(df.variables['n_sas_params'])}
                            time = onp.array(df.variables.get('Time'))
                for dfs in diag_files:
                    with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                        if not f.groups[tm_structure].dimensions:
                            f.groups[tm_structure].dimensions = dict_dim
                            v = f.groups[tm_structure].create_variable('x', ('x',), float)
                            v.attrs['long_name'] = 'model run'
                            v.attrs['units'] = ''
                            v[:] = onp.arange(dict_dim["x"])
                            v = f.groups[tm_structure].create_variable('y', ('y',), float)
                            v.attrs['long_name'] = ''
                            v.attrs['units'] = ''
                            v[:] = onp.arange(dict_dim["y"])
                            v = f.groups[tm_structure].create_variable('ages', ('ages',), float)
                            v.attrs['long_name'] = 'Water ages'
                            v.attrs['units'] = 'days'
                            v[:] = onp.arange(1, dict_dim["ages"]+1)
                            v = f.groups[tm_structure].create_variable('nages', ('nages',), float)
                            v.attrs['long_name'] = 'Water ages (cumulated)'
                            v.attrs['units'] = 'days'
                            v[:] = onp.arange(0, dict_dim["nages"])
                            v = f.groups[tm_structure].create_variable('n_sas_params', ('n_sas_params',), float)
                            v.attrs['long_name'] = 'Number of SAS parameters'
                            v.attrs['units'] = ''
                            v[:] = onp.arange(0, dict_dim["n_sas_params"])
                            v = f.groups[tm_structure].create_variable('Time', ('Time',), float)
                            var_obj = df.variables.get('Time')
                            v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                           units=var_obj.attrs["units"])
                            v[:] = time
                        for var_sim in list(df.variables.keys()):
                            var_obj = df.variables.get(var_sim)
                            if var_sim not in list(dict_dim.keys()) and ('Time', 'y', 'x') == var_obj.dimensions and var_obj.shape[0] > 2:
                                v = f.groups[tm_structure].create_variable(var_sim, ('x', 'y', 'Time'), float)
                                vals = onp.array(var_obj)
                                v[:, :, :] = vals.swapaxes(0, 2)
                                v.attrs.update(long_name=var_obj.attrs["long_name"],
                                               units=var_obj.attrs["units"])
                            elif var_sim not in list(dict_dim.keys()) and ('Time', 'y', 'x') == var_obj.dimensions and var_obj.shape[0] <= 2:
                                v = f.groups[tm_structure].create_variable(var_sim, ('x', 'y'), float)
                                vals = onp.array(var_obj)
                                v[:, :] = vals.swapaxes(0, 2)[:, :, 0]
                                v.attrs.update(long_name=var_obj.attrs["long_name"],
                                               units=var_obj.attrs["units"])
                            elif var_sim not in list(dict_dim.keys()) and ('Time', 'n_sas_params', 'y', 'x') == var_obj.dimensions:
                                v = f.groups[tm_structure].create_variable(var_sim, ('x', 'y', 'n_sas_params'), float)
                                vals = onp.array(var_obj)
                                vals = vals.swapaxes(0, 3)
                                vals = vals.swapaxes(1, 2)
                                v[:, :, :] = vals[:, :, :, 0]
                                v.attrs.update(long_name=var_obj.attrs["long_name"],
                                               units=var_obj.attrs["units"])
                            elif var_sim not in list(dict_dim.keys()) and ('Time', 'ages', 'y', 'x') == var_obj.dimensions:
                                v = f.groups[tm_structure].create_variable(var_sim, ('x', 'y', 'Time', 'ages'), float)
                                vals = onp.array(var_obj)
                                vals = vals.swapaxes(0, 3)
                                vals = vals.swapaxes(1, 2)
                                vals = vals.swapaxes(2, 3)
                                v[:, :, :, :] = vals
                                v.attrs.update(long_name=var_obj.attrs["long_name"],
                                               units=var_obj.attrs["units"])
                            elif var_sim not in list(dict_dim.keys()) and ('Time', 'nages', 'y', 'x') == var_obj.dimensions:
                                v = f.groups[tm_structure].create_variable(var_sim, ('x', 'y', 'Time', 'nages'), float)
                                vals = onp.array(var_obj)
                                vals = vals.swapaxes(0, 3)
                                vals = vals.swapaxes(1, 2)
                                vals = vals.swapaxes(2, 3)
                                v[:, :, :, :] = vals
                                v.attrs.update(long_name=var_obj.attrs["long_name"],
                                               units=var_obj.attrs["units"])

    # load hydrologic simulation
    states_hm_file = base_path / "states_hm.nc"
    ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")

    # load observations (measured data)
    path_obs = base_path.parent / "observations" / "rietholzbach_lysimeter.nc"
    ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")
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
    theta_p33 = df_thetap.loc[:, f'theta_avg{window}'].quantile(0.33)
    theta_p66 = df_thetap.loc[:, f'theta_avg{window}'].quantile(0.66)
    cond1 = (df_thetap[f'theta_avg{window}'] < theta_p33)
    cond2 = (df_thetap[f'theta_avg{window}'] >= theta_p33) & (df_thetap[f'theta_avg{window}'] < theta_p66)
    cond3 = (df_thetap[f'theta_avg{window}'] >= theta_p66)
    df_thetap.loc[cond1, 'sc'] = 1  # dry
    df_thetap.loc[cond2, 'sc'] = 2  # normal
    df_thetap.loc[cond3, 'sc'] = 3  # wet

    dict_params_eff = {}
    tm_structures = ['complete-mixing', 'piston',
                     'advection-dispersion',
                     'time-variant advection-dispersion']
    for tm_structure in tm_structures:
        tms = tm_structure.replace(" ", "_")

        # load transport simulation
        states_tm_file = base_path / "states_tm_monte_carlo.nc"
        ds_sim_tm = xr.open_dataset(states_tm_file, group=tm_structure, engine="h5netcdf")

        # assign date
        days_sim_hm = (ds_sim_hm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date_sim_hm = num2date(days_sim_hm, units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_sim_hm = ds_sim_hm.assign_coords(date=("Time", date_sim_hm))
        ds_sim_tm = ds_sim_tm.assign_coords(date=("Time", date_sim_tm))
        ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

        # DataFrame with sampled model parameters and the corresponding metrics
        nx = ds_sim_tm.dims['x']  # number of rows
        ny = ds_sim_tm.dims['y']  # number of columns
        df_params_eff = pd.DataFrame(index=range(nx * ny))
        # sampled model parameters
        if tm_structure == "preferential":
            df_params_eff.loc[:, 'b_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=2).values.flatten()
            df_params_eff.loc[:, 'b_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=2).values.flatten()
            df_params_eff.loc[:, 'b_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=2).values.flatten()
        elif tm_structure == "advection-dispersion":
            df_params_eff.loc[:, 'b_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=2).values.flatten()
            df_params_eff.loc[:, 'a_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=1).values.flatten()
            df_params_eff.loc[:, 'a_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=1).values.flatten()
        elif tm_structure == "complete-mixing advection-dispersion":
            df_params_eff.loc[:, 'a_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=1).values.flatten()
            df_params_eff.loc[:, 'a_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=1).values.flatten()
        elif tm_structure == "time-variant advection-dispersion":
            df_params_eff.loc[:, 'b_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=4).values.flatten()
            df_params_eff.loc[:, 'a_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=4).values.flatten()
            df_params_eff.loc[:, 'a_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=4).values.flatten()
        elif tm_structure == "time-variant preferential":
            df_params_eff.loc[:, 'b_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=4).values.flatten()
            df_params_eff.loc[:, 'b_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=4).values.flatten()
            df_params_eff.loc[:, 'b_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=4).values.flatten()
        elif tm_structure == "time-variant":
            df_params_eff.loc[:, 'ab_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=4).values.flatten()
            df_params_eff.loc[:, 'ab_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=4).values.flatten()
            df_params_eff.loc[:, 'ab_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=4).values.flatten()

        # compare observations and simulations
        ncol = 0
        idx = ds_sim_tm.date.values  # time index
        d18O_perc_bs = onp.zeros((nx, 1, len(idx)))
        df_idx_bs = pd.DataFrame(index=date_obs, columns=['sol'])
        df_idx_bs.loc[:, 'sol'] = ds_obs['d18O_PERC'].isel(x=0, y=0).values
        idx_bs = df_idx_bs['sol'].dropna().index
        for nrow in range(nx):
            # calculate simulated oxygen-18 bulk sample
            df_perc_18O_obs = pd.DataFrame(index=date_obs, columns=['perc_obs', 'd18O_perc_obs'])
            df_perc_18O_obs.loc[:, 'perc_obs'] = ds_obs['PERC'].isel(x=0, y=0).values
            df_perc_18O_obs.loc[:, 'd18O_perc_obs'] = ds_obs['d18O_PERC'].isel(x=0, y=0).values
            sample_no = pd.DataFrame(index=idx_bs, columns=['sample_no'])
            sample_no = sample_no.loc['1997':'2007']
            sample_no['sample_no'] = range(len(sample_no.index))
            df_perc_18O_sim = pd.DataFrame(index=date_sim_tm, columns=['perc_sim', 'd18O_perc_sim'])
            df_perc_18O_sim['perc_sim'] = ds_sim_hm['q_ss'].isel(x=0, y=0).values
            df_perc_18O_sim['d18O_perc_sim'] = ds_sim_tm['C_q_ss'].isel(x=nrow, y=ncol).values
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
            df_perc_18O_sim.loc[:, 'd18O_sample'] = df_perc_18O_sim.loc[:, 'd18O_sample'].fillna(method='bfill', limit=14)
            cond = (df_perc_18O_sim['d18O_sample'] == 0)
            df_perc_18O_sim.loc[cond, 'd18O_sample'] = onp.NaN
            d18O_perc_bs[nrow, ncol, :] = df_perc_18O_sim.loc[:, 'd18O_sample'].values
            # calculate observed oxygen-18 bulk sample
            df_perc_18O_obs.loc[:, 'd18O_perc_bs'] = df_perc_18O_obs['d18O_perc_obs'].fillna(method='bfill', limit=14)

            perc_sample_sum_obs = df_perc_18O_sim.join(df_perc_18O_obs).groupby(['sample_no']).sum().loc[:, 'perc_obs']
            sample_no['perc_obs_sum'] = perc_sample_sum_obs.values
            df_perc_18O_sim = df_perc_18O_sim.join(sample_no['perc_obs_sum'])
            df_perc_18O_sim.loc[:, 'perc_obs_sum'] = df_perc_18O_sim.loc[:, 'perc_obs_sum'].fillna(method='bfill', limit=14)

            # join observations on simulations
            for sc, sc1 in zip([0, 1, 2, 3], ['', 'dry', 'normal', 'wet']):
                obs_vals = ds_obs['d18O_PERC'].isel(x=0, y=0).values
                sim_vals = d18O_perc_bs[nrow, ncol, :]
                df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
                df_obs.loc[:, 'obs'] = obs_vals
                df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)

                if sc > 0:
                    df_rows = pd.DataFrame(index=df_eval.index).join(df_thetap)
                    rows = (df_rows['sc'].values == sc)
                    df_eval = df_eval.loc[rows, :]
                df_eval = df_eval.dropna()
                # calculate metrics
                var_sim = 'C_q_ss'
                obs_vals = df_eval.loc[:, 'obs'].values
                sim_vals = df_eval.loc[:, 'sim'].values
                key_kge = f'KGE_{var_sim}{sc1}'
                df_params_eff.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals, sim_vals)
                key_kge_alpha = f'KGE_alpha_{var_sim}{sc1}'
                df_params_eff.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals, sim_vals)
                key_kge_beta = f'KGE_beta_{var_sim}{sc1}'
                df_params_eff.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals, sim_vals)
                key_r = f'r_{var_sim}{sc1}'
                df_params_eff.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals, sim_vals)

            # avoid defragmentation of DataFrame
            df_params_eff = df_params_eff.copy()

        # write bulk sample to output file
        ds_sim_tm = ds_sim_tm.close()
        states_tm_file = base_path / "states_tm_monte_carlo.nc"
        with h5netcdf.File(states_tm_file, 'a', decode_vlen_strings=False) as f:
            try:
                v = f.groups[tm_structure].create_variable('d18O_perc_bs', ('x', 'y', 'Time'), float)
            except ValueError:
                v = f.groups[tm_structure].get('d18O_perc_bs')
            v[:, :, :] = d18O_perc_bs
            v.attrs.update(long_name="bulk sample of d18O in percolation",
                           units="permil")

        # write to .txt
        file = base_path_results / f"params_eff_{tm_structure}.txt"
        df_params_eff.to_bsv(file, header=True, index=False, sep="\t")
        dict_params_eff[tm_structure] = {}
        dict_params_eff[tm_structure]['params_eff'] = df_params_eff

        # dotty plots
        if tm_structure in ['preferential', 'advection-dispersion',
                            'time-variant preferential',
                            'time-variant advection-dispersion']:
            df_eff = df_params_eff.loc[:, ['KGE_C_q_ss']]
            if tm_structure == "preferential":
                df_params = df_params_eff.loc[:, ['b_transp', 'b_q_rz', 'b_q_ss']]
            elif tm_structure == "advection-dispersion":
                df_params = df_params_eff.loc[:, ['b_transp', 'a_q_rz', 'a_q_ss']]
            elif tm_structure == "complete-mixing advection-dispersion":
                df_params = df_params_eff.loc[:, ['a_q_rz', 'a_q_ss']]
            elif tm_structure == "time-variant advection-dispersion":
                df_params = df_params_eff.loc[:, ['b_transp', 'a_q_rz', 'a_q_ss']]
            elif tm_structure == "time-variant preferential":
                df_params = df_params_eff.loc[:, ['b_transp', 'b_q_rz', 'b_q_ss']]
            elif tm_structure == "time-variant":
                df_params = df_params_eff.loc[:, ['ab_transp', 'ab_q_rz', 'ab_q_ss']]
            nrow = len(df_eff.columns)
            ncol = len(df_params.columns)
            fig, ax1 = plt.subplots(nrow, ncol, sharey=True, figsize=(14, 7))
            ax = ax1.reshape(nrow, ncol)
            for i in range(nrow):
                for j in range(ncol):
                    y = df_eff.iloc[:, i]
                    x = df_params.iloc[:, j]
                    sns.regplot(x=x, y=y, ax=ax[i, j], ci=None, color='k',
                                scatter_kws={'alpha': 0.2, 's': 4, 'color': 'grey'})
                    ax[i, j].set_xlabel('')
                    ax[i, j].set_ylabel('')

            for j in range(ncol):
                xlabel = labs._LABS[df_params.columns[j]]
                ax[-1, j].set_xlabel(xlabel)

            ax[0, 0].set_ylabel(r'$KGE_{d_{18}O_{PERC}}$ [-]')

            fig.subplots_adjust(wspace=0.2, hspace=0.3)
            file = base_path_figs / f"dotty_plots_{tm_structure}.png"
            fig.savefig(file, dpi=250)

        # select best model run
        idx_best = df_params_eff['KGE_C_q_ss'].idxmax()
        dict_params_eff[tm_structure]['idx_best'] = idx_best

        # write SAS parameters of best model run
        params_tm_file = Path(__file__).parent / "sas_params.nc"
        with h5netcdf.File(params_tm_file, 'a', decode_vlen_strings=False) as f:
            if tm_structure not in list(f.groups.keys()):
                f.create_group(tm_structure)
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='RoGeR SAS parameters of best monte carlo simulation at Rietholzbach Lysimeter site',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                model_structure='SVAT transport model with free drainage',
                roger_version=f'{roger.__version__}'
            )
            dict_dim = {'x': nx, 'y': 1, 'n_sas_params': 8}
            if not f.groups[tm_structure].dimensions:
                f.groups[tm_structure].dimensions = dict_dim
                v = f.groups[tm_structure].create_variable('x', ('x',), float)
                v.attrs['long_name'] = 'Zonal coordinate'
                v.attrs['units'] = 'meters'
                v[:] = onp.arange(dict_dim["x"])
                v = f.groups[tm_structure].create_variable('y', ('y',), float)
                v.attrs['long_name'] = 'Meridonial coordinate'
                v.attrs['units'] = 'meters'
                v[:] = onp.arange(dict_dim["y"])
                v = f.groups[tm_structure].create_variable('n_sas_params', ('n_sas_params',), float)
                v.attrs['long_name'] = 'Number of SAS parameters'
                v.attrs['units'] = ' '
                v[:] = onp.arange(dict_dim["n_sas_params"])

            if tm_structure in ['preferential', 'advection-dispersion',
                                'time-variant preferential',
                                'time-variant advection-dispersion',
                                'time-variant']:
                try:
                    v = f.groups[tm_structure].create_variable('sas_params_transp', ('x', 'y', 'n_sas_params'), float)
                except ValueError:
                    v = f.groups[tm_structure].get('sas_params_transp')
                v[:, :, :] = ds_sim_tm["sas_params_transp"].isel(x=idx_best)
                v.attrs.update(long_name="SAS parameters of transpiration",
                               units=" ")
            try:
                v = f.groups[tm_structure].create_variable('sas_params_q_rz', ('x', 'y', 'n_sas_params'), float)
            except ValueError:
                v = f.groups[tm_structure].get('sas_params_q_rz')
            v[:, :, :] = ds_sim_tm["sas_params_q_rz"].isel(x=idx_best)
            v.attrs.update(long_name="SAS parameters of root zone percolation",
                           units=" ")
            try:
                v = f.groups[tm_structure].create_variable('sas_params_q_ss', ('x', 'y', 'n_sas_params'), float)
            except ValueError:
                v = f.groups[tm_structure].get('sas_params_q_ss')
            v[:, :, :] = ds_sim_tm["sas_params_q_ss"].isel(x=idx_best)
            v.attrs.update(long_name="SAS parameters of subsoil percolation",
                           units=" ")

        # write bulk sample to output file
        ds_sim_tm = ds_sim_tm.close()
        states_tm_file = base_path / "states_tm_monte_carlo.nc"
        with h5netcdf.File(states_tm_file, 'r+', decode_vlen_strings=False) as f:
            try:
                v = f.groups[tm_structure].create_variable('d18O_perc_bs', ('x', 'y', 'Time'), float)
            except ValueError:
                v = f.groups[tm_structure].get('d18O_perc_bs')
            v[:, :, :] = d18O_perc_bs
            v.attrs.update(long_name="bulk sample of d18O in percolation",
                           units="permil")

        # write states of best model run
        states_tm_mc_file = base_path / "states_tm_monte_carlo.nc"
        states_tm_file = base_path / "states_tm.nc"
        with h5netcdf.File(states_tm_file, 'a', decode_vlen_strings=False) as f:
            if tm_structure not in list(f.groups.keys()):
                f.create_group(tm_structure)
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='RoGeR best transport model Monte Carlo simulations at Rietholzbach lysimeter site',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment='',
                model_structure='SVAT transport model with free drainage',
                roger_version=f'{roger.__version__}'
            )
            # collect dimensions
            with h5netcdf.File(states_tm_mc_file, 'r', decode_vlen_strings=False) as df:
                # set dimensions with a dictionary
                if not dfs.split('/')[-1].split('.')[1] == 'constant':
                    dict_dim = {'x': 1, 'y': 1, 'Time': len(df.variables['Time']), 'ages': len(df.variables['ages']), 'nages': len(df.variables['nages']), 'n_sas_params': len(df.variables['n_sas_params'])}
                    time = onp.array(df.variables.get('Time'))
            for dfs in diag_files:
                with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                    if not f.groups[tm_structure].dimensions:
                        f.groups[tm_structure].dimensions = dict_dim
                        v = f.groups[tm_structure].create_variable('x', ('x',), float)
                        v.attrs['long_name'] = 'model run'
                        v.attrs['units'] = ''
                        v[:] = onp.arange(dict_dim["x"])
                        v = f.groups[tm_structure].create_variable('y', ('y',), float)
                        v.attrs['long_name'] = ''
                        v.attrs['units'] = ''
                        v[:] = onp.arange(dict_dim["y"])
                        v = f.groups[tm_structure].create_variable('ages', ('ages',), float)
                        v.attrs['long_name'] = 'Water ages'
                        v.attrs['units'] = 'days'
                        v[:] = onp.arange(1, dict_dim["ages"]+1)
                        v = f.groups[tm_structure].create_variable('nages', ('nages',), float)
                        v.attrs['long_name'] = 'Water ages (cumulated)'
                        v.attrs['units'] = 'days'
                        v[:] = onp.arange(0, dict_dim["nages"])
                        v = f.groups[tm_structure].create_variable('n_sas_params', ('n_sas_params',), float)
                        v.attrs['long_name'] = 'Number of SAS parameters'
                        v.attrs['units'] = ''
                        v[:] = onp.arange(0, dict_dim["n_sas_params"])
                        v = f.groups[tm_structure].create_variable('Time', ('Time',), float)
                        var_obj = df.variables.get('Time')
                        v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                       units=var_obj.attrs["units"])
                        v[:] = time
                    for var_sim in list(df.variables.keys()):
                        var_obj = df.variables.get(var_sim)
                        if var_sim not in list(dict_dim.keys()) and ('x', 'y', 'Time') == var_obj.dimensions:
                            v = f.groups[tm_structure].create_variable(var_sim, ('x', 'y', 'Time'), float)
                            vals = onp.array(var_obj)
                            v[:, :, :] = vals[idx_best, :, :]
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
                        elif var_sim not in list(dict_dim.keys()) and ('x', 'y') == var_obj.dimensions:
                            v = f.groups[tm_structure].create_variable(var_sim, ('x', 'y'), float)
                            vals = onp.array(var_obj)
                            v[:, :] = vals[idx_best, :]
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
                        elif var_sim not in list(dict_dim.keys()) and ('x', 'y', 'n_sas_params') == var_obj.dimensions:
                            v = f.groups[tm_structure].create_variable(var_sim, ('x', 'y', 'n_sas_params'), float)
                            vals = onp.array(var_obj)
                            v[:, :, :] = vals[idx_best, :, :]
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
                        elif var_sim not in list(dict_dim.keys()) and ('x', 'y', 'Time', 'ages') == var_obj.dimensions:
                            v = f.groups[tm_structure].create_variable(var_sim, ('x', 'y', 'Time', 'ages'), float)
                            vals = onp.array(var_obj)
                            v[:, :, :, :] = vals[idx_best, :, :, :]
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
                        elif var_sim not in list(dict_dim.keys()) and ('x', 'y', 'Time', 'nages') == var_obj.dimensions:
                            v = f.groups[tm_structure].create_variable(var_sim, ('x', 'y', 'Time', 'nages'), float)
                            vals = onp.array(var_obj)
                            v[:, :, :, :] = vals[idx_best, :, :, :]
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])

    # move best SAS parameters to directories of transport model
    base_path_tm = Path(__file__).parent / "svat_transport_bromide_benchmark"
    params_tm_file1 = base_path_tm / "sas_params.nc"
    shutil.copy(states_hm_file, params_tm_file1)

    base_path_tm = Path(__file__).parent / "svat_transport_monte_carlo_reverse"
    params_tm_file1 = base_path_tm / "sas_params.nc"
    shutil.copy(states_hm_file, params_tm_file1)

    base_path_tm = Path(__file__).parent / "svat_transport_sensitivity_reverse"
    params_tm_file1 = base_path_tm / "sas_params.nc"
    shutil.copy(states_hm_file, params_tm_file1)

    # duration curve of 18O in percolation
    return


if __name__ == "__main__":
    main()
