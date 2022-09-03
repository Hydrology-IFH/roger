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

import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs

base_path = Path(__file__).parent
# directory of results
base_path_results = base_path / "results"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# merge bromide model output into single file
lys_experiments = ["lys3_bromide", "lys8_bromide", "lys9_bromide"]
tm_structures = ['complete-mixing', 'piston',
                 'preferential', 'complete-mixing + advection-dispersion',
                 'time-variant preferential',
                 'time-variant complete-mixing + advection-dispersion']
for lys_experiment in lys_experiments:
    for tm_structure in tm_structures:
        tms = tm_structure.replace(" ", "_")
        path = str(base_path / f'SVATCROPTRANSPORT_{tms}_{lys_experiment}_bromide.*.nc')
        diag_files = glob.glob(path)
        states_tm_file = base_path / f"states_{tms}_{lys_experiment}_monte_carlo_bromide.nc"
        with h5netcdf.File(states_tm_file, 'a', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='RoGeR bromide transport model Monte Carlo simulations at Reckenholz lysimeter site',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment=f'SVAT {tm_structure} model with free drainage and crop phenology/crop rotation'
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
                    # set dimensions with a dictionary
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
                        with h5netcdf.File(base_path / 'forcing_tracer.nc', "r", decode_vlen_strings=False) as infile:
                            time_origin = infile.variables['time'].attrs['time_origin']
                        v.attrs.update(time_origin=time_origin,
                                       units=var_obj.attrs["units"])
                        v[:] = onp.array(var_obj)
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
                    for var_sim in list(df.variables.keys()):
                        var_obj = df.variables.get(var_sim)
                        if var_sim not in list(dict_dim.keys()) and ('Time', 'y', 'x') == var_obj.dimensions and var_obj.shape[0] > 2:
                            v = f.create_variable(var_sim, ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                            vals = onp.array(var_obj)
                            v[:, :, :] = vals.swapaxes(0, 2)
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
                        elif var_sim not in list(dict_dim.keys()) and ('Time', 'n_sas_params', 'y', 'x') == var_obj.dimensions and var_obj.shape[0] <= 2:
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
dict_params_eff = {}
for lys_experiment in lys_experiments:
    dict_params_eff[lys_experiment] = {}
    # load hydrologic simulation
    states_hm_file = base_path / "states_hm.nc"
    ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf", group=lys_experiment)

    # load observations (measured data)
    path_obs = base_path.parent / "observations" / "reckenholz_lysimeter.nc"
    ds_obs = xr.open_dataset(path_obs, engine="h5netcdf", group=lys_experiment)
    for tm_structure in tm_structures:
        tms = tm_structure.replace(" ", "_")

        # load transport simulation
        states_tm_file = base_path / f"states_{tms}_{lys}_monte_carlo_bromide.nc"
        ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")

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
        Br_perc_cs = onp.zeros((1, 1, len(idx)))
        Br_perc_mass_cs = onp.zeros((1, 1, len(idx)))
        df_idx_cs = pd.DataFrame(index=date_obs, columns=['sol'])
        df_idx_cs.loc[:, 'sol'] = ds_obs['BR_PERC'].isel(x=0, y=0).values
        idx_cs = df_idx_cs['sol'].dropna().index
        for nrow in range(nx):
            # calculate simulated bromide composite sample
            sample_no = pd.DataFrame(index=idx_cs, columns=['sample_no'])
            sample_no['sample_no'] = range(len(sample_no.index))
            df_perc_Br_sim = pd.DataFrame(index=idx, columns=['perc', 'Br'])
            df_perc_Br_sim['perc'] = ds_sim_hm['q_ss'].isel(x=0, y=0).values
            df_perc_Br_sim['Br'] = ds_sim_tm['C_q_ss'].isel(x=nrow, y=0).values
            df_perc_Br_sim = df_perc_Br_sim.join(sample_no)
            df_perc_Br_sim.loc[:, 'sample_no'] = df_perc_Br_sim.loc[:, 'sample_no'].fillna(method='bfill')
            perc_sum = df_perc_Br_sim.groupby(['sample_no']).sum().loc[:, 'perc']
            sample_no['perc_sum'] = perc_sum.values
            df_perc_Br_sim = df_perc_Br_sim.join(sample_no['perc_sum'])
            df_perc_Br_sim.loc[:, 'perc_sum'] = df_perc_Br_sim.loc[:, 'perc_sum'].fillna(method='bfill')
            df_perc_Br_sim['Br_weight'] = df_perc_Br_sim['Br'] * df_perc_Br_sim['perc']
            Br_sample = df_perc_Br_sim.groupby(['sample_no']).sum().loc[:, 'Br_weight']
            sample_no['Br_sample'] = Br_sample.values
            df_perc_Br_sim = df_perc_Br_sim.join(sample_no['Br_sample'])
            df_perc_Br_sim.loc[:, 'Br_sample'] = df_perc_Br_sim.loc[:, 'Br_sample']
            Br_perc_cs[nrow, ncol, :] = df_perc_Br_sim.loc[:, 'Br_sample'].values

            # calculate metrics
            vars_sim = ['Br_perc_cs', 'Br_perc_mass_cs']
            vars_obs = ['BR_PERC', 'BR_PERC_MASS']
            for var_sim, var_obs in zip(vars_sim, vars_obs):
                # join observations on simulations
                obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
                if var_sim == 'Br_perc_cs':
                    sim_vals = Br_perc_cs[nrow, ncol, :]
                elif var_sim == 'Br_perc_mass_cs':
                    sim_vals = Br_perc_mass_cs[nrow, ncol, :]
                df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
                df_obs.loc[:, 'obs'] = obs_vals
                df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
                df_eval = df_eval.dropna()
                obs_vals = df_eval.loc[:, 'obs'].values
                sim_vals = df_eval.loc[:, 'sim'].values
                key_kge = f'KGE_{var_sim}'
                df_params_eff.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals, sim_vals)
                key_kge_alpha = f'KGE_alpha_{var_sim}'
                df_params_eff.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals, sim_vals)
                key_kge_beta = f'KGE_beta_{var_sim}'
                df_params_eff.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals, sim_vals)
                key_r = f'r_{var_sim}'
                df_params_eff.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals, sim_vals)

        # write to .txt
        file = base_path_results / f"params_eff_{lys_experiment}_{tm_structure}.txt"
        df_params_eff.to_csv(file, header=True, index=False, sep="\t")
        dict_params_eff[lys_experiment][tm_structure] = {}
        dict_params_eff[lys_experiment][tm_structure]['params_eff'] = df_params_eff

        # dotty plots
        if tm_structure in ['preferential', 'advection-dispersion',
                            'time-variant preferential',
                            'time-variant advection-dispersion']:
            df_eff = df_params_eff.loc[:, ['KGE_Br_perc_cs']]
            if tm_structure == "preferential":
                df_params = df_params_eff.loc[:, ['b_transp', 'b_q_rz', 'b_q_ss', 'alpha_transp', 'alpha_q']]
            elif tm_structure == "advection-dispersion":
                df_params = df_params_eff.loc[:, ['b_transp', 'a_q_rz', 'a_q_ss', 'alpha_transp', 'alpha_q']]
            elif tm_structure == "complete-mixing advection-dispersion":
                df_params = df_params_eff.loc[:, ['a_q_rz', 'a_q_ss', 'alpha_transp', 'alpha_q']]
            elif tm_structure == "time-variant advection-dispersion":
                df_params = df_params_eff.loc[:, ['b_transp', 'a_q_rz', 'a_q_ss', 'alpha_transp', 'alpha_q']]
            elif tm_structure == "time-variant preferential":
                df_params = df_params_eff.loc[:, ['b_transp', 'b_q_rz', 'b_q_ss', 'alpha_transp', 'alpha_q']]
            elif tm_structure == "time-variant":
                df_params = df_params_eff.loc[:, ['ab_transp', 'ab_q_rz', 'ab_q_ss', 'alpha_transp', 'alpha_q']]
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

            ax[0, 0].set_ylabel(r'$KGE_{Br_{PERC}}$ [-]')

            fig.subplots_adjust(wspace=0.2, hspace=0.3)
            file = base_path_figs / "dotty_plots.png"
            fig.savefig(file, dpi=250)

        # select best model run
        idx_best = df_params_eff['KGE_C_q_ss'].idxmax()
        dict_params_eff[lys_experiment][tm_structure]['idx_best'] = idx_best

        # write transport model parameters of best model run
        params_tm_file = base_path / f"{tms}_bromide_params.nc"
        with h5netcdf.File(params_tm_file, 'a', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='RoGeR bromide transport model parameters of best monte carlo simulation at Reckenholz Lysimeter site',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment='SVAT transport model with free drainage and crop phenology/crop rotation'
            )
            dict_dim = {'x': nx, 'y': 1, 'n_sas_params': 8}
            if not f.dimensions:
                f.dimensions = dict_dim
                v = f.create_variable('x', ('x',), float, compression="gzip", compression_opts=1)
                v.attrs['long_name'] = 'Zonal coordinate'
                v.attrs['units'] = 'meters'
                v[:] = onp.arange(dict_dim["x"])
                v = f.create_variable('y', ('y',), float, compression="gzip", compression_opts=1)
                v.attrs['long_name'] = 'Meridonial coordinate'
                v.attrs['units'] = 'meters'
                v[:] = onp.arange(dict_dim["y"])
                v = f.create_variable('n_sas_params', ('n_sas_params',), float, compression="gzip", compression_opts=1)
                v.attrs['long_name'] = 'Number of SAS parameters'
                v.attrs['units'] = ' '
                v[:] = onp.arange(dict_dim["n_sas_params"])

            if tm_structure in ['preferential', 'advection-dispersion',
                                'time-variant preferential',
                                'time-variant advection-dispersion',
                                'time-variant']:
                try:
                    v = f.create_variable('sas_params_transp', ('x', 'y', 'n_sas_params'), float, compression="gzip", compression_opts=1)
                except ValueError:
                    v = f.get('sas_params_transp')
                v[:, :, :] = ds_sim_tm["sas_params_transp"].isel(x=idx_best)
                v.attrs.update(long_name="SAS parameters of transpiration",
                               units=" ")
                try:
                    v = f.create_variable('sas_params_q_rz', ('x', 'y', 'n_sas_params'), float, compression="gzip", compression_opts=1)
                except ValueError:
                    v = f.get('sas_params_q_rz')
                v[:, :, :] = ds_sim_tm["sas_params_q_rz"].isel(x=idx_best)
                v.attrs.update(long_name="SAS parameters of root zone percolation",
                               units=" ")
                try:
                    v = f.create_variable('sas_params_q_ss', ('x', 'y', 'n_sas_params'), float, compression="gzip", compression_opts=1)
                except ValueError:
                    v = f.get('sas_params_q_ss')
                v[:, :, :] = ds_sim_tm["sas_params_q_ss"].isel(x=idx_best)
                v.attrs.update(long_name="SAS parameters of subsoil percolation",
                               units=" ")
            try:
                v = f.create_variable('alpha_transp', ('x', 'y'), float, compression="gzip", compression_opts=1)
            except ValueError:
                v = f.get('alpha_transp')
            v[:, :] = ds_sim_tm["alpha_transp"].isel(x=idx_best)
            v.attrs.update(long_name="Partition coefficient of transpiration",
                           units="-")
            try:
                v = f.create_variable('alpha_q', ('x', 'y'), float, compression="gzip", compression_opts=1)
            except ValueError:
                v = f.get('alpha_q')
            v[:, :] = ds_sim_tm["alpha_q"].isel(x=idx_best)
            v.attrs.update(long_name="Partition coefficient of flow processes",
                           units="-")

        # write composite sample to output file
        ds_sim_tm = ds_sim_tm.close()
        states_tm_file = base_path / "states_tm_monte_carlo_bromide.nc"
        with h5netcdf.File(states_tm_file, 'r+', decode_vlen_strings=False) as f:
            try:
                v = f.create_variable('Br_perc_cs', ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
            except ValueError:
                v = f.get('Br_perc_cs')
            v[:, :, :] = Br_perc_cs
            v.attrs.update(long_name="composite sample of bromide concentration in percolation",
                           units="mg/l")
            try:
                v = f.create_variable('Br_perc_mass_cs', ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
            except ValueError:
                v = f.get('Br_perc_mass_cs')
            v[:, :, :] = Br_perc_mass_cs
            v.attrs.update(long_name="composite sample of bromide mass in percolation",
                           units="mg")

# move best SAS parameters to directories of transport model
base_path_tm = base_path.parent / "svat_transport_monte_carlo_reverse"
params_tm_file1 = base_path_tm / "tm_bromide_params.nc"
shutil.copy(states_hm_file, params_tm_file1)

base_path_tm = base_path.parent / "svat_transport_sensitivity_reverse"
params_tm_file1 = base_path_tm / "tm_bromide_params.nc"
shutil.copy(states_hm_file, params_tm_file1)

# compare best model runs
fig, ax = plt.subplots(2, 3, sharey=True, figsize=(14, 7))
for i, tm_structure in enumerate(tm_structures):
    # load transport simulation
    states_tm_file = base_path / "states_tm_monte_carlo_bromide.nc"
    ds_sim_tm = xr.open_dataset(states_tm_file, group=tm_structure, engine="h5netcdf")
    # join observations on simulations
    obs_vals = ds_obs['BR_PERC'].isel(x=0, y=0).values
    idx_best = dict_params_eff[tm_structure]['idx_best']
    sim_vals = ds_sim_tm['Br_perc_cs'].isel(x=idx_best, y=0).values
    df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
    df_obs.loc[:, 'obs'] = obs_vals
    df_sim = pd.DataFrame(index=date_sim_tm, columns=['sim'])
    df_sim.loc[:, 'sim'] = ds_sim_tm['C_q_ss'].isel(x=idx_best, y=0).values
    df_sim = df_sim.iloc[1:, :]
    df_eval = eval_utils.join_obs_on_sim(date_sim_tm, sim_vals, df_obs)
    df_eval = df_eval.dropna()
    ax.flatten()[i].plot(df_sim.index, df_sim.iloc[:, 0], color='red')
    ax.flatten()[i].scatter(df_eval.index, df_eval.iloc[:, 0], color='red', s=2)
    ax.flatten()[i].scatter(df_eval.index, df_eval.iloc[:, 1], color='blue', s=2)

ax[0, 0].set_ylabel(r'$Br_{PERC}$ [mg/l]')
ax[1, 0].set_ylabel(r'$Br_{PERC}$ [mg/l]')
ax[1, 1].set_xlabel('Time')
