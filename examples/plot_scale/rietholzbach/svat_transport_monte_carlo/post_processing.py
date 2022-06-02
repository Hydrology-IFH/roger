from pathlib import Path
import os
import shutil
import datetime
import h5netcdf
import matplotlib.pyplot as plt
import xarray as xr
from cftime import num2date
import pandas as pd

import roger.tools.evaluation as eval_utils
import numpy as onp

base_path = Path(__file__).parent
# directory of results
base_path_results = base_path / "results"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)

# load hydrologic simulation
states_hm_file = base_path / "states_hm.nc"
ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")

# load observations (measured data)
path_obs = base_path / "observations" / "rietholzbach_lysimeter.nc"
ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")


dict_params_eff = {}
tm_structures = ['complete-mixing', 'piston',
                 'preferential', 'advection-dispersion',
                 'time-variant preferential',
                 'time-variant advection-dispersion']
for tm_structure in tm_structures:
    tms = tm_structure.replace(" ", "_")

    # load transport simulation
    states_tm_file = base_path / "states_tm_monte_carlo.nc"
    ds_sim_tm = xr.open_dataset(states_tm_file, group=f"{tm_structure}", engine="h5netcdf")

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
        df_params_eff.loc[:, 'a_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=4).values.flatten()
        df_params_eff.loc[:, 'a_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=4).values.flatten()
    elif tm_structure == "time-variant":
        df_params_eff.loc[:, 'b_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=4).values.flatten()
        df_params_eff.loc[:, 'a_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=4).values.flatten()
        df_params_eff.loc[:, 'a_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=4).values.flatten()

    # compare observations and simulations
    ncol = 0
    idx = ds_sim_tm.date.values  # time index
    d18O_perc_cs = onp.zeros((nx, 1, len(idx)))
    df_idx_cs = pd.DataFrame(index=date_obs, columns=['sol'])
    df_idx_cs.loc[:, 'sol'] = ds_obs['d18O_PERC'].isel(x=0, y=0).values
    idx_cs = df_idx_cs['sol'].dropna().index
    for nrow in range(nx):
        # calculate simulated oxygen-18 composite sample
        df_perc_18O_obs = pd.DataFrame(index=idx_cs, columns=['perc_obs', 'd18O_perc_obs'])
        df_perc_18O_obs.loc[:, 'perc_obs'] = ds_obs['PERC'].isel(x=0, y=0).values
        df_perc_18O_obs.loc[:, 'd18O_perc_obs'] = ds_obs['d18O_PERC'].isel(x=0, y=0).values
        sample_no = pd.DataFrame(index=df_perc_18O_obs.dropna().index, columns=['sample_no'])
        sample_no = sample_no.loc['1997':'2007']
        sample_no['sample_no'] = range(len(sample_no.index))
        df_perc_18O_sim = pd.DataFrame(index=idx, columns=['perc_sim', 'd18O_perc_sim'])
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
        d18O_perc_cs[nrow, ncol, :] = df_perc_18O_sim.loc[:, 'd18O_sample'].values
        # calculate observed oxygen-18 composite sample
        df_perc_18O_obs.loc[:, 'd18O_perc_cs'] = df_perc_18O_obs['d18O_perc_obs'].fillna(method='bfill', limit=14)

        perc_sample_sum_obs = df_perc_18O_sim.join(df_perc_18O_obs).groupby(['sample_no']).sum().loc[:, 'perc_obs']
        sample_no['perc_obs_sum'] = perc_sample_sum_obs.values
        df_perc_18O_sim = df_perc_18O_sim.join(sample_no['perc_obs_sum'])
        df_perc_18O_sim.loc[:, 'perc_obs_sum'] = df_perc_18O_sim.loc[:, 'perc_obs_sum'].fillna(method='bfill', limit=14)

        # join observations on simulations
        obs_vals = ds_obs['d18O_PERC'].isel(x=0, y=0).values
        sim_vals = d18O_perc_cs[nrow, ncol, :]
        df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
        df_obs.loc[:, 'obs'] = obs_vals
        df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
        df_eval = df_eval.dropna()

        # calculate metrics
        var_sim = 'C_q_ss'
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

    # # write composite sample to output file
    # states_tm_file = base_path / "states_tm_monte_carlo.nc"
    # with h5netcdf.File(states_tm_file, 'a', decode_vlen_strings=False) as f:
    #     v = f.groups[tm_structure].create_variable('d18O_perc_cs', ('x', 'y', 'Time'), float)
    #     v[:, :, :] = d18O_perc_cs
    #     v.attrs.update(long_name="composite sample of d18O in percolation",
    #                    units="permil")
    # write to .txt
    file = base_path_results / f"params_eff_{tm_structure}.txt"
    df_params_eff.to_csv(file, header=True, index=False, sep="\t")
    dict_params_eff[tm_structure] = df_params_eff

    # dotty plots
    df_eff = df_params_eff.loc[:, ['KGE_C_q_ss']]
    if tm_structure == "preferential":
        df_params = df_params_eff.loc[:, ['b_transp', 'b_q_rz', 'b_q_ss']]
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
        df_params_eff.loc[:, 'a_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=4).values.flatten()
        df_params_eff.loc[:, 'a_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=4).values.flatten()
    elif tm_structure == "time-variant":
        df_params_eff.loc[:, 'b_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=4).values.flatten()
        df_params_eff.loc[:, 'a_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=4).values.flatten()
        df_params_eff.loc[:, 'a_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=4).values.flatten()
    nrow = len(df_eff.columns)
    ncol = len(df_params.columns)
    fig, ax = plt.subplots(nrow, ncol, sharey=True, figsize=(14, 7))
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

    ax[0, 0].set_ylabel('$KGE_{ET}$ [-]')
    ax[1, 0].set_ylabel('$KGE_{PERC}$ [-]')
    ax[2, 0].set_ylabel(r'$r_{\Delta S}$ [-]')
    ax[3, 0].set_ylabel('$E_{multi}$\n [-]')

    fig.subplots_adjust(wspace=0.2, hspace=0.3)
    file = base_path_figs / "dotty_plots.png"
    fig.savefig(file, dpi=250)


    # select best model run
    idx_best = df_params_eff['KGE_C_q_ss'].idxmax()

    # write SAS parameters of best model run
    params_tm_file = base_path / "sas_params.nc"
    with h5netcdf.File(params_tm_file, 'a', decode_vlen_strings=False) as f:
        if tm_structure not in list(f.groups.keys()):
            f.create_group(tm_structure)
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title=f'RoGeR SAS parameters of best monte carlo run of {tm_structure} transport model at Rietholzbach Lysimeter site',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment=f'SVAT {tm_structure} transport model with free drainage'
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
            v = f.groups[tm_structure].create_variable('sas_params_transp', ('x', 'y', 'n_sas_params'), float)
            v[:, :, :] = ds_sim_tm["sas_params_transp"].isel(x=idx_best)
            v.attrs.update(long_name="SAS parameters of transpiration",
                           units=" ")

        v = f.groups[tm_structure].create_variable('sas_params_q_rz', ('x', 'y', 'n_sas_params'), float)
        v[:, :, :] = ds_sim_tm["sas_params_q_rz"].isel(x=idx_best)
        v.attrs.update(long_name="SAS parameters of root zone percolation",
                       units=" ")

        v = f.groups[tm_structure].create_variable('sas_params_q_ss', ('x', 'y', 'n_sas_params'), float)
        v[:, :, :] = ds_sim_tm["sas_params_q_ss"].isel(x=idx_best)
        v.attrs.update(long_name="SAS parameters of subsoil percolation",
                       units=" ")

# move best SAS parameters to directories of transport model
base_path_tm = base_path.parent / "svat_transport_bromide_benchmark"
params_tm_file1 = base_path_tm / "sas_params.nc"
shutil.copy(states_hm_file, params_tm_file1)

base_path_tm = base_path.parent / "svat_transport_monte_carlo_reverse"
params_tm_file1 = base_path_tm / "sas_params.nc"
shutil.copy(states_hm_file, params_tm_file1)

base_path_tm = base_path.parent / "svat_transport_sensitivity_reverse"
params_tm_file1 = base_path_tm / "sas_params.nc"
shutil.copy(states_hm_file, params_tm_file1)

# compare best model runs

# duration curve of 18O in percolation
