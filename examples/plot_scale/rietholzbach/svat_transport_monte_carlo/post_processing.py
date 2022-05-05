from pathlib import Path
import glob
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
path_obs = Path("/Users/robinschwemmle/Desktop/PhD/data/plot/rietholzbach/rietholzbach_lysimeter.nc")
ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")


dict_params_eff = {}
tm_structures = ['preferential', 'advection-dispersion',
                 'complete-mixing advection-dispersion',
                 'time-variant preferential',
                 'time-variant advection-dispersion']
for tm_structure in tm_structures:
    tms = tm_structure.replace(" ", "_")

    # load transport simulation
    states_tm_file = base_path / f"states_tm_{tms}_monte_carlo.nc"
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
        df_params_eff.loc[:, 'a_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=4).values.flatten()
        df_params_eff.loc[:, 'a_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=4).values.flatten()
    elif tm_structure == "time-variant":
        df_params_eff.loc[:, 'b_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=4).values.flatten()
        df_params_eff.loc[:, 'a_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=4).values.flatten()
        df_params_eff.loc[:, 'a_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=4).values.flatten()

    # compare observations and simulations
    ncol = 0
    idx = ds_sim_tm.Time  # time index
    for nrow in range(nx):
        # calculate simulated oxygen-18 composite sample
        df_perc_18O_obs = pd.DataFrame(index=idx, columns=['perc_obs', 'd18O_perc_obs'])
        df_perc_18O_obs.loc[:, 'perc_obs'] = ds_obs['PERC'].isel(x=nrow, y=ncol).values
        df_perc_18O_obs.loc[:, 'd18O_perc_obs'] = ds_obs['d18O_perc'].isel(x=nrow, y=ncol).values
        sample_no = pd.DataFrame(index=df_perc_18O_obs.dropna().index, columns=['sample_no'])
        sample_no = sample_no.loc['1997':'2007']
        sample_no['sample_no'] = range(len(sample_no.index))
        df_perc_18O_sim = pd.DataFrame(index=idx, columns=['perc_sim', 'd18O_perc_sim'])
        df_perc_18O_sim['perc_sim'] = ds_sim_hm['q_ss'].isel(x=nrow, y=ncol).values
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
        d18O_perc_cs = onp.zeros((1, 1, len(idx)))
        d18O_perc_cs[nrow, ncol, :] = df_perc_18O_sim.loc[:, 'd18O_sample'].values
        ds_sim_tm.assign(d18O_perc_cs=d18O_perc_cs)
        # calculate observed oxygen-18 composite sample
        df_perc_18O_obs.loc[:, 'd18O_perc_cs'] = df_perc_18O_obs['d18O_perc'].fillna(method='bfill', limit=14)

        perc_sample_sum_obs = df_perc_18O_sim.join(df_perc_18O_obs).groupby(['sample_no']).sum().loc[:, 'perc_obs']
        sample_no['perc_obs_sum'] = perc_sample_sum_obs.values
        df_perc_18O_sim = df_perc_18O_sim.join(sample_no['perc_obs_sum'])
        df_perc_18O_sim.loc[:, 'perc_obs_sum'] = df_perc_18O_sim.loc[:, 'perc_obs_sum'].fillna(method='bfill', limit=14)

        # join observations on simulations
        obs_vals = ds_obs['d18O_perc'].isel(x=nrow, y=ncol).values
        sim_vals = ds_sim_tm['d18O_perc_cs'].isel(x=nrow, y=ncol).values
        df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
        df_obs.loc[:, 'obs'] = ds_obs['d18O_perc'].isel(x=nrow, y=ncol).values
        df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
        df_eval = df_eval.dropna()

        # calculate metrics
        var_sim = 'C_q_ss'
        obs_vals = df_eval.loc[:, 'obs'].values
        sim_vals = df_eval.loc[:, 'sim'].values
        key_kge = 'KGE_' + var_sim
        df_params_eff.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals, sim_vals)
        key_kge_alpha = 'KGE_alpha_' + var_sim
        df_params_eff.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals, sim_vals)
        key_kge_beta = 'KGE_beta_' + var_sim
        df_params_eff.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals, sim_vals)
        key_r = 'r_' + var_sim
        df_params_eff.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals, sim_vals)

    # write to .txt
    file = base_path_results / f"params_eff_{tm_structure}.txt"
    df_params_eff.to_csv(file, header=True, index=False, sep="\t")
    dict_params_eff[tm_structure] = df_params_eff

    # select best model run
    idx_best = df_params_eff['KGE_C_q_ss'].idxmax() + 2

    # write SAS parameters of best model run
    params_tm_file = base_path / f"sas_params_{tm_structure}.nc"
    with h5netcdf.File(params_tm_file, 'w', decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title=f'RoGeR SAS parameters of best monte carlo run of {tm_structure} transport model at Rietholzbach Lysimeter site',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment=f'SVAT {tm_structure} transport model with free drainage'
        )
        f.dimensions = {'x': nx, 'y': 1, 'n_sas_params': 8}
        v = f.create_variable('x', ('x',), float)
        v.attrs['long_name'] = 'Zonal coordinate'
        v.attrs['units'] = 'meters'
        v[:] = onp.arange(f.dimensions["x"])
        v = f.create_variable('y', ('y',), float)
        v.attrs['long_name'] = 'Meridonial coordinate'
        v.attrs['units'] = 'meters'
        v[:] = onp.arange(f.dimensions["y"])
        v = f.create_variable('n_sas_params', ('n_sas_params',), float)
        v.attrs['long_name'] = 'Number of SAS parameters'
        v.attrs['units'] = ' '
        v[:] = onp.arange(f.dimensions["n_sas_params"])

        if tm_structure in ['preferential', 'advection-dispersion',
                            'time-variant preferential',
                            'time-variant advection-dispersion',
                            'time-variant']:
            v = f.create_variable('sas_params_transp', ('x', 'y', 'n_sas_params'), float)
            v[:, :, :] = ds_sim_tm["sas_params_transp"].isel(x=idx_best)
            v.attrs.update(long_name="SAS parameters of transpiration",
                           units=" ")

        v = f.create_variable('sas_params_q_rz', ('x', 'y', 'n_sas_params'), float)
        v[:, :, :] = ds_sim_tm["sas_params_q_rz"].isel(x=idx_best)
        v.attrs.update(long_name="SAS parameters of root zone percolation",
                       units=" ")

        v = f.create_variable('sas_params_q_ss', ('x', 'y', 'n_sas_params'), float)
        v[:, :, :] = ds_sim_tm["sas_params_q_ss"].isel(x=idx_best)
        v.attrs.update(long_name="SAS parameters of subsoil percolation",
                       units=" ")

    # move hydrologic states to directories of transport model
    base_path_tm = base_path.parent / "svat_transport_monte_carlo_reverse"
    params_tm_file1 = base_path_tm / f"sas_params_{tm_structure}.nc"
    shutil.copy(states_hm_file, params_tm_file1)

    base_path_tm = base_path.parent / "svat_transport_sensitivity_reverse"
    params_tm_file1 = base_path_tm / f"sas_params_{tm_structure}.nc"
    shutil.copy(states_hm_file, params_tm_file1)
