from pathlib import Path
import xarray as xr
import h5netcdf
from cftime import num2date
import pandas as pd
import numpy as onp

import roger.tools.evaluation as eval_utils

# base_path = Path(__file__).parent
base_path = Path("/Volumes/LaCie/roger/examples/plot_scale/reckenholz")

lys_experiments = ["lys2", "lys3", "lys8"]
tm_structures = ["time-variant_advection-dispersion-power"]

for lys_experiment in lys_experiments:
    # load hydrologic simulation
    states_hm_file = base_path / "output" / "svat_crop_monte_carlo_crop-specific" / f"SVATCROP_{lys_experiment}_bootstrap.nc"
    ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")

    # assign date
    days_sim_hm = (ds_sim_hm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim_hm = num2date(days_sim_hm, units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_hm = ds_sim_hm.assign_coords(date=("Time", date_sim_hm))

    # load observations (measured data)
    path_obs = Path(__file__).parent.parent / "observations" / "reckenholz_lysimeter.nc"
    ds_obs = xr.open_dataset(path_obs, engine="h5netcdf", group=lys_experiment)

    # assign date
    days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

    for tm_structure in tm_structures:
        tms = tm_structure.replace(" ", "_")

        # load transport simulation
        states_tm_file = base_path / "output" / "svat_crop_nitrate_monte_carlo_crop-specific" / f"SVATCROPNITRATE_{tms}_{lys_experiment}.nc"
        ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")

        # assign date
        days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_sim_tm = ds_sim_tm.assign_coords(date=("Time", date_sim_tm))

        # DataFrame with sampled model parameters and the corresponding metrics
        csv_file = Path(__file__).parent / f"parameters_for_{tm_structure}_{lys_experiment}.csv"
        df_params_metrics = pd.read_csv(csv_file, sep=";", skiprows=1)

        crop_types_sim = onp.unique(ds_sim_hm['lu_id'].values.flatten()).tolist()

        # compare observations and simulations
        nx = len(df_params_metrics.index)
        idx = ds_sim_tm.date.values  # time index
        df_idx_bs = pd.DataFrame(index=date_obs, columns=['sol'])
        df_idx_bs.loc[:, 'sol'] = ds_obs['NO3_PERC'].isel(x=0, y=0).values
        idx_bs = df_idx_bs['sol'].dropna().index
        perc_bs_sim = onp.zeros((nx, 1, len(idx)))
        perc_bs_obs = onp.zeros((nx, 1, len(idx)))
        NO3_perc_bs = onp.zeros((nx, 1, len(idx)))
        NO3_perc_mass_bs = onp.zeros((nx, 1, len(idx)))
        for nrow in range(nx):
            sim_vals = ds_sim_tm['M_q_ss'].isel(x=nrow, y=0).values
            cond = onp.isfinite(sim_vals)
            if cond.all():
                # calculate simulated nitrate bulk samples
                sample_no = pd.DataFrame(index=idx_bs, columns=['sample_no'])
                sample_no['sample_no'] = range(len(sample_no.index))
                df_perc_NO3_sim = pd.DataFrame(index=idx, columns=['perc', 'NO3_mass'])
                df_perc_NO3_sim['perc'] = ds_sim_hm['q_ss'].isel(x=nrow, y=0).values
                df_perc_NO3_sim.loc[df_perc_NO3_sim.index[1]:, 'perc_obs'] = ds_obs['PERC'].isel(x=0, y=0).values
                df_perc_NO3_sim['NO3_mass'] = ds_sim_tm['M_q_ss'].isel(x=nrow, y=0).values
                df_perc_NO3_sim = df_perc_NO3_sim.join(sample_no)
                df_perc_NO3_sim.loc[:, 'sample_no'] = df_perc_NO3_sim.loc[:, 'sample_no'].bfill(limit=14)
                perc_sim_sum = df_perc_NO3_sim.groupby(['sample_no']).sum().loc[:, 'perc']
                perc_obs_sum = df_perc_NO3_sim.groupby(['sample_no']).sum().loc[:, 'perc_obs']
                NO3_sim_sum = df_perc_NO3_sim.groupby(['sample_no']).sum().loc[:, 'NO3_mass']
                sample_no['perc_sim_sum'] = perc_sim_sum.values
                sample_no['perc_obs_sum'] = perc_obs_sum.values
                sample_no['NO3_mass_sum'] = NO3_sim_sum.values
                sample_no['NO3_conc'] = sample_no['NO3_mass_sum'] / sample_no['perc_sim_sum']
                df_perc_NO3_sim = df_perc_NO3_sim.join(sample_no['NO3_conc'])
                df_perc_NO3_sim = df_perc_NO3_sim.join(sample_no['NO3_mass_sum'])
                df_perc_NO3_sim = df_perc_NO3_sim.join(sample_no['perc_sim_sum'])
                df_perc_NO3_sim = df_perc_NO3_sim.join(sample_no['perc_obs_sum'])
                # volume of observed bulk samples
                perc_bs_obs[nrow, 0, :] = df_perc_NO3_sim.loc[:, 'perc_obs_sum'].values.astype(float)
                # volume of simulated bulk samples
                perc_bs_sim[nrow, 0, :] = df_perc_NO3_sim.loc[:, 'perc_sim_sum'].values.astype(float)
                # concentration of simulated bulk samples
                NO3_perc_bs[nrow, 0, :] = df_perc_NO3_sim.loc[:, 'NO3_conc'].values.astype(float)
                # mass of simulated bulk samples
                NO3_perc_mass_bs[nrow, 0, :] = df_perc_NO3_sim.loc[:, 'NO3_mass_sum'].values.astype(float)

                # calculate metrics
                vars_sim = ['NO3_perc_bs', 'NO3_perc_mass_bs', 'perc_bs_sim']
                vars_obs = ['NO3_PERC', 'NO3_PERC_MASS', 'perc_bs_obs']
                for var_sim, var_obs in zip(vars_sim, vars_obs):
                    # join observations on simulations
                    if var_sim == 'NO3_perc_bs':
                        sim_vals = NO3_perc_bs[nrow, 0, :]
                        obs_vals = ds_obs[var_obs].isel(x=0, y=0).values.astype(float)
                        df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
                        df_obs.loc[:, 'obs'] = obs_vals
                    elif var_sim == 'NO3_perc_mass_bs':
                        sim_vals = NO3_perc_mass_bs[nrow, 0, :]
                        obs_vals = ds_obs[var_obs].isel(x=0, y=0).values.astype(float)
                        df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
                        df_obs.loc[:, 'obs'] = obs_vals
                    elif var_sim == 'perc_bs_sim':
                        sim_vals = perc_bs_sim[nrow, 0, :]
                        sim_vals = onp.where(sim_vals == 0, onp.nan, sim_vals)
                        obs_vals = perc_bs_obs[nrow, 0, 1:]
                        obs_vals = onp.where(obs_vals == 0, onp.nan, obs_vals)
                        df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
                        df_obs.loc[:, 'obs'] = obs_vals
                    df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
                    df_eval = df_eval.dropna()
                    obs_vals = df_eval.loc[:, 'obs'].values.astype(float)[1:]
                    sim_vals = df_eval.loc[:, 'sim'].values.astype(float)[1:]
                    key_kge = f'KGE_{var_sim}'
                    df_params_metrics.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals, sim_vals)
                    key_kge_alpha = f'KGE_alpha_{var_sim}'
                    df_params_metrics.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals, sim_vals)
                    key_kge_beta = f'KGE_beta_{var_sim}'
                    df_params_metrics.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals, sim_vals)
                    key_r = f'r_{var_sim}'
                    df_params_metrics.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals, sim_vals)
                    key_mae = "MAE_" + var_sim
                    df_params_metrics.loc[nrow, key_mae] = eval_utils.calc_mae(obs_vals, sim_vals)
                    key_mae = "50AE_" + var_sim
                    df_params_metrics.loc[nrow, key_mae] = eval_utils.calc_50ae(obs_vals, sim_vals)
                    key_rbs = "RBS_" + var_sim
                    df_params_metrics.loc[nrow, key_rbs] = eval_utils.calc_rbs(obs_vals, sim_vals)

                    obs_vals_year = df_eval.loc['2011':'2015', "obs"].values.astype(float)
                    sim_vals_year = df_eval.loc['2011':'2015', "sim"].values.astype(float)
                    year = "2011-2015"
                    key_kge = f'KGE_{var_sim}_{year}'
                    df_params_metrics.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals_year, sim_vals_year)
                    key_kge_alpha = "KGE_alpha_" + var_sim + f"_{year}"
                    df_params_metrics.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals_year, sim_vals_year)
                    key_kge_beta = "KGE_beta_" + var_sim + f"_{year}"
                    df_params_metrics.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals_year, sim_vals_year)
                    key_r = "r_" + var_sim + f"_{year}"
                    df_params_metrics.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals_year, sim_vals_year)
                    key_mae = "MAE_" + var_sim + f"_{year}"
                    df_params_metrics.loc[nrow, key_mae] = eval_utils.calc_mae(obs_vals_year, sim_vals_year)
                    key_mae = "50AE_" + var_sim + f"_{year}"
                    df_params_metrics.loc[nrow, key_mae] = eval_utils.calc_50ae(obs_vals_year, sim_vals_year)
                    key_rbs = "RBS_" + var_sim + f"_{year}"
                    df_params_metrics.loc[nrow, key_rbs] = eval_utils.calc_rbs(obs_vals_year, sim_vals_year)

                    obs_vals_year = df_eval.loc['2016':'2017', "obs"].values.astype(float)
                    sim_vals_year = df_eval.loc['2016':'2017', "sim"].values.astype(float)
                    year = "2016-2017"
                    key_kge = f'KGE_{var_sim}_{year}'
                    df_params_metrics.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals_year, sim_vals_year)
                    key_kge_alpha = "KGE_alpha_" + var_sim + f"_{year}"
                    df_params_metrics.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals_year, sim_vals_year)
                    key_kge_beta = "KGE_beta_" + var_sim + f"_{year}"
                    df_params_metrics.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals_year, sim_vals_year)
                    key_r = "r_" + var_sim + f"_{year}"
                    df_params_metrics.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals_year, sim_vals_year)
                    key_mae = "MAE_" + var_sim + f"_{year}"
                    df_params_metrics.loc[nrow, key_mae] = eval_utils.calc_mae(obs_vals_year, sim_vals_year)
                    key_mae = "50AE_" + var_sim + f"_{year}"
                    df_params_metrics.loc[nrow, key_mae] = eval_utils.calc_50ae(obs_vals_year, sim_vals_year)
                    key_rbs = "RBS_" + var_sim + f"_{year}"
                    df_params_metrics.loc[nrow, key_rbs] = eval_utils.calc_rbs(obs_vals_year, sim_vals_year)

                    for year in range(2011, 2018):
                        obs_vals_year = df_eval.loc[f'{year}', "obs"].values.astype(float)
                        sim_vals_year = df_eval.loc[f'{year}', "sim"].values.astype(float)
                        key_kge = f'KGE_{var_sim}_{year}'
                        df_params_metrics.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals_year, sim_vals_year)
                        key_kge_alpha = "KGE_alpha_" + var_sim + f"_{year}"
                        df_params_metrics.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals_year, sim_vals_year)
                        key_kge_beta = "KGE_beta_" + var_sim + f"_{year}"
                        df_params_metrics.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals_year, sim_vals_year)
                        key_r = "r_" + var_sim + f"_{year}"
                        df_params_metrics.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals_year, sim_vals_year)
                        key_mae = "MAE_" + var_sim + f"_{year}"
                        df_params_metrics.loc[nrow, key_mae] = eval_utils.calc_mae(obs_vals_year, sim_vals_year)
                        key_mae = "50AE_" + var_sim + f"_{year}"
                        df_params_metrics.loc[nrow, key_mae] = eval_utils.calc_50ae(obs_vals_year, sim_vals_year)
                        key_rbs = "RBS_" + var_sim + f"_{year}"
                        df_params_metrics.loc[nrow, key_rbs] = eval_utils.calc_rbs(obs_vals_year, sim_vals_year)

                sim_vals1 = ds_sim_tm['M_transp'].isel(x=nrow, y=0).values[1:]
                sim_vals2 = ds_sim_tm['nh4_up'].isel(x=nrow, y=0).values[1:]
                sim_vals = sim_vals1 + sim_vals2
                df_sim = pd.DataFrame(index=date_sim_tm[1:], columns=['sim'])
                df_sim.loc[:, 'sim'] = sim_vals * 0.01 # convert from mg/m2 to kg/ha
                df_sim_annual = df_sim.resample('YE').sum()
                sim_vals = df_sim_annual.loc[:, 'sim'].values.astype(float)[1:]
                obs_vals = onp.sum(ds_obs["N_UP"].isel(x=0, y=0).values, axis=-1)[1:]
                var_sim = 'N_uptake'
                key_kge_alpha = f'KGE_alpha_{var_sim}'
                df_params_metrics.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals, sim_vals)
                key_kge_beta = f'KGE_beta_{var_sim}'
                df_params_metrics.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals, sim_vals)
                key_r = f'r_{var_sim}'
                df_params_metrics.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals, sim_vals)
                key_mae = f"MAE_{var_sim}"
                df_params_metrics.loc[nrow, key_mae] = eval_utils.calc_mae(obs_vals, sim_vals)
                key_rbs = f"RBS_{var_sim}"
                df_params_metrics.loc[nrow, key_rbs] = eval_utils.calc_rbs(obs_vals, sim_vals)
                df_params_metrics = df_params_metrics.copy()

        # write to .txt
        file = base_path / "output" / "svat_crop_nitrate_monte_carlo_crop-specific" / f"params_metrics_{lys_experiment}_{tm_structure}.txt"
        df_params_metrics.to_csv(file, header=True, index=False, sep="\t")

        file = base_path / "output" / "svat_crop_nitrate_monte_carlo_crop-specific" / f"params_metrics_{lys_experiment}_{tm_structure}.csv"
        df_params_metrics.to_csv(file, header=True, index=False, sep=";")

        # add simulated bulk samples to the dataset
        ds_sim_tm.close()
        del(ds_sim_tm)
        with h5netcdf.File(states_tm_file, "a", decode_vlen_strings=False) as f:
            try:
                v = f.create_variable("M_q_ss_bs", ("x", "y", "Time"), float, compression="gzip", compression_opts=1)
                v[:, :, :] = NO3_perc_mass_bs
                v.attrs.update(long_name="Nitrate nitrogen load of bulk samples", units="mg")
            except ValueError:
                var_obj = f.variables.get("M_q_ss_bs")
                var_obj[:, :, :] = NO3_perc_mass_bs
            try:
                v = f.create_variable("C_q_ss_bs", ("x", "y", "Time"), float, compression="gzip", compression_opts=1)
                v[:, :, :] = NO3_perc_bs
                v.attrs.update(long_name="Nitrate nitrogen concentration of bulk samples", units="mg/l")
            except ValueError:
                var_obj = f.variables.get("C_q_ss_bs")
                var_obj[:, :, :] = NO3_perc_bs
            try:
                v = f.create_variable("q_ss_bs", ("x", "y", "Time"), float, compression="gzip", compression_opts=1)
                v[:, :, :] = perc_bs_sim
                v.attrs.update(long_name="Volume of simulated bulk samples", units="mg/l")
            except ValueError:
                var_obj = f.variables.get("q_ss_bs")
                var_obj[:, :, :] = perc_bs_sim
            try:
                v = f.create_variable("q_ss_bs_obs", ("x", "y", "Time"), float, compression="gzip", compression_opts=1)
                v[:, :, :] = perc_bs_obs
                v.attrs.update(long_name="Volume of measured bulk samples", units="mg/l")
            except ValueError:
                var_obj = f.variables.get("q_ss_bs_obs")
                var_obj[:, :, :] = perc_bs_obs