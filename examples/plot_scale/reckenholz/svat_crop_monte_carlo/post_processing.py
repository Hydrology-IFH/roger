import os
import glob
import h5netcdf
import datetime
from pathlib import Path
from cftime import num2date
import xarray as xr
import pandas as pd
from de import de
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as onp
import click
import roger
import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs
import roger.lookuptables as lut
onp.random.seed(42)


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

    states_hm_file = base_path / "states_svat_crop_monte_carlo.nc"
    if not os.path.exists(states_hm_file):
        lys_experiments = ["lys1", "lys2", "lys3", "lys4", "lys8", "lys9", "lys2_bromide", "lys8_bromide", "lys9_bromide"]
        for lys_experiment in lys_experiments:
            # merge model output into single file
            path = str(base_path / f'SVATCROP_{lys_experiment}.*.nc')
            diag_files = glob.glob(path)
            with h5netcdf.File(states_hm_file, 'a', decode_vlen_strings=False) as f:
                if lys_experiment not in list(f.groups.keys()):
                    f.create_group(lys_experiment)
                f.attrs.update(
                    date_created=datetime.datetime.today().isoformat(),
                    title='RoGeR model Monte Carlo simulations at Reckenholz lysimeter site',
                    institution='University of Freiburg, Chair of Hydrology',
                    references='',
                    comment='SVAT model with free drainage and crop phenology/crop rotation'
                )
                # collect dimensions
                for dfs in diag_files:
                    with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                        # set dimensions with a dictionary
                        if not dfs.split('/')[-1].split('.')[1] == 'constant':
                            dict_dim = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'n_crop_types': len(df.variables['n_crop_types']), 'crops': len(df.variables['crops']), 'Time': len(df.variables['Time'])}
                            time = onp.array(df.variables.get('Time'))
                for dfs in diag_files:
                    with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                        if not f.groups[lys_experiment].dimensions:
                            f.groups[lys_experiment].dimensions = dict_dim
                            v = f.groups[lys_experiment].create_variable('x', ('x',), float, compression="gzip", compression_opts=1)
                            v.attrs['long_name'] = 'model run'
                            v.attrs['units'] = ''
                            v[:] = onp.arange(dict_dim["x"])
                            v = f.groups[lys_experiment].create_variable('y', ('y',), float, compression="gzip", compression_opts=1)
                            v.attrs['long_name'] = ''
                            v.attrs['units'] = ''
                            v[:] = onp.arange(dict_dim["y"])
                            v = f.groups[lys_experiment].create_variable('Time', ('Time',), float, compression="gzip", compression_opts=1)
                            var_obj = df.variables.get('Time')
                            v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                           units=var_obj.attrs["units"])
                            v[:] = time
                            v = f.groups[lys_experiment].create_variable('n_crop_types', ('n_crop_types',), int, compression="gzip", compression_opts=1)
                            v.attrs['long_name'] = 'number of crop types'
                            v.attrs['units'] = ''
                            v[:] = onp.arange(dict_dim["n_crop_types"])
                            v = f.groups[lys_experiment].create_variable('crops', ('crops',), int, compression="gzip", compression_opts=1)
                            v.attrs['long_name'] = 'number of crops per growing cycle'
                            v.attrs['units'] = ''
                            v[:] = onp.arange(dict_dim["crops"])
                        for key in list(df.variables.keys()):
                            var_obj = df.variables.get(key)
                            if key not in list(f.groups[lys_experiment].dimensions.keys()) and ('Time', 'y', 'x') == var_obj.dimensions and var_obj.shape[0] > 2:
                                v = f.groups[lys_experiment].create_variable(key, ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                                vals = onp.array(var_obj)
                                v[:, :, :] = vals.swapaxes(0, 2)
                                v.attrs.update(long_name=var_obj.attrs["long_name"],
                                               units=var_obj.attrs["units"])
                            elif key not in list(f.groups[lys_experiment].dimensions.keys()) and ('Time', 'crops', 'y', 'x') == var_obj.dimensions and var_obj.shape[0] > 2:
                                v = f.groups[lys_experiment].create_variable(key, ('x', 'y', 'Time', 'crops'), float, compression="gzip", compression_opts=1)
                                vals = onp.array(var_obj)
                                vals = vals.swapaxes(0, 3)
                                vals = vals.swapaxes(1, 2)
                                vals = vals.swapaxes(2, 3)
                                v[:, :, :, :] = vals
                                v.attrs.update(long_name=var_obj.attrs["long_name"],
                                               units=var_obj.attrs["units"])
                            elif key not in list(f.groups[lys_experiment].dimensions.keys()) and ('Time', 'y', 'x') == var_obj.dimensions and var_obj.shape[0] <= 2:
                                v = f.groups[lys_experiment].create_variable(key, ('x', 'y'), float, compression="gzip", compression_opts=1)
                                vals = onp.array(var_obj)
                                v[:, :] = vals.swapaxes(0, 2)[:, :, 0]
                                v.attrs.update(long_name=var_obj.attrs["long_name"],
                                               units=var_obj.attrs["units"])
                            elif key not in list(f.groups[lys_experiment].dimensions.keys()) and ('Time', 'n_crop_types', 'y', 'x') == var_obj.dimensions and var_obj.shape[0] <= 2:
                                v = f.groups[lys_experiment].create_variable(key, ('x', 'y', 'n_crop_types'), float, compression="gzip", compression_opts=1)
                                vals = onp.array(var_obj)
                                vals = vals.swapaxes(0, 3)
                                vals = vals.swapaxes(1, 2)
                                v[:, :, :] = vals[:, :, :, 0]
                                v.attrs.update(long_name=var_obj.attrs["long_name"],
                                               units=var_obj.attrs["units"])

    lys_experiments = ["lys2", "lys3", "lys4", "lys8", "lys9", "lys2_bromide", "lys8_bromide", "lys9_bromide"]
    for lys_experiment in lys_experiments:
        dict_params_eff = {}
        # directory of results
        base_path_results = base_path / "results" / lys_experiment
        if not os.path.exists(base_path_results):
            os.mkdir(base_path_results)
        # directory of figures
        base_path_figs = base_path / "figures" / lys_experiment
        if not os.path.exists(base_path_figs):
            os.mkdir(base_path_figs)

        # load simulation
        states_hm_mc_file = base_path / "states_svat_crop_monte_carlo.nc"
        ds_sim = xr.open_dataset(states_hm_mc_file, engine="h5netcdf", group=lys_experiment)

        # load observations (measured data)
        path_obs = Path(__file__).parent.parent / "observations" / "reckenholz_lysimeter.nc"
        ds_obs = xr.open_dataset(path_obs, engine="h5netcdf", group=lys_experiment)

        # assign date
        days_sim = (ds_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date_sim = num2date(days_sim, units=f"days since {ds_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_sim = ds_sim.assign_coords(date=("Time", date_sim))
        ds_obs = ds_obs.assign_coords(date=("Time", date_obs))
        # Dataframe with observed crop types
        df_crop_types = pd.DataFrame(index=date_obs)
        df_crop_types.loc[:, 'crop_type'] = ds_obs['CROP_TYPE'].isel(x=0, y=0).values.astype(str)
        # get list with land use IDs
        crop_types_sim1 = onp.unique(ds_sim['lu_id'].values.flatten()).tolist()
        crop_types_sim = [0] + crop_types_sim1
        for crop_type_sim in crop_types_sim:
            crop_type_sim = int(crop_type_sim)
            # DataFrame with sampled model parameters and the corresponding metrics
            nx = ds_sim.dims['x']  # number of rows
            ny = ds_sim.dims['y']  # number of columns
            df_params_eff = pd.DataFrame(index=range(nx * ny))
            # sampled model parameters
            df_params_eff.loc[:, 'dmpv'] = ds_sim["dmpv"].isel(y=0).values.flatten()
            df_params_eff.loc[:, 'lmpv'] = ds_sim["lmpv"].isel(y=0).values.flatten()
            df_params_eff.loc[:, 'theta_ac'] = ds_sim["theta_ac"].isel(y=0).values.flatten()
            df_params_eff.loc[:, 'theta_ufc'] = ds_sim["theta_ufc"].isel(y=0).values.flatten()
            df_params_eff.loc[:, 'theta_pwp'] = ds_sim["theta_pwp"].isel(y=0).values.flatten()
            df_params_eff.loc[:, 'ks'] = ds_sim["ks"].isel(y=0).values.flatten()
            if crop_type_sim > 500:
                row = onp.where(lut.ARR_CP[:, 0] == crop_type_sim)[0]
                df_params_eff.loc[:, f'crop_scale_{crop_type_sim}'] = ds_sim["lut_crop_scale"].isel(y=0, n_crop_types=row).values
            else:
                df_params_eff.loc[:, f'crop_scale_{crop_type_sim}'] = onp.nan
            # calculate metrics
            vars_sim = ['q_ss', 'theta', 'S_s', 'S']
            vars_obs = ['PERC', 'THETA', 'WEIGHT', 'WEIGHT']
            for var_sim, var_obs in zip(vars_sim, vars_obs):
                if var_sim == 'theta':
                    obs_vals = onp.mean(ds_obs['THETA'].isel(x=0, y=0).values, axis=0)
                else:
                    obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
                df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
                df_obs.loc[:, 'obs'] = obs_vals
                for nrow in range(nx * ny):
                    sim_vals = ds_sim[var_sim].isel(x=nrow, y=0).values
                    # join observations on simulations
                    df_eval = eval_utils.join_obs_on_sim(date_sim, sim_vals, df_obs)
                    # select data by crop type
                    if crop_type_sim > 500:
                        df_rows = pd.DataFrame(index=df_eval.index).join(df_crop_types)
                        rows = (df_rows['crop_type'].values == lut.dict_crops[crop_type_sim])
                        df_eval = df_eval.loc[rows, :]
                    df_eval = df_eval.dropna()
                    # number of data points
                    N_obs = len(df_eval.index)
                    df_params_eff.loc[nrow, 'N'] = N_obs
                    if N_obs > 30:
                        if var_sim in ['theta']:
                            Ni = len(df_eval.index)
                            obs_vals = df_eval.loc[:, 'obs'].values
                            sim_vals = df_eval.loc[:, 'sim'].values
                            Nz = len(obs_vals)
                            eff_swc = eval_utils.calc_kge(obs_vals, sim_vals)
                            key_kge = 'KGE_' + var_sim
                            df_params_eff.loc[nrow, key_kge] = (Nz / Ni) * eff_swc
                        elif var_sim in ['S', 'S_s']:
                            obs_vals = df_eval.loc[:, 'obs'].values
                            sim_vals = df_eval.loc[:, 'sim'].values
                            key_r = 'r_' + var_sim
                            df_params_eff.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals, sim_vals)
                        else:
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
                            cond0 = (df_eval['obs'] == 0)
                            if cond0.any():
                                # simulations and observations for which observed
                                # values are exclusively zero
                                df_obs0_sim = df_eval.loc[cond0, :]
                                N_obs0 = (df_obs0_sim['obs'] == 0).sum()
                                N_sim0 = (df_obs0_sim['sim'] == 0).sum()
                                # share of observations with zero values
                                key_p0 = 'p0_' + var_sim
                                df_params_eff.loc[nrow, key_p0] = N_obs0 / N_obs
                                # agreement of zero values
                                N_obs0 = (df_obs0_sim['obs'] == 0).sum()
                                N_sim0 = (df_obs0_sim['sim'] == 0).sum()
                                ioa0 = 1 - (N_sim0 / N_obs0)
                                key_ioa0 = 'ioa0_' + var_sim
                                df_params_eff.loc[nrow, key_ioa0] = ioa0
                                # mean absolute error from observations with zero values
                                obs0_vals = df_obs0_sim.loc[:, 'obs'].values
                                sim0_vals = df_obs0_sim.loc[:, 'sim'].values
                                key_mae0 = 'MAE0_' + var_sim
                                df_params_eff.loc[nrow, key_mae0] = eval_utils.calc_mae(obs0_vals,
                                                                                        sim0_vals)
                                # peak difference from observations with zero values
                                key_pdiff0 = 'PDIFF0_' + var_sim
                                df_params_eff.loc[nrow, key_pdiff0] = onp.max(sim0_vals)
                                # simulations and observations with non-zero values
                                cond_no0 = (df_eval['obs'] > 0)
                                df_obs_sim_no0 = df_eval.loc[cond_no0, :]
                                obs_vals_no0 = df_obs_sim_no0.loc[:, 'obs'].values
                                sim_vals_no0 = df_obs_sim_no0.loc[:, 'sim'].values
                                # number of data with non-zero observations
                                N_no0 = len(df_obs_sim_no0.index)
                                # mean absolute relative error
                                key_mare = 'MARE_' + var_sim
                                df_params_eff.loc[nrow, key_mare] = eval_utils.calc_mare(obs_vals_no0, sim_vals_no0)
                                # mean relative bias
                                key_brel_mean = 'brel_mean_' + var_sim
                                brel_mean = de.calc_brel_mean(obs_vals_no0, sim_vals_no0)
                                df_params_eff.loc[nrow, key_brel_mean] = brel_mean
                                # residual relative bias
                                brel_res = de.calc_brel_res(obs_vals_no0, sim_vals_no0)
                                # area of relative residual bias
                                key_b_area = 'b_area_' + var_sim
                                b_area = de.calc_bias_area(brel_res)
                                df_params_eff.loc[nrow, key_b_area] = b_area
                                # temporal correlation
                                key_temp_cor = 'temp_cor_' + var_sim
                                temp_cor = de.calc_temp_cor(obs_vals_no0, sim_vals_no0)
                                df_params_eff.loc[nrow, key_temp_cor] = temp_cor
                                # diagnostic efficiency
                                key_de = 'DE_' + var_sim
                                df_params_eff.loc[nrow, key_de] = de.calc_de(obs_vals_no0, sim_vals_no0)
                                # relative bias
                                brel = de.calc_brel(obs_vals, sim_vals)
                                # total bias
                                key_b_tot = 'b_tot_' + var_sim
                                b_tot = de.calc_bias_tot(brel)
                                df_params_eff.loc[nrow, key_b_tot] = b_tot
                                # bias of lower exceedance probability
                                key_b_hf = 'b_hf_' + var_sim
                                b_hf = de.calc_bias_hf(brel)
                                df_params_eff.loc[nrow, key_b_hf] = b_hf
                                # error contribution of higher exceedance probability
                                key_err_hf = 'err_hf_' + var_sim
                                err_hf = de.calc_err_hf(b_hf, b_tot)
                                df_params_eff.loc[nrow, key_err_hf] = err_hf
                                # bias of higher exceedance probability
                                key_b_lf = 'b_lf_' + var_sim
                                b_lf = de.calc_bias_lf(brel)
                                df_params_eff.loc[nrow, key_b_lf] = b_lf
                                # error contribution of lower exceedance probability
                                key_err_lf = 'err_lf_' + var_sim
                                err_lf = de.calc_err_hf(b_lf, b_tot)
                                df_params_eff.loc[nrow, key_err_lf] = err_lf
                                # direction of bias
                                key_b_dir = 'b_dir_' + var_sim
                                b_dir = de.calc_bias_dir(brel_res)
                                df_params_eff.loc[nrow, key_b_dir] = b_dir
                                # slope of bias
                                key_b_slope = 'b_slope_' + var_sim
                                b_slope = de.calc_bias_slope(b_area, b_dir)
                                df_params_eff.loc[nrow, key_b_slope] = b_slope
                                # (y, x) trigonometric inverse tangent
                                key_phi = 'phi_' + var_sim
                                df_params_eff.loc[nrow, key_phi] = de.calc_phi(brel_mean, b_slope)
                                # combined diagnostic efficiency
                                key_de0 = 'DE0_' + var_sim
                                df_params_eff.loc[nrow, key_de0] = (N_no0 / N_obs) * df_params_eff.loc[nrow, key_de] + (N_obs0 / N_obs) * ioa0
                            else:
                                # share of observations with zero values
                                key_p0 = 'p0_' + var_sim
                                df_params_eff.loc[nrow, key_p0] = 0
                                # mean absolute relative error
                                key_mare = 'MARE_' + var_sim
                                df_params_eff.loc[nrow, key_mare] = eval_utils.calc_mare(obs_vals, sim_vals)
                                # mean relative bias
                                key_brel_mean = 'brel_mean_' + var_sim
                                brel_mean = de.calc_brel_mean(obs_vals, sim_vals)
                                df_params_eff.loc[nrow, key_brel_mean] = brel_mean
                                # residual relative bias
                                brel_res = de.calc_brel_res(obs_vals, sim_vals)
                                # area of relative residual bias
                                key_b_area = 'b_area_' + var_sim
                                b_area = de.calc_bias_area(brel_res)
                                df_params_eff.loc[nrow, key_b_area] = b_area
                                # temporal correlation
                                key_temp_cor = 'temp_cor_' + var_sim
                                temp_cor = de.calc_temp_cor(obs_vals, sim_vals)
                                df_params_eff.loc[nrow, key_temp_cor] = temp_cor
                                # diagnostic efficiency
                                key_de = 'DE_' + var_sim
                                df_params_eff.loc[nrow, key_de] = de.calc_de(obs_vals, sim_vals)
                                # relative bias
                                brel = de.calc_brel(obs_vals, sim_vals)
                                # total bias
                                key_b_tot = 'b_tot_' + var_sim
                                b_tot = de.calc_bias_tot(brel)
                                df_params_eff.loc[nrow, key_b_tot] = b_tot
                                # bias of lower exceedance probability
                                key_b_hf = 'b_hf_' + var_sim
                                b_hf = de.calc_bias_hf(brel)
                                df_params_eff.loc[nrow, key_b_hf] = b_hf
                                # error contribution of higher exceedance probability
                                key_err_hf = 'err_hf_' + var_sim
                                err_hf = de.calc_err_hf(b_hf, b_tot)
                                df_params_eff.loc[nrow, key_err_hf] = err_hf
                                # bias of higher exceedance probability
                                key_b_lf = 'b_lf_' + var_sim
                                b_lf = de.calc_bias_lf(brel)
                                df_params_eff.loc[nrow, key_b_lf] = b_lf
                                # error contribution of lower exceedance probability
                                key_err_lf = 'err_lf_' + var_sim
                                err_lf = de.calc_err_hf(b_lf, b_tot)
                                df_params_eff.loc[nrow, key_err_lf] = err_lf
                                # direction of bias
                                key_b_dir = 'b_dir_' + var_sim
                                b_dir = de.calc_bias_dir(brel_res)
                                df_params_eff.loc[nrow, key_b_dir] = b_dir
                                # slope of bias
                                key_b_slope = 'b_slope_' + var_sim
                                b_slope = de.calc_bias_slope(b_area, b_dir)
                                df_params_eff.loc[nrow, key_b_slope] = b_slope
                                # (y, x) trigonometric inverse tangent
                                key_phi = 'phi_' + var_sim
                                df_params_eff.loc[nrow, key_phi] = de.calc_phi(brel_mean, b_slope)

            # Calculate multi-objective metric
            if 'r_S' in df_params_eff.columns:
                df_params_eff.loc[:, 'E_multi'] = 1/2 * df_params_eff.loc[:, 'r_S'] + 1/2 * df_params_eff.loc[:, 'KGE_q_ss']

            # write .txt-file
            file = base_path_results / f"params_eff_{crop_type_sim}_{lys_experiment}.txt"
            df_params_eff.to_csv(file, header=True, index=False, sep="\t")
            dict_params_eff[f'{crop_type_sim}'] = df_params_eff

            # dotty plots
            if 'E_multi' in df_params_eff.columns:
                df_eff = df_params_eff.loc[:, ['KGE_q_ss', 'r_S', 'E_multi']]
                df_params = df_params_eff.loc[:, ['dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks', f'crop_scale_{crop_type_sim}']]
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

                ax[0, 0].set_ylabel('$KGE_{PERC}$ [-]')
                ax[1, 0].set_ylabel(r'$r_{S}$ [-]')
                ax[2, 0].set_ylabel('$E_{multi}$\n [-]')

                fig.subplots_adjust(wspace=0.2, hspace=0.3)
                file = base_path_figs / f"dotty_plots_{crop_type_sim}_{lys_experiment}.png"
                fig.savefig(file, dpi=250)

        # select best model run
        idx_best = df_params_eff['E_multi'].idxmax()

        ds_sim = ds_sim.close()
        del ds_sim
        # write states of best model run
        states_hm_mc_file = base_path / "states_svat_crop_monte_carlo.nc"
        states_hm_file = base_path / "states_svat_crop.nc"
        with h5netcdf.File(states_hm_file, 'a', decode_vlen_strings=False) as f:
            if lys_experiment not in list(f.groups.keys()):
                f.create_group(lys_experiment)
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='RoGeR best Monte Carlo simulation at Reckenholz Lysimeter site',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment='First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).',
                model_structure='SVAT model with free drainage and crop phenology/crop rotation',
                roger_version=f'{roger.__version__}'
            )
            with h5netcdf.File(states_hm_mc_file, 'r', decode_vlen_strings=False) as df:
                # set dimensions with a dictionary
                dict_dim = {'x': 1, 'y': 1, 'Time': len(df.groups[lys_experiment].variables['Time'])}
                if not f.groups[lys_experiment].dimensions:
                    f.groups[lys_experiment].dimensions = dict_dim
                    v = f.groups[lys_experiment].create_variable('x', ('x',), float, compression="gzip", compression_opts=1)
                    v.attrs['long_name'] = 'Number of model run'
                    v.attrs['units'] = ''
                    v[:] = onp.arange(dict_dim["x"])
                    v = f.groups[lys_experiment].create_variable('y', ('y',), float, compression="gzip", compression_opts=1)
                    v.attrs['long_name'] = ''
                    v.attrs['units'] = ''
                    v[:] = onp.arange(dict_dim["y"])
                    v = f.groups[lys_experiment].create_variable('Time', ('Time',), float, compression="gzip", compression_opts=1)
                    var_obj = df.groups[lys_experiment].variables.get('Time')
                    v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                   units=var_obj.attrs["units"])
                    v[:] = onp.array(var_obj)
                for var_sim in list(df.variables.keys()):
                    var_obj = df.groups[lys_experiment].variables.get(var_sim)
                    if var_sim not in list(f.groups[lys_experiment].dimensions.keys()) and ('x', 'y', 'Time') == var_obj.dimensions:
                        v = f.groups[lys_experiment].create_variable(var_sim, ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        v[:, :, :] = vals[idx_best, :, :]
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                       units=var_obj.attrs["units"])
                    elif var_sim not in list(f.groups[lys_experiment].dimensions.keys()) and ('x', 'y') == var_obj.dimensions:
                        v = f.groups[lys_experiment].create_variable(var_sim, ('x', 'y'), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        v[:, :] = vals[idx_best, :]
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                       units=var_obj.attrs["units"])

        # select best 1% simulations
        df_params_eff1 = df_params_eff.copy()
        df_params_eff1.loc[:, 'id'] = range(len(df_params_eff1.index))
        df_params_eff1 = df_params_eff1.sort_values(by=['E_multi'], ascending=False)
        idx_best1 = df_params_eff1.loc[:df_params_eff1.index[99], 'id'].values.tolist()
        # write states of best model run
        states_hm_mc_file = base_path / "states_svat_crop_monte_carlo.nc"
        states_hm_file = base_path / "states_svat_crop1.nc"
        with h5netcdf.File(states_hm_file, 'a', decode_vlen_strings=False) as f:
            if lys_experiment not in list(f.groups.keys()):
                f.create_group(lys_experiment)
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='RoGeR best 1% Monte Carlo simulations at Reckenholz lysimeter site',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment='First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).',
                model_structure='SVAT model with free drainage and crop phenology/crop rotation',
                roger_version=f'{roger.__version__}'
            )
            with h5netcdf.File(states_hm_mc_file, 'r', decode_vlen_strings=False) as df:
                # set dimensions with a dictionary
                dict_dim = {'x': len(idx_best1), 'y': 1, 'Time': len(df.variables['Time'])}
                if not f.groups[lys_experiment].dimensions:
                    f.groups[lys_experiment].dimensions = dict_dim
                    v = f.create_variable('x', ('x',), float, compression="gzip", compression_opts=1)
                    v.attrs['long_name'] = 'Number of model run'
                    v.attrs['units'] = ''
                    v[:] = onp.arange(dict_dim["x"])
                    v = f.groups[lys_experiment].create_variable('y', ('y',), float, compression="gzip", compression_opts=1)
                    v.attrs['long_name'] = ''
                    v.attrs['units'] = ''
                    v[:] = onp.arange(dict_dim["y"])
                    v = f.groups[lys_experiment].create_variable('Time', ('Time',), float, compression="gzip", compression_opts=1)
                    var_obj = df.variables.get('Time')
                    v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                   units=var_obj.attrs["units"])
                    v[:] = onp.array(var_obj)
                for var_sim in list(df.variables.keys()):
                    var_obj = df.variables.get(var_sim)
                    if var_sim not in list(f.dimensions.keys()) and ('x', 'y', 'Time') == var_obj.dimensions:
                        v = f.groups[lys_experiment].create_variable(var_sim, ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        v[:, :, :] = vals[idx_best1, :, :]
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                       units=var_obj.attrs["units"])
                    elif var_sim not in list(f.dimensions.keys()) and ('x', 'y') == var_obj.dimensions:
                        v = f.groups[lys_experiment].create_variable(var_sim, ('x', 'y'), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        v[:, :] = vals[idx_best1, :]
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                       units=var_obj.attrs["units"])

    return


if __name__ == "__main__":
    main()
