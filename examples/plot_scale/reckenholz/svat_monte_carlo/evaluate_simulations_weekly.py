import os
from pathlib import Path
from cftime import num2date
import xarray as xr
import pandas as pd
from de import de
import numpy as onp
import click
import roger.tools.evaluation as eval_utils

onp.random.seed(42)


@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path("/Volumes/LaCie/roger/examples/plot_scale/reckenholz")

    # directory of results
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)

    lys_experiments = ["lys2", "lys3", "lys8"]
    for lys_experiment in lys_experiments:
        # directory of results
        base_path_output = base_path / "output" / "svat_monte_carlo"
        if not os.path.exists(base_path_output):
            os.mkdir(base_path_output)

        # load simulation
        states_hm_mc_file = base_path_output / f"SVAT_{lys_experiment}.nc"
        ds_sim = xr.open_dataset(states_hm_mc_file, engine="h5netcdf")

        # load observations (measured data)
        path_obs = Path(__file__).parent.parent / "observations" / "reckenholz_lysimeter.nc"
        ds_obs = xr.open_dataset(path_obs, engine="h5netcdf", group=lys_experiment)

        # assign date
        days_sim = ds_sim["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        days_obs = ds_obs["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        date_sim = num2date(
            days_sim,
            units=f"days since {ds_sim['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        date_obs = num2date(
            days_obs,
            units=f"days since {ds_obs['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_sim = ds_sim.assign_coords(date=("Time", date_sim))
        ds_obs = ds_obs.assign_coords(date=("Time", date_obs))
        # DataFrame with sampled model parameters and the corresponding metrics
        csv_file = Path(__file__).parent / "parameters.csv"
        df_params_metrics = pd.read_csv(csv_file, sep=";", skiprows=1)
        # calculate metrics
        if lys_experiment in ["lys2", "lys3", "lys4", "lys8", "lys9"]:
            vars_sim = ["q_ss", "theta", "S"]
            vars_obs = ["PERC", "THETA", "WEIGHT"]
        else:
            vars_sim = ["q_ss"]
            vars_obs = ["PERC"]
        for var_sim, var_obs in zip(vars_sim, vars_obs):
            if var_sim == "theta":
                obs_vals = onp.mean(ds_obs["THETA"].isel(x=0, y=0).values, axis=0)
            elif var_sim == "S":
                obs_vals = ds_obs["WEIGHT"].isel(x=0, y=0).values - ds_obs["WEIGHT"].isel(x=0, y=0).values[0]
            else:
                obs_vals = ds_obs[var_obs].isel(x=0, y=0).values

            df_lys = pd.DataFrame(index=date_obs)
            df_lys.loc[:, "prec"] = ds_obs["PREC"].isel(x=0, y=0).values
            df_lys.loc[:, "pet"] = ds_obs["PET"].isel(x=0, y=0).values
            df_lys.loc[:, "perc"] = ds_obs["PERC"].isel(x=0, y=0).values
            df_lys.loc[:, "theta_avg"] = onp.mean(ds_obs["THETA"].isel(x=0, y=0).values, axis=0)
            df_lys.loc[:, "weight"] = ds_obs["WEIGHT"].isel(x=0, y=0).values
            df_lys.loc[df_lys.index[1]:, "dS"] = df_lys.loc[:, "weight"].values[1:] - df_lys.loc[:, "weight"].values[:-1]
            df_lys.loc[:, "perc_pet_ratio"] = df_lys.loc[:, "perc"] / (df_lys.loc[:, "perc"] + df_lys.loc[:, "pet"])
            df_lys = pd.DataFrame(index=date_sim).join(df_lys)
            # condition for plausible lysimeter seepage
            cond_perc = (df_lys.loc[:, "perc_pet_ratio"] >= 0.5) & ((df_lys.loc[:, "dS"] < (df_lys.loc[:, "pet"] + df_lys.loc[:, "perc"]) * (-1)) | (df_lys.loc[:, "dS"] <= df_lys.loc[:, "prec"]))
            cond_pet = (df_lys.loc[:, "perc_pet_ratio"] < 0.5) & ((df_lys.loc[:, "dS"] < (df_lys.loc[:, "pet"] + df_lys.loc[:, "perc"]) * (-1)) | (df_lys.loc[:, "dS"] <= df_lys.loc[:, "prec"]))
            cond0 = ((df_lys.loc[:, "theta_avg"] <= 0.3) & (df_lys.loc[:, "perc"] <= 0)) | ((df_lys.loc[:, "theta_avg"] >= 0.3) & (df_lys.loc[:, "prec"] > 0))
            cond_perc_pet = (cond_perc | cond_pet) & cond0

            ll_conds = ['_all', '_perc_dom', '_pet_dom', '_perc_pet']

            df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
            df_obs.loc[:, "obs"] = obs_vals
            for cond1 in ll_conds:
                for nrow in range(len(df_params_metrics.index)):
                    if var_sim == "S":
                        sim_vals = ds_sim[var_sim].isel(x=nrow, y=0).values - ds_sim[var_sim].isel(x=nrow, y=0).values[0]
                    else:
                        sim_vals = ds_sim[var_sim].isel(x=nrow, y=0).values
                    # join observations on simulations
                    df_eval = eval_utils.join_obs_on_sim(date_sim, sim_vals, df_obs)
                    if cond1 == "_perc_dom":
                        df_eval.loc[~cond_perc, :] = onp.nan
                    elif cond1 == "_pet_dom":
                        df_eval.loc[~cond_pet, :] = onp.nan
                    elif cond1 == "_perc_pet":
                        df_eval.loc[~cond_perc_pet, :] = onp.nan
                    df_eval = df_eval.dropna()
                    # number of data points
                    N_obs = len(df_eval.index)
                    df_params_metrics.loc[nrow, f"N_{var_obs}"] = N_obs
                    if N_obs > 30:
                        if var_sim in ["theta"]:
                            Ni = len(df_eval.index)
                            obs_vals = df_eval.loc[:, "obs"].values.astype(float)
                            sim_vals = df_eval.loc[:, "sim"].values.astype(float)
                            Nz = len(obs_vals)
                            eff_swc = eval_utils.calc_kge(obs_vals, sim_vals)
                            key_kge = "KGE_" + var_sim + cond1
                            df_params_metrics.loc[nrow, key_kge] = (Nz / Ni) * eff_swc
                            key_r = "r_" + var_sim + cond1
                            df_params_metrics.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals, sim_vals)
                        else:
                            if var_sim == "S":
                                df_eval_weekly = df_eval.resample("W").mean()
                                obs_vals = df_eval_weekly.loc[:, "obs"].values.astype(float)
                                sim_vals = df_eval_weekly.loc[:, "sim"].values.astype(float)
                            else:
                                df_eval_weekly = df_eval.resample("W").sum()
                                obs_vals = df_eval_weekly.loc[:, "obs"].values.astype(float)
                                sim_vals = df_eval_weekly.loc[:, "sim"].values.astype(float)
                            key_kge = "KGE_" + var_sim + cond1
                            df_params_metrics.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals, sim_vals)
                            key_kge_alpha = "KGE_alpha_" + var_sim + cond1
                            df_params_metrics.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals, sim_vals)
                            key_kge_beta = "KGE_beta_" + var_sim + cond1
                            df_params_metrics.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals, sim_vals)
                            key_r = "r_" + var_sim + cond1
                            df_params_metrics.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals, sim_vals)
                            key_mae = "MAE_" + var_sim + cond1
                            df_params_metrics.loc[nrow, key_mae] = eval_utils.calc_mae(obs_vals, sim_vals)
                            key_rbs = "RBS_" + var_sim + cond1
                            df_params_metrics.loc[nrow, key_rbs] = eval_utils.calc_rbs(obs_vals, sim_vals)
                            for year in range(2010, 2018):
                                try:
                                    obs_vals_year = df_eval.loc[f'{year}', "obs"].values.astype(float)
                                    sim_vals_year = df_eval.loc[f'{year}', "sim"].values.astype(float)
                                    key_kge = "KGE_" + var_sim + cond1 + f"_{year}"
                                    df_params_metrics.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals_year, sim_vals_year)
                                    key_kge_alpha = "KGE_alpha_" + var_sim + cond1 + f"_{year}"
                                    df_params_metrics.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals_year, sim_vals_year)
                                    key_kge_beta = "KGE_beta_" + var_sim + cond1 + f"_{year}"
                                    df_params_metrics.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals_year, sim_vals_year)
                                    key_r = "r_" + var_sim + cond1 + f"_{year}"
                                    df_params_metrics.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals_year, sim_vals_year)
                                    key_mae = "MAE_" + var_sim + cond1 + f"_{year}"
                                    df_params_metrics.loc[nrow, key_mae] = eval_utils.calc_mae(obs_vals_year, sim_vals_year)
                                    key_rbs = "RBS_" + var_sim + cond1 + f"_{year}"
                                    df_params_metrics.loc[nrow, key_rbs] = eval_utils.calc_rbs(obs_vals_year, sim_vals_year)
                                except KeyError:
                                    key_kge = "KGE_" + var_sim + cond1 + f"_{year}"
                                    df_params_metrics.loc[nrow, key_kge] = onp.nan
                                    key_kge_alpha = "KGE_alpha_" + var_sim + cond1 + f"_{year}"
                                    df_params_metrics.loc[nrow, key_kge_alpha] = onp.nan
                                    key_kge_beta = "KGE_beta_" + var_sim + cond1 + f"_{year}"
                                    df_params_metrics.loc[nrow, key_kge_beta] = onp.nan
                                    key_r = "r_" + var_sim + cond1 + f"_{year}"
                                    df_params_metrics.loc[nrow, key_r] = onp.nan
                                    key_mae = "MAE_" + var_sim + cond1 + f"_{year}"
                                    df_params_metrics.loc[nrow, key_mae] = onp.nan
                                    key_rbs = "RBS_" + var_sim + cond1 + f"_{year}"
                                    df_params_metrics.loc[nrow, key_rbs] = onp.nan

                            for year1, year2 in zip([2011, 2016, 2011, 2011], [2015, 2017, 2017, 2015]):
                                if var_sim == "S":
                                    df_eval_weekly = df_eval.resample("W").mean()
                                    obs_vals_year = df_eval_weekly.loc[f'{year1}':f'{year2}', "obs"].values.astype(float)
                                    sim_vals_year = df_eval_weekly.loc[f'{year1}':f'{year2}', "sim"].values.astype(float)
                                else:
                                    df_eval_weekly = df_eval.resample("W").sum()
                                    obs_vals_year = df_eval_weekly.loc[f'{year1}':f'{year2}', "obs"].values.astype(float)
                                    sim_vals_year = df_eval_weekly.loc[f'{year1}':f'{year2}', "sim"].values.astype(float)
                                key_kge = "KGE_" + var_sim + cond1 + f"_{year1}-{year2}"
                                df_params_metrics.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals_year, sim_vals_year)
                                key_kge_alpha = "KGE_alpha_" + var_sim + cond1 + f"_{year1}-{year2}"
                                df_params_metrics.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals_year, sim_vals_year)
                                key_kge_beta = "KGE_beta_" + var_sim + cond1 + f"_{year1}-{year2}"
                                df_params_metrics.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals_year, sim_vals_year)
                                key_r = "r_" + var_sim + cond1 + f"_{year1}-{year2}"
                                df_params_metrics.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals_year, sim_vals_year)
                                key_mae = "MAE_" + var_sim + cond1 + f"_{year1}-{year2}"
                                df_params_metrics.loc[nrow, key_mae] = eval_utils.calc_mae(obs_vals_year, sim_vals_year)
                                key_rbs = "RBS_" + var_sim + cond1 + f"_{year1}-{year2}"
                                df_params_metrics.loc[nrow, key_rbs] = eval_utils.calc_rbs(obs_vals_year, sim_vals_year)
                            cond0 = df_eval["obs"] == 0
                            if cond0.any():
                                # simulations and observations for which observed
                                # values are exclusively zero
                                df_obs0_sim = df_eval.loc[cond0, :]
                                N_obs0 = (df_obs0_sim["obs"] == 0).sum()
                                N_sim0 = (df_obs0_sim["sim"] == 0).sum()
                                # share of observations with zero values
                                key_p0 = "p0_" + var_sim + cond1
                                df_params_metrics.loc[nrow, key_p0] = N_obs0 / N_obs
                                # agreement of zero values
                                N_obs0 = (df_obs0_sim["obs"] == 0).sum()
                                N_sim0 = (df_obs0_sim["sim"] == 0).sum()
                                ioa0 = 1 - (N_sim0 / N_obs0)
                                key_ioa0 = "ioa0_" + var_sim + cond1
                                df_params_metrics.loc[nrow, key_ioa0] = ioa0
                                # mean absolute error from observations with zero values
                                obs0_vals = df_obs0_sim.loc[:, "obs"].values.astype(float)
                                sim0_vals = df_obs0_sim.loc[:, "sim"].values.astype(float)
                                key_mae0 = "MAE0_" + var_sim + cond1
                                df_params_metrics.loc[nrow, key_mae0] = eval_utils.calc_mae(obs0_vals, sim0_vals)
                                # peak difference from observations with zero values
                                key_pdiff0 = "PDIFF0_" + var_sim + cond1
                                df_params_metrics.loc[nrow, key_pdiff0] = onp.max(sim0_vals)
                                try:
                                    # simulations and observations with non-zero values
                                    cond_no0 = df_eval["obs"] > 0
                                    df_obs_sim_no0 = df_eval.loc[cond_no0, :]
                                    obs_vals_no0 = df_obs_sim_no0.loc[:, "obs"].values.astype(float)
                                    sim_vals_no0 = df_obs_sim_no0.loc[:, "sim"].values.astype(float)
                                    # number of data with non-zero observations
                                    N_no0 = len(df_obs_sim_no0.index)
                                    key_kge = "KGE_no0_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals_no0, sim_vals_no0)
                                    key_kge_alpha = "KGE_alpha_no0_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals_no0, sim_vals_no0)
                                    key_kge_beta = "KGE_beta_no0_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals_no0, sim_vals_no0)
                                    # mean absolute relative error
                                    key_mare = "MARE_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_mare] = eval_utils.calc_mare(obs_vals_no0, sim_vals_no0)
                                    # mean relative bias
                                    key_brel_mean = "brel_mean_" + var_sim + cond1
                                    brel_mean = de.calc_brel_mean(obs_vals_no0, sim_vals_no0)
                                    df_params_metrics.loc[nrow, key_brel_mean] = brel_mean
                                    # residual relative bias
                                    brel_res = de.calc_brel_res(obs_vals_no0, sim_vals_no0)
                                    # area of relative residual bias
                                    key_b_area = "b_area_" + var_sim + cond1
                                    b_area = de.calc_bias_area(brel_res)
                                    df_params_metrics.loc[nrow, key_b_area] = b_area
                                    # temporal correlation
                                    key_temp_cor = "temp_cor_" + var_sim + cond1
                                    temp_cor = de.calc_temp_cor(obs_vals_no0, sim_vals_no0)
                                    df_params_metrics.loc[nrow, key_temp_cor] = temp_cor
                                    # diagnostic efficiency
                                    key_de = "DE_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_de] = de.calc_de(obs_vals_no0, sim_vals_no0)
                                    # relative bias
                                    brel = de.calc_brel(obs_vals, sim_vals)
                                    # total bias
                                    key_b_tot = "b_tot_" + var_sim + cond1
                                    b_tot = de.calc_bias_tot(brel)
                                    df_params_metrics.loc[nrow, key_b_tot] = b_tot
                                    # bias of lower exceedance probability
                                    key_b_hf = "b_hf_" + var_sim + cond1
                                    b_hf = de.calc_bias_hf(brel)
                                    df_params_metrics.loc[nrow, key_b_hf] = b_hf
                                    # error contribution of higher exceedance probability
                                    key_err_hf = "err_hf_" + var_sim + cond1
                                    err_hf = de.calc_err_hf(b_hf, b_tot)
                                    df_params_metrics.loc[nrow, key_err_hf] = err_hf
                                    # bias of higher exceedance probability
                                    key_b_lf = "b_lf_" + var_sim + cond1
                                    b_lf = de.calc_bias_lf(brel)
                                    df_params_metrics.loc[nrow, key_b_lf] = b_lf
                                    # error contribution of lower exceedance probability
                                    key_err_lf = "err_lf_" + var_sim + cond1
                                    err_lf = de.calc_err_hf(b_lf, b_tot)
                                    df_params_metrics.loc[nrow, key_err_lf] = err_lf
                                    # direction of bias
                                    key_b_dir = "b_dir_" + var_sim + cond1
                                    b_dir = de.calc_bias_dir(brel_res)
                                    df_params_metrics.loc[nrow, key_b_dir] = b_dir
                                    # slope of bias
                                    key_b_slope = "b_slope_" + var_sim + cond1
                                    b_slope = de.calc_bias_slope(b_area, b_dir)
                                    df_params_metrics.loc[nrow, key_b_slope] = b_slope
                                    # (y, x) trigonometric inverse tangent
                                    key_phi = "phi_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_phi] = de.calc_phi(brel_mean, b_slope)
                                    # combined diagnostic efficiency
                                    key_de0 = "DE0_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_de0] = (N_no0 / N_obs) * df_params_metrics.loc[
                                        nrow, key_de
                                    ] + (N_obs0 / N_obs) * ioa0
                                except ValueError:
                                    key_kge = "KGE_no0_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_kge] = onp.nan
                                    key_kge_alpha = "KGE_alpha_no0_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_kge_alpha] = onp.nan
                                    key_kge_beta = "KGE_beta_no0_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_kge_beta] = onp.nan
                                    # mean absolute relative error
                                    key_mare = "MARE_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_mare] = onp.nan
                                    # mean relative bias
                                    key_brel_mean = "brel_mean_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_brel_mean] = onp.nan
                                    # area of relative residual bias
                                    key_b_area = "b_area_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_b_area] = onp.nan
                                    # temporal correlation
                                    key_temp_cor = "temp_cor_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_temp_cor] = onp.nan
                                    # diagnostic efficiency
                                    key_de = "DE_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_de] = onp.nan
                                    # total bias
                                    key_b_tot = "b_tot_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_b_tot] = onp.nan
                                    # bias of lower exceedance probability
                                    key_b_hf = "b_hf_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_b_hf] = onp.nan
                                    # error contribution of higher exceedance probability
                                    key_err_hf = "err_hf_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_err_hf] = onp.nan
                                    # bias of higher exceedance probability
                                    key_b_lf = "b_lf_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_b_lf] = onp.nan
                                    # error contribution of lower exceedance probability
                                    key_err_lf = "err_lf_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_err_lf] = onp.nan
                                    # direction of bias
                                    key_b_dir = "b_dir_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_b_dir] = onp.nan
                                    # slope of bias
                                    key_b_slope = "b_slope_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_b_slope] = onp.nan
                                    # (y, x) trigonometric inverse tangent
                                    key_phi = "phi_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_phi] = onp.nan
                                    # combined diagnostic efficiency
                                    key_de0 = "DE0_" + var_sim + cond1
                                    df_params_metrics.loc[nrow, key_de0] = onp.nan
                            else:
                                # share of observations with zero values
                                key_p0 = "p0_" + var_sim + cond1
                                df_params_metrics.loc[nrow, key_p0] = 0
                                # mean absolute relative error
                                key_mare = "MARE_" + var_sim + cond1
                                df_params_metrics.loc[nrow, key_mare] = eval_utils.calc_mare(obs_vals, sim_vals)
                                # mean relative bias
                                key_brel_mean = "brel_mean_" + var_sim + cond1
                                brel_mean = de.calc_brel_mean(obs_vals, sim_vals)
                                df_params_metrics.loc[nrow, key_brel_mean] = brel_mean
                                # residual relative bias
                                brel_res = de.calc_brel_res(obs_vals, sim_vals)
                                # area of relative residual bias
                                key_b_area = "b_area_" + var_sim + cond1
                                b_area = de.calc_bias_area(brel_res)
                                df_params_metrics.loc[nrow, key_b_area] = b_area
                                # temporal correlation
                                key_temp_cor = "temp_cor_" + var_sim + cond1
                                temp_cor = de.calc_temp_cor(obs_vals, sim_vals)
                                df_params_metrics.loc[nrow, key_temp_cor] = temp_cor
                                # diagnostic efficiency
                                key_de = "DE_" + var_sim + cond1
                                df_params_metrics.loc[nrow, key_de] = de.calc_de(obs_vals, sim_vals)
                                # relative bias
                                brel = de.calc_brel(obs_vals, sim_vals)
                                # total bias
                                key_b_tot = "b_tot_" + var_sim + cond1
                                b_tot = de.calc_bias_tot(brel)
                                df_params_metrics.loc[nrow, key_b_tot] = b_tot
                                # bias of lower exceedance probability
                                key_b_hf = "b_hf_" + var_sim + cond1
                                b_hf = de.calc_bias_hf(brel)
                                df_params_metrics.loc[nrow, key_b_hf] = b_hf
                                # error contribution of higher exceedance probability
                                key_err_hf = "err_hf_" + var_sim + cond1
                                err_hf = de.calc_err_hf(b_hf, b_tot)
                                df_params_metrics.loc[nrow, key_err_hf] = err_hf
                                # bias of higher exceedance probability
                                key_b_lf = "b_lf_" + var_sim + cond1
                                b_lf = de.calc_bias_lf(brel)
                                df_params_metrics.loc[nrow, key_b_lf] = b_lf
                                # error contribution of lower exceedance probability
                                key_err_lf = "err_lf_" + var_sim + cond1
                                err_lf = de.calc_err_hf(b_lf, b_tot)
                                df_params_metrics.loc[nrow, key_err_lf] = err_lf
                                # direction of bias
                                key_b_dir = "b_dir_" + var_sim + cond1
                                b_dir = de.calc_bias_dir(brel_res)
                                df_params_metrics.loc[nrow, key_b_dir] = b_dir
                                # slope of bias
                                key_b_slope = "b_slope_" + var_sim + cond1
                                b_slope = de.calc_bias_slope(b_area, b_dir)
                                df_params_metrics.loc[nrow, key_b_slope] = b_slope
                                # (y, x) trigonometric inverse tangent
                                key_phi = "phi_" + var_sim + cond1
                                df_params_metrics.loc[nrow, key_phi] = de.calc_phi(brel_mean, b_slope)

                df_params_metrics = df_params_metrics.copy()

        # write .txt-file
        file = base_path_output / f"params_eff_{lys_experiment}_weekly.txt"
        df_params_metrics.to_csv(file, header=True, index=False, sep="\t")

    return


if __name__ == "__main__":
    main()
