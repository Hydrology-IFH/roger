import os
from pathlib import Path
from cftime import num2date
import xarray as xr
import pandas as pd
from de import de
import numpy as onp
import click
import roger.tools.evaluation as eval_utils


@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent
    # directory of results
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    # load simulation
    hm_mc_file = base_path / "output" / "SVAT.nc"
    ds_sim = xr.open_dataset(hm_mc_file, engine="h5netcdf")
    # assign date
    days_sim = ds_sim["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_sim = num2date(
        days_sim,
        units=f"days since {ds_sim['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_sim = ds_sim.assign_coords(date=("Time", date_sim))

    # load observations (measured data)
    path_obs = Path(__file__).parent.parent / "observations" / "rietholzbach_lysimeter.nc"
    ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")
    # assign date
    days_obs = ds_obs["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_obs = num2date(
        days_obs,
        units=f"days since {ds_obs['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

    # average observed soil water content of previous 5 days
    window = 5
    df_thetap = pd.DataFrame(index=date_obs, columns=["doy", "theta", "sc"])
    df_thetap.loc[:, "doy"] = df_thetap.index.day_of_year
    df_thetap.loc[:, "theta"] = onp.mean(ds_obs["THETA"].isel(x=0, y=0).values, axis=0)
    df_thetap.loc[df_thetap.index[window - 1] :, f"theta_avg{window}"] = (
        df_thetap.loc[:, "theta"].rolling(window=window).mean().iloc[window - 1 :].values
    )
    df_thetap.iloc[:window, 2] = onp.nan
    theta_lower = df_thetap.loc[:, f"theta_avg{window}"].quantile(0.1)
    theta_upper = df_thetap.loc[:, f"theta_avg{window}"].quantile(0.9)
    cond1 = df_thetap[f"theta_avg{window}"] < theta_lower
    cond2 = (df_thetap[f"theta_avg{window}"] >= theta_lower) & (df_thetap[f"theta_avg{window}"] < theta_upper)
    cond3 = df_thetap[f"theta_avg{window}"] >= theta_upper
    df_thetap.loc[cond1, "sc"] = 1  # dry
    df_thetap.loc[cond2, "sc"] = 2  # normal
    df_thetap.loc[cond3, "sc"] = 3  # wet

    # DataFrame with sampled model parameters and the corresponding metrics
    nx = ds_sim.dims["x"]  # number of rows
    ny = ds_sim.dims["y"]  # number of columns
    file = base_path_figs / "params_metrics_monthly.txt"
    if not os.path.exists(file):
        click.echo("Calculate metrics ...")
        df_params_metrics = pd.DataFrame(index=range(nx * ny))
        # sampled model parameters
        df_params_metrics.loc[:, "c1_mak"] = ds_sim["c1_mak"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "c2_mak"] = ds_sim["c2_mak"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "dmpv"] = ds_sim["dmpv"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "lmpv"] = ds_sim["lmpv"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "theta_eff"] = ds_sim["theta_eff"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "frac_lp"] = ds_sim["frac_lp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "frac_fp"] = ds_sim["frac_fp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "theta_ac"] = ds_sim["theta_ac"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "theta_ufc"] = ds_sim["theta_ufc"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "theta_pwp"] = ds_sim["theta_pwp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "ks"] = ds_sim["ks"].isel(y=0).values.flatten()
        # calculate metrics
        vars_sim = ["aet", "q_ss", "dS"]
        vars_obs = ["AET", "PERC", "dWEIGHT"]
        for var_sim, var_obs in zip(vars_sim, vars_obs):
            obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
            df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
            df_obs.loc[:, "obs"] = obs_vals
            for nrow in range(nx * ny):
                for sc, sc1 in zip([0, 1, 2, 3], ["", "dry", "normal", "wet"]):
                    sim_vals = ds_sim[var_sim].isel(x=nrow, y=0).values
                    # join observations on simulations
                    df_eval = eval_utils.join_obs_on_sim(date_sim, sim_vals, df_obs)

                    if sc > 0:
                        df_rows = pd.DataFrame(index=df_eval.index).join(df_thetap)
                        rows = df_rows["sc"].values == sc
                        df_eval = df_eval.loc[rows, :]

                    if var_sim in ["dS"]:
                        df_eval.loc["2000-01":"2000-06", :] = onp.nan
                        df_eval_monthly = df_eval.resample("ME").mean()
                        df_eval = df_eval_monthly.dropna()
                        obs_vals = df_eval.loc[:, "obs"].values.astype(float)
                        sim_vals = df_eval.loc[:, "sim"].values.astype(float)
                        # add offset since diagnostic efficiency requires positive values
                        offset = onp.nanmin(df_eval.values) * (-1) + 1
                        obs_vals = df_eval.loc[:, "obs"].values.astype(float) + offset
                        sim_vals = df_eval.loc[:, "sim"].values.astype(float) + offset
                        key_kge = "KGE_" + var_sim + f"{sc1}"
                        df_params_metrics.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals, sim_vals)
                        key_kge_alpha = "KGE_alpha_" + var_sim + f"{sc1}"
                        df_params_metrics.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals, sim_vals)
                        key_kge_beta = "KGE_beta_" + var_sim + f"{sc1}"
                        df_params_metrics.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals, sim_vals)
                        key_r = "r_" + var_sim + f"{sc1}"
                        df_params_metrics.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals, sim_vals)
                        key_E = "E_" + var_sim + f"{sc1}"
                        df_params_metrics.loc[nrow, key_E] = (1 - ((1 - df_params_metrics.loc[nrow, key_r])**2 + (1 - df_params_metrics.loc[nrow, key_kge_alpha])**2)**(0.5))
                        # share of observations with zero values
                        key_p0 = "p0_" + var_sim + f"{sc1}"
                        df_params_metrics.loc[nrow, key_p0] = 0
                        # mean absolute relative error
                        key_mare = "MARE_" + var_sim + f"{sc1}"
                        df_params_metrics.loc[nrow, key_mare] = eval_utils.calc_mare(obs_vals, sim_vals)
                        # mean relative bias
                        key_brel_mean = "brel_mean_" + var_sim + f"{sc1}"
                        brel_mean = de.calc_brel_mean(obs_vals, sim_vals)
                        df_params_metrics.loc[nrow, key_brel_mean] = brel_mean
                        # residual relative bias
                        brel_res = de.calc_brel_res(obs_vals, sim_vals)
                        # area of relative residual bias
                        key_b_area = "b_area_" + var_sim + f"{sc1}"
                        b_area = de.calc_bias_area(brel_res)
                        df_params_metrics.loc[nrow, key_b_area] = b_area
                        # temporal correlation
                        key_temp_cor = "temp_cor_" + var_sim + f"{sc1}"
                        temp_cor = de.calc_temp_cor(obs_vals, sim_vals)
                        df_params_metrics.loc[nrow, key_temp_cor] = temp_cor
                        # diagnostic efficiency
                        key_de = "DE_" + var_sim + f"{sc1}"
                        df_params_metrics.loc[nrow, key_de] = de.calc_de(obs_vals, sim_vals)
                        # relative bias
                        brel = de.calc_brel(obs_vals, sim_vals)
                        # total bias
                        key_b_tot = "b_tot_" + var_sim + f"{sc1}"
                        b_tot = de.calc_bias_tot(brel)
                        df_params_metrics.loc[nrow, key_b_tot] = b_tot
                        # bias of lower exceedance probability
                        key_b_hf = "b_hf_" + var_sim + f"{sc1}"
                        b_hf = de.calc_bias_hf(brel)
                        df_params_metrics.loc[nrow, key_b_hf] = b_hf
                        # error contribution of higher exceedance probability
                        key_err_hf = "err_hf_" + var_sim + f"{sc1}"
                        err_hf = de.calc_err_hf(b_hf, b_tot)
                        df_params_metrics.loc[nrow, key_err_hf] = err_hf
                        # bias of higher exceedance probability
                        key_b_lf = "b_lf_" + var_sim + f"{sc1}"
                        b_lf = de.calc_bias_lf(brel)
                        df_params_metrics.loc[nrow, key_b_lf] = b_lf
                        # error contribution of lower exceedance probability
                        key_err_lf = "err_lf_" + var_sim + f"{sc1}"
                        err_lf = de.calc_err_hf(b_lf, b_tot)
                        df_params_metrics.loc[nrow, key_err_lf] = err_lf
                        # direction of bias
                        key_b_dir = "b_dir_" + var_sim + f"{sc1}"
                        b_dir = de.calc_bias_dir(brel_res)
                        df_params_metrics.loc[nrow, key_b_dir] = b_dir
                        # slope of bias
                        key_b_slope = "b_slope_" + var_sim + f"{sc1}"
                        b_slope = de.calc_bias_slope(b_area, b_dir)
                        df_params_metrics.loc[nrow, key_b_slope] = b_slope
                        # (y, x) trigonometric inverse tangent
                        key_phi = "phi_" + var_sim + f"{sc1}"
                        df_params_metrics.loc[nrow, key_phi] = de.calc_phi(brel_mean, b_slope)

                    else:
                        # skip first seven days for warmup
                        df_eval.loc[:"1997-01-07", :] = onp.nan
                        df_eval_monthly = df_eval.resample("ME").sum()
                        df_eval = df_eval_monthly.dropna()
                        obs_vals = df_eval.loc[:, "obs"].values.astype(float)
                        sim_vals = df_eval.loc[:, "sim"].values.astype(float)
                        key_kge = "KGE_" + var_sim + f"{sc1}"
                        df_params_metrics.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals, sim_vals)
                        key_kge_alpha = "KGE_alpha_" + var_sim + f"{sc1}"
                        df_params_metrics.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals, sim_vals)
                        key_kge_beta = "KGE_beta_" + var_sim + f"{sc1}"
                        df_params_metrics.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals, sim_vals)
                        key_r = "r_" + var_sim + f"{sc1}"
                        df_params_metrics.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals, sim_vals)
                        key_relsum = "rel_sum_" + var_sim + f"{sc1}"
                        df_params_metrics.loc[nrow, key_relsum] = onp.sum(sim_vals) / onp.sum(obs_vals)
                        cond0 = df_eval["obs"] == 0
                        if cond0.any():
                            # number of data points
                            N_obs = len(df_eval.index)
                            # simulations and observations for which observed
                            # values are exclusively zero
                            df_obs0_sim = df_eval.loc[cond0, :]
                            N_obs0 = (df_obs0_sim["obs"] == 0).sum()
                            N_sim0 = (df_obs0_sim["sim"] == 0).sum()
                            # share of observations with zero values
                            key_p0 = "p0_" + var_sim + f"{sc1}"
                            df_params_metrics.loc[nrow, key_p0] = N_obs0 / N_obs
                            # agreement of zero values
                            N_obs0 = (df_obs0_sim["obs"] == 0).sum()
                            N_sim0 = (df_obs0_sim["sim"] == 0).sum()
                            ioa0 = N_sim0 / N_obs0
                            key_ioa0 = "ioa0_" + var_sim + f"{sc1}"
                            df_params_metrics.loc[nrow, key_ioa0] = ioa0
                            # mean absolute error from observations with zero values
                            obs0_vals = df_obs0_sim.loc[:, "obs"].values.astype(float)
                            sim0_vals = df_obs0_sim.loc[:, "sim"].values.astype(float)
                            key_mae0 = "MAE0_" + var_sim + f"{sc1}"
                            df_params_metrics.loc[nrow, key_mae0] = eval_utils.calc_mae(obs0_vals, sim0_vals)
                            # peak difference from observations with zero values
                            key_pdiff0 = "PDIFF0_" + var_sim + f"{sc1}"
                            df_params_metrics.loc[nrow, key_pdiff0] = onp.max(sim0_vals)
                            # simulations and observations with non-zero values
                            cond_no0 = df_eval["obs"] > 0
                            df_obs_sim_no0 = df_eval.loc[cond_no0, :]
                            obs_vals_no0 = df_obs_sim_no0.loc[:, "obs"].values.astype(float)
                            sim_vals_no0 = df_obs_sim_no0.loc[:, "sim"].values.astype(float)
                            # number of data with non-zero observations
                            N_no0 = len(df_obs_sim_no0.index)
                            # mean absolute relative error
                            key_mare = "MARE_" + var_sim + f"{sc1}"
                            df_params_metrics.loc[nrow, key_mare] = eval_utils.calc_mare(obs_vals_no0, sim_vals_no0)
                            # mean relative bias
                            key_brel_mean = "brel_mean_" + var_sim + f"{sc1}"
                            brel_mean = de.calc_brel_mean(obs_vals_no0, sim_vals_no0)
                            df_params_metrics.loc[nrow, key_brel_mean] = brel_mean
                            # residual relative bias
                            brel_res = de.calc_brel_res(obs_vals_no0, sim_vals_no0)
                            # area of relative residual bias
                            key_b_area = "b_area_" + var_sim + f"{sc1}"
                            b_area = de.calc_bias_area(brel_res)
                            df_params_metrics.loc[nrow, key_b_area] = b_area
                            # temporal correlation
                            key_temp_cor = "temp_cor_" + var_sim + f"{sc1}"
                            temp_cor = de.calc_temp_cor(obs_vals_no0, sim_vals_no0)
                            df_params_metrics.loc[nrow, key_temp_cor] = temp_cor
                            # diagnostic efficiency
                            key_de = "DE_" + var_sim + f"{sc1}"
                            df_params_metrics.loc[nrow, key_de] = de.calc_de(obs_vals_no0, sim_vals_no0)
                            # relative bias
                            brel = de.calc_brel(obs_vals, sim_vals)
                            # total bias
                            key_b_tot = "b_tot_" + var_sim + f"{sc1}"
                            b_tot = de.calc_bias_tot(brel)
                            df_params_metrics.loc[nrow, key_b_tot] = b_tot
                            # bias of lower exceedance probability
                            key_b_hf = "b_hf_" + var_sim + f"{sc1}"
                            b_hf = de.calc_bias_hf(brel)
                            df_params_metrics.loc[nrow, key_b_hf] = b_hf
                            # error contribution of higher exceedance probability
                            key_err_hf = "err_hf_" + var_sim + f"{sc1}"
                            err_hf = de.calc_err_hf(b_hf, b_tot)
                            df_params_metrics.loc[nrow, key_err_hf] = err_hf
                            # bias of higher exceedance probability
                            key_b_lf = "b_lf_" + var_sim + f"{sc1}"
                            b_lf = de.calc_bias_lf(brel)
                            df_params_metrics.loc[nrow, key_b_lf] = b_lf
                            # error contribution of lower exceedance probability
                            key_err_lf = "err_lf_" + var_sim + f"{sc1}"
                            err_lf = de.calc_err_hf(b_lf, b_tot)
                            df_params_metrics.loc[nrow, key_err_lf] = err_lf
                            # direction of bias
                            key_b_dir = "b_dir_" + var_sim + f"{sc1}"
                            b_dir = de.calc_bias_dir(brel_res)
                            df_params_metrics.loc[nrow, key_b_dir] = b_dir
                            # slope of bias
                            key_b_slope = "b_slope_" + var_sim + f"{sc1}"
                            b_slope = de.calc_bias_slope(b_area, b_dir)
                            df_params_metrics.loc[nrow, key_b_slope] = b_slope
                            # (y, x) trigonometric inverse tangent
                            key_phi = "phi_" + var_sim + f"{sc1}"
                            df_params_metrics.loc[nrow, key_phi] = de.calc_phi(brel_mean, b_slope)
                            # combined diagnostic efficiency
                            key_de0 = "DE0_" + var_sim + f"{sc1}"
                            df_params_metrics.loc[nrow, key_de0] = (N_no0 / N_obs) * df_params_metrics.loc[
                                nrow, key_de
                            ] + (N_obs0 / N_obs) * ioa0
                        else:
                            # share of observations with zero values
                            key_p0 = "p0_" + var_sim + f"{sc1}"
                            df_params_metrics.loc[nrow, key_p0] = 0
                            # mean absolute relative error
                            key_mare = "MARE_" + var_sim + f"{sc1}"
                            df_params_metrics.loc[nrow, key_mare] = eval_utils.calc_mare(obs_vals, sim_vals)
                            # mean relative bias
                            key_brel_mean = "brel_mean_" + var_sim + f"{sc1}"
                            brel_mean = de.calc_brel_mean(obs_vals, sim_vals)
                            df_params_metrics.loc[nrow, key_brel_mean] = brel_mean
                            # residual relative bias
                            brel_res = de.calc_brel_res(obs_vals, sim_vals)
                            # area of relative residual bias
                            key_b_area = "b_area_" + var_sim + f"{sc1}"
                            b_area = de.calc_bias_area(brel_res)
                            df_params_metrics.loc[nrow, key_b_area] = b_area
                            # temporal correlation
                            key_temp_cor = "temp_cor_" + var_sim + f"{sc1}"
                            temp_cor = de.calc_temp_cor(obs_vals, sim_vals)
                            df_params_metrics.loc[nrow, key_temp_cor] = temp_cor
                            # diagnostic efficiency
                            key_de = "DE_" + var_sim + f"{sc1}"
                            df_params_metrics.loc[nrow, key_de] = de.calc_de(obs_vals, sim_vals)
                            # relative bias
                            brel = de.calc_brel(obs_vals, sim_vals)
                            # total bias
                            key_b_tot = "b_tot_" + var_sim + f"{sc1}"
                            b_tot = de.calc_bias_tot(brel)
                            df_params_metrics.loc[nrow, key_b_tot] = b_tot
                            # bias of lower exceedance probability
                            key_b_hf = "b_hf_" + var_sim + f"{sc1}"
                            b_hf = de.calc_bias_hf(brel)
                            df_params_metrics.loc[nrow, key_b_hf] = b_hf
                            # error contribution of higher exceedance probability
                            key_err_hf = "err_hf_" + var_sim + f"{sc1}"
                            err_hf = de.calc_err_hf(b_hf, b_tot)
                            df_params_metrics.loc[nrow, key_err_hf] = err_hf
                            # bias of higher exceedance probability
                            key_b_lf = "b_lf_" + var_sim + f"{sc1}"
                            b_lf = de.calc_bias_lf(brel)
                            df_params_metrics.loc[nrow, key_b_lf] = b_lf
                            # error contribution of lower exceedance probability
                            key_err_lf = "err_lf_" + var_sim + f"{sc1}"
                            err_lf = de.calc_err_hf(b_lf, b_tot)
                            df_params_metrics.loc[nrow, key_err_lf] = err_lf
                            # direction of bias
                            key_b_dir = "b_dir_" + var_sim + f"{sc1}"
                            b_dir = de.calc_bias_dir(brel_res)
                            df_params_metrics.loc[nrow, key_b_dir] = b_dir
                            # slope of bias
                            key_b_slope = "b_slope_" + var_sim + f"{sc1}"
                            b_slope = de.calc_bias_slope(b_area, b_dir)
                            df_params_metrics.loc[nrow, key_b_slope] = b_slope
                            # (y, x) trigonometric inverse tangent
                            key_phi = "phi_" + var_sim + f"{sc1}"
                            df_params_metrics.loc[nrow, key_phi] = de.calc_phi(brel_mean, b_slope)

                # avoid defragmentation of DataFrame
                click.echo(f"{var_sim}: {nrow}")
                df_params_metrics = df_params_metrics.copy()
        # Calculate multi-objective metric
        for sc, sc1 in zip([0, 1, 2, 3], ["", "dry", "normal", "wet"]):
            df_params_metrics.loc[:, f"E_multi{sc1}"] = (
                0.2 * df_params_metrics.loc[:, f"E_dS{sc1}"]
                + 0.4 * df_params_metrics.loc[:, f"KGE_aet{sc1}"]
                + 0.4 * df_params_metrics.loc[:, f"KGE_q_ss{sc1}"]
            )
            df_params_metrics.loc[:, f"KGE_multi{sc1}"] = (
                0.2 * df_params_metrics.loc[:, f"KGE_dS{sc1}"]
                + 0.4 * df_params_metrics.loc[:, f"KGE_aet{sc1}"]
                + 0.4 * df_params_metrics.loc[:, f"KGE_q_ss{sc1}"]
            )

        # write .txt-file
        file = base_path_figs / "params_metrics_monthly.txt"
        df_params_metrics.to_csv(file, header=True, index=False, sep="\t")
    return


if __name__ == "__main__":
    main()
