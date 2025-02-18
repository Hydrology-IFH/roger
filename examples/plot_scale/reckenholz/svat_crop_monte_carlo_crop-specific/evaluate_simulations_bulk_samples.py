import os
from pathlib import Path
from cftime import num2date
import xarray as xr
import pandas as pd
from de import de
import numpy as onp
import click
import roger.tools.evaluation as eval_utils
import h5netcdf

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
        base_path_output = base_path / "output" / "svat_crop_monte_carlo_crop-specific"
        if not os.path.exists(base_path_output):
            os.mkdir(base_path_output)

        # load simulation
        states_hm_mc_file = base_path_output / f"SVATCROP_{lys_experiment}.nc"
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
        
        nx = len(df_params_metrics.index)
        idx = ds_sim.date.values  # time index
        df_idx_bs = pd.DataFrame(index=date_obs, columns=['sol'])
        df_idx_bs.loc[:, 'sol'] = ds_obs['NO3_PERC'].isel(x=0, y=0).values
        idx_bs = df_idx_bs['sol'].dropna().index
        perc_bs_sim = onp.zeros((nx, 1, len(idx)))
        perc_bs_obs = onp.zeros((nx, 1, len(idx)))

        for nrow in range(nx):
            # calculate simulated nitrate bulk samples
            sample_no = pd.DataFrame(index=idx_bs, columns=['sample_no'])
            sample_no['sample_no'] = range(len(sample_no.index))
            df_perc_bs = pd.DataFrame(index=idx, columns=['perc', 'NO3_mass'])
            df_perc_bs['perc_sim'] = ds_sim['q_ss'].isel(x=nrow, y=0).values
            df_perc_bs.loc[df_perc_bs.index[1]:, 'perc_obs'] = ds_obs['PERC'].isel(x=0, y=0).values
            df_perc_bs = df_perc_bs.join(sample_no)
            df_perc_bs.loc[:, 'sample_no'] = df_perc_bs.loc[:, 'sample_no'].bfill(limit=14)
            perc_sim_sum = df_perc_bs.groupby(['sample_no']).sum().loc[:, 'perc_sim']
            perc_obs_sum = df_perc_bs.groupby(['sample_no']).sum().loc[:, 'perc_obs']
            sample_no['perc_sim_sum'] = perc_sim_sum.values
            sample_no['perc_obs_sum'] = perc_obs_sum.values
            df_perc_bs = df_perc_bs.join(sample_no['perc_sim_sum'])
            df_perc_bs = df_perc_bs.join(sample_no['perc_obs_sum'])
            # volume of observed bulk samples
            perc_bs_obs[nrow, 0, :] = df_perc_bs.loc[:, 'perc_obs_sum'].values.astype(float)
            # volume of simulated bulk samples
            perc_bs_sim[nrow, 0, :] = df_perc_bs.loc[:, 'perc_sim_sum'].values.astype(float)

            sim_vals = perc_bs_sim[nrow, 0, :]
            sim_vals = onp.where(sim_vals == 0, onp.nan, sim_vals)
            obs_vals = perc_bs_obs[nrow, 0, 1:]
            obs_vals = onp.where(obs_vals == 0, onp.nan, obs_vals)
            df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
            df_obs.loc[:, 'obs'] = obs_vals

            # calculate metrics
            var_sim = "q_ss"
            df_eval = eval_utils.join_obs_on_sim(date_sim, sim_vals, df_obs)
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

        # write to .txt
        file = base_path / "output" / "svat_crop_monte_carlo_crop-specific" / f"params_eff_{lys_experiment}_bulk_samples.txt"
        df_params_metrics.to_csv(file, header=True, index=False, sep="\t")

        file = base_path / "output" / "svat_crop_monte_carlo_crop-specific" / f"params_eff_{lys_experiment}_bulk_samples.csv"
        df_params_metrics.to_csv(file, header=True, index=False, sep=";")

        # add simulated bulk samples to the dataset
        ds_sim.close()
        del(ds_sim)
        with h5netcdf.File(states_hm_mc_file, "a", decode_vlen_strings=False) as f:
            try:
                v = f.create_variable("q_ss_bs", ("x", "y", "Time"), float, compression="gzip", compression_opts=1)
                v[:, :, :] = perc_bs_sim
                v.attrs.update(long_name="Volume of simulated bulk samples", units="mm")
            except ValueError:
                var_obj = f.variables.get("q_ss_bs")
                var_obj[:, :, :] = perc_bs_sim
            try:
                v = f.create_variable("q_ss_bs_obs", ("x", "y", "Time"), float, compression="gzip", compression_opts=1)
                v[:, :, :] = perc_bs_obs
                v.attrs.update(long_name="Volume of measured bulk samples", units="mm")
            except ValueError:
                var_obj = f.variables.get("q_ss_bs_obs")
                var_obj[:, :, :] = perc_bs_obs
    return


if __name__ == "__main__":
    main()
