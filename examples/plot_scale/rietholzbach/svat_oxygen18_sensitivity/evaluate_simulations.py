from pathlib import Path
import os
import h5netcdf
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import click
import roger.tools.evaluation as eval_utils
import matplotlib as mpl
import seaborn as sns
mpl.use("agg")
sns.set_style("ticks")

def delta_to_conc(delta_iso):
    """Calculate isotope concentration from isotope ratio
    """
    return 2005.2e-6*(delta_iso/1000.+1.)/(1.+(delta_iso/1000.+1.)*2005.2e-6)


@click.option("-tms", "--transport-model-structure", type=click.Choice(['complete-mixing', 'piston', 'advection-dispersion-power', 'time-variant_advection-dispersion-power']), default='advection-dispersion-power')
@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(transport_model_structure, tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent
    tms = transport_model_structure.replace("_", " ")
    # directory of results
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

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
    theta_lower = df_thetap.loc[:, f'theta_avg{window}'].quantile(0.1)
    theta_upper = df_thetap.loc[:, f'theta_avg{window}'].quantile(0.9)
    cond1 = (df_thetap[f'theta_avg{window}'] < theta_lower)
    cond2 = (df_thetap[f'theta_avg{window}'] >= theta_lower) & (df_thetap[f'theta_avg{window}'] < theta_upper)
    cond3 = (df_thetap[f'theta_avg{window}'] >= theta_upper)
    df_thetap.loc[cond1, 'sc'] = 1  # dry
    df_thetap.loc[cond2, 'sc'] = 2  # normal
    df_thetap.loc[cond3, 'sc'] = 3  # wet

    click.echo(f'Calculate metrics for {tms} ...')

    # load hydrologic simulations
    states_hm_file = Path(__file__).parent / "input" / f"states_hm_saltelli_for_{transport_model_structure}.nc"
    ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")
    # assign date
    days_sim_hm = (ds_sim_hm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim_hm = num2date(days_sim_hm, units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))

    # load transport simulations
    states_tm_file = base_path_output / f"states_{transport_model_structure}_saltelli.nc"
    ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf", decode_times=False)
    # assign date
    date_sim_tm = num2date(ds_sim_tm['Time'].values, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))
    # load parameters
    params_file = base_path / "params_saltelli.nc"
    ds_params = xr.open_dataset(params_file, engine="h5netcdf", decode_times=False, group=tms)

    # DataFrame with sampled model parameters and the corresponding metrics
    nx = ds_sim_tm.dims['x']  # number of rows
    ny = ds_sim_tm.dims['y']  # number of columns
    df_params_metrics = pd.DataFrame(index=range(nx * ny))
    # sampled model parameters
    if tms == "complete-mixing":
        df_params_metrics.loc[:, 'c1_mak'] = ds_params["c1_mak"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'c2_mak'] = ds_params["c2_mak"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'dmpv'] = ds_params["dmpv"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'lmpv'] = ds_params["lmpv"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'theta_eff'] = ds_params["theta_eff"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'frac_lp'] = ds_params["frac_lp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'frac_fp'] = 1 - ds_params["frac_lp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'theta_ac'] = df_params_metrics.loc[:, 'frac_lp'] * df_params_metrics.loc[:, 'theta_eff']
        df_params_metrics.loc[:, 'theta_ufc'] = df_params_metrics.loc[:, 'frac_fp'] * df_params_metrics.loc[:, 'theta_eff']
        df_params_metrics.loc[:, 'theta_pwp'] = ds_params["theta_pwp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'ks'] = ds_params["ks"].isel(y=0).values.flatten()
    elif tms == "piston":
        df_params_metrics.loc[:, 'c1_mak'] = ds_params["c1_mak"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'c2_mak'] = ds_params["c2_mak"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'dmpv'] = ds_params["dmpv"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'lmpv'] = ds_params["lmpv"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'theta_eff'] = ds_params["theta_eff"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'frac_lp'] = ds_params["frac_lp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'frac_fp'] = 1 - ds_params["frac_lp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'theta_ac'] = df_params_metrics.loc[:, 'frac_lp'] * df_params_metrics.loc[:, 'theta_eff']
        df_params_metrics.loc[:, 'theta_ufc'] = df_params_metrics.loc[:, 'frac_fp'] * df_params_metrics.loc[:, 'theta_eff']
        df_params_metrics.loc[:, 'theta_pwp'] = ds_params["theta_pwp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'ks'] = ds_params["ks"].isel(y=0).values.flatten()
    elif tms == "advection-dispersion-power":
        df_params_metrics.loc[:, 'c1_mak'] = ds_params["c1_mak"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'c2_mak'] = ds_params["c2_mak"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'dmpv'] = ds_params["dmpv"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'lmpv'] = ds_params["lmpv"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'theta_eff'] = ds_params["theta_eff"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'frac_lp'] = ds_params["frac_lp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'frac_fp'] = 1 - ds_params["frac_lp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'theta_ac'] = df_params_metrics.loc[:, 'frac_lp'] * df_params_metrics.loc[:, 'theta_eff']
        df_params_metrics.loc[:, 'theta_ufc'] = df_params_metrics.loc[:, 'frac_fp'] * df_params_metrics.loc[:, 'theta_eff']
        df_params_metrics.loc[:, 'theta_pwp'] = ds_params["theta_pwp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'ks'] = ds_params["ks"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'k_transp'] = ds_params["k_transp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'k_q_rz'] = ds_params["k_q_rz"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'k_q_ss'] = ds_params["k_q_ss"].isel(y=0).values.flatten()
    elif tms == "time-variant advection-dispersion-power":
        df_params_metrics.loc[:, 'c1_mak'] = ds_params["c1_mak"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'c2_mak'] = ds_params["c2_mak"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'dmpv'] = ds_params["dmpv"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'lmpv'] = ds_params["lmpv"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'theta_eff'] = ds_params["theta_eff"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'frac_lp'] = ds_params["frac_lp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'frac_fp'] = 1 - ds_params["frac_lp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'theta_ac'] = df_params_metrics.loc[:, 'frac_lp'] * df_params_metrics.loc[:, 'theta_eff']
        df_params_metrics.loc[:, 'theta_ufc'] = df_params_metrics.loc[:, 'frac_fp'] * df_params_metrics.loc[:, 'theta_eff']
        df_params_metrics.loc[:, 'theta_pwp'] = ds_params["theta_pwp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'ks'] = ds_params["ks"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'k1_transp'] = ds_params["c1_transp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'k2_transp'] = ds_params["c1_transp"].isel(y=0).values.flatten() + ds_params["c1_transp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'k1_q_rz'] = ds_params["c1_q_rz"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'k2_q_rz'] = ds_params["c1_q_rz"].isel(y=0).values.flatten() + ds_params["c2_q_rz"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'k1_q_ss'] = ds_params["c1_q_ss"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, 'k2_q_ss'] = ds_params["c1_q_ss"].isel(y=0).values.flatten() + ds_params["c2_q_ss"].isel(y=0).values.flatten()

    # compare observations and simulations
    ncol = 0
    idx = ds_sim_tm.Time.values  # time index
    d18O_perc_bs = onp.zeros((ds_sim_tm.dims['x'], 1, len(idx)))
    df_idx_bs = pd.DataFrame(index=date_obs, columns=['sol'])
    df_idx_bs.loc[:, 'sol'] = ds_obs['d18O_PERC'].isel(x=0, y=0).values
    idx_bs = df_idx_bs['sol'].dropna().index
    for nrow in range(ds_sim_tm.dims['x']):
        # calculate simulated oxygen-18 bulk sample
        df_perc_18O_obs = pd.DataFrame(index=date_obs, columns=['perc_obs', 'd18O_perc_obs'])
        df_perc_18O_obs.loc[:, 'perc_obs'] = ds_obs['PERC'].isel(x=0, y=0).values
        df_perc_18O_obs.loc[:, 'd18O_perc_obs'] = ds_obs['d18O_PERC'].isel(x=0, y=0).values
        sample_no = pd.DataFrame(index=idx_bs, columns=['sample_no'])
        sample_no = sample_no.loc['1997':'2007']
        sample_no['sample_no'] = range(len(sample_no.index))
        df_perc_18O_sim = pd.DataFrame(index=date_sim_tm, columns=['perc_sim', 'd18O_perc_sim'])
        df_perc_18O_sim['perc_sim'] = ds_sim_hm['q_ss'].isel(x=nrow, y=0).values
        C_iso_q_ss = ds_sim_tm['C_iso_q_ss'].isel(x=nrow, y=0).values
        df_perc_18O_sim['d18O_perc_sim'] = onp.where(C_iso_q_ss < -20, onp.nan, C_iso_q_ss)
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
        cond = (df_perc_18O_sim['d18O_sample'] == 0)
        df_perc_18O_sim.loc[cond, 'd18O_sample'] = onp.NaN
        d18O_perc_bs[nrow, 0, :] = df_perc_18O_sim.loc[:, 'd18O_sample'].values
        # calculate observed oxygen-18 bulk sample
        df_perc_18O_obs.loc[:, 'd18O_perc_bs'] = df_perc_18O_obs['d18O_perc_obs'].fillna(method='bfill', limit=14)
        df_perc_18O_obs.loc[:, 'd18O_perc_mass'] = df_perc_18O_obs['perc_obs'].values * delta_to_conc(df_perc_18O_obs['d18O_perc_bs'].values)

        perc_sample_sum_obs = df_perc_18O_sim.join(df_perc_18O_obs).groupby(['sample_no']).sum().loc[:, 'perc_obs']
        sample_no['perc_obs_sum'] = perc_sample_sum_obs.values
        df_perc_18O_sim = df_perc_18O_sim.join(sample_no['perc_obs_sum'])
        df_perc_18O_sim.loc[:, 'perc_obs_sum'] = df_perc_18O_sim.loc[:, 'perc_obs_sum'].fillna(method='bfill', limit=14)
        df_perc_18O_sim.loc[:, 'd18O_perc_mass'] = df_perc_18O_sim['perc_sim'].values * delta_to_conc(df_perc_18O_sim.loc[:, 'd18O_sample'].fillna(method='bfill', limit=14).values)

        # join observations on simulations
        for sc, sc1 in zip([0, 1, 2, 3], ['', 'dry', 'normal', 'wet']):
            obs_vals = df_perc_18O_obs.loc[:, 'd18O_perc_mass'].values
            sim_vals = df_perc_18O_sim.loc[:, 'd18O_perc_mass'].values
            df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
            df_obs.loc[:, 'obs'] = obs_vals
            df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)

            if sc > 0:
                df_rows = pd.DataFrame(index=df_eval.index).join(df_thetap.loc["2006":"2007", :])
                rows = (df_rows['sc'].values == sc)
                df_eval = df_eval.loc[rows, :]
            df_eval = df_eval.dropna()

            # calculate metrics
            if len(df_eval.index) > 10:
                var_sim = 'M_iso_q_ss'
                obs_vals = df_eval.loc[:, 'obs'].values
                sim_vals = df_eval.loc[:, 'sim'].values
                key_kge = f'KGE_{var_sim}{sc1}'
                kge_val = eval_utils.calc_kge(obs_vals, sim_vals)
                if kge_val < -1:
                    kge_val = -1
                df_params_metrics.loc[nrow, key_kge] = kge_val
                key_kge_alpha = f'KGE_alpha_{var_sim}{sc1}'
                df_params_metrics.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals, sim_vals)
                key_kge_beta = f'KGE_beta_{var_sim}{sc1}'
                df_params_metrics.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals, sim_vals)
                key_r = f'r_{var_sim}{sc1}'
                df_params_metrics.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals, sim_vals)
            # average age metrics
            vars_sim = ['tt50_transp', 'ttavg_transp',
                        'tt50_q_ss', 'ttavg_q_ss',
                        'rt50_s', 'rtavg_s']
            for var_sim in vars_sim:
                df_eval = pd.DataFrame(index=idx)
                df_eval.loc[:, 'sim'] = ds_sim_tm[var_sim].isel(x=nrow, y=ncol).values
                if sc > 0:
                    df_rows = pd.DataFrame(index=df_eval.index).join(df_thetap)
                    rows = (df_rows['sc'].values == sc)
                    df_eval = df_eval.loc[rows, :]
                df_eval = df_eval.dropna()
                df_params_metrics.loc[nrow, f'{var_sim}{sc1}'] = onp.nanmean(df_eval.loc[:, 'sim'].values)

        # avoid defragmentation of DataFrame
        click.echo(f'{nrow}')
        df_params_metrics = df_params_metrics.copy()

    # write to .txt
    file = base_path_figs / f"params_metrics_{transport_model_structure}.txt"
    df_params_metrics.to_csv(file, header=True, index=False, sep="\t")

    # write simulated bulk sample to output file
    ds_sim_tm = ds_sim_tm.load()
    ds_sim_tm = ds_sim_tm.close()
    states_tm_file = base_path_output / f"states_{transport_model_structure}_saltelli.nc"
    with h5netcdf.File(states_tm_file, 'a', decode_vlen_strings=False) as f:
        try:
            v = f.create_variable('d18O_perc_bs', ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
            v.attrs.update(long_name="bulk sample of d18O in percolation",
                           units="permil")
            v[:, :, :] = d18O_perc_bs
        except ValueError:
            v = f.get('d18O_perc_bs')
            v[:, :, :] = d18O_perc_bs
    return


if __name__ == "__main__":
    main()
