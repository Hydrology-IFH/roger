from pathlib import Path
import os
import glob
import h5netcdf
import datetime
import matplotlib.pyplot as plt
import xarray as xr
from cftime import num2date
import pandas as pd
import roger.tools.evaluation as eval_utils
import numpy as onp
import roger
import click


@click.option("-td", "--tmp-dir", type=str, default=None)
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
    tm_structures = ['complete-mixing', 'piston',
                     'preferential', 'advection-dispersion',
                     'time-variant preferential',
                     'time-variant advection-dispersion']
    for tm_structure in tm_structures:
        tms = tm_structure.replace(" ", "_")
        path = str(base_path / f"SVATTRANSPORT_{tms}.*.nc")
        diag_files = glob.glob(path)
        states_tm_file = base_path / "states_tm_monte_carlo_reverse.nc"
        with h5netcdf.File(states_tm_file, 'a', decode_vlen_strings=False) as f:
            if tm_structure not in list(f.groups.keys()):
                f.create_group(tm_structure)
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='RoGeR transport model Monte Carlo simulations reverse at Rietholzbach lysimeter site',
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

    # load simulation
    states_hm_file = base_path / "states_hm_monte_carlo.nc"
    ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")

    # load observations (measured data)
    path_obs = base_path.parent / "observations" / "rietholzbach_lysimeter.nc"
    ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")


    tm_structures = ['complete-mixing', 'piston',
                     'preferential', 'advection-dispersion',
                     'time-variant preferential',
                     'time-variant advection-dispersion']
    for tm_structure in tm_structures:
        tms = tm_structure.replace(" ", "_")

        # load simulation
        states_tm_file = base_path / "states_tm_monte_carlo_reverse.nc"
        ds_sim_tm = xr.open_dataset(states_tm_file, f"{tm_structure}", engine="h5netcdf")

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
        for param in ['dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks']:
            df_params_eff.loc[:, param] = ds_sim_hm[param].isel(y=0).values

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

        # write composite sample to output file
        states_tm_file = base_path / "states_tm_monte_carlo_reverse.nc"
        with h5netcdf.File(states_tm_file, 'a', decode_vlen_strings=False) as f:
            v = f.groups[tm_structure].create_variable('d18O_perc_cs', ('x', 'y', 'Time'), float)
            v[:, :, :] = d18O_perc_cs
            v.attrs.update(long_name="composite sample of d18O in percolation",
                           units="permil")

        # write to .txt
        file = base_path_results / f"params_eff_{tm_structure}.txt"
        df_params_eff.to_csv(file, header=True, index=False, sep="\t")
        return


if __name__ == "__main__":
    main()
