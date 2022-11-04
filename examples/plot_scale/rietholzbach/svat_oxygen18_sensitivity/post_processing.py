from pathlib import Path
import os
import glob
import h5netcdf
import xarray as xr
import datetime
from cftime import num2date
import pandas as pd
import numpy as onp
import click
import roger
import roger.tools.evaluation as eval_utils
import matplotlib as mpl
import seaborn as sns
mpl.use("agg")
sns.set_style("ticks")


@click.option("--sas-solver", type=click.Choice(['RK4', 'Euler', 'deterministic']), default='deterministic')
@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(sas_solver, tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent
    age_max = "age_max_11"
    # directory of results
    base_path_results = base_path / "results"
    if not os.path.exists(base_path_results):
        os.mkdir(base_path_results)
    base_path_results = base_path / "results" / sas_solver
    if not os.path.exists(base_path_results):
        os.mkdir(base_path_results)
    base_path_results = base_path / "results" / sas_solver / age_max
    if not os.path.exists(base_path_results):
        os.mkdir(base_path_results)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)
    base_path_figs = base_path / "figures" / sas_solver
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)
    base_path_figs = base_path / "figures" / sas_solver / age_max
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    # merge diagnostics into single file
    tm_structures = ['advection-dispersion',
                     'time-variant advection-dispersion']
    diagnostics = ['average',
                   'constant',
                   'maximum']
    for tm_structure in tm_structures:
        if tm_structure in ['advection-dispersion']:
            nsamples = 1024 * 5
        elif tm_structure in ['time-variant advection-dispersion']:
            nsamples = 1024 * 8
        x1x2 = onp.arange(0, nsamples, 500).tolist()
        if nsamples not in x1x2:
            x1x2.append(nsamples)
        for diagnostic in diagnostics:
            tms = tm_structure.replace(" ", "_")
            path = str(base_path / sas_solver / age_max / f"SVATTRANSPORT_{tms}_{sas_solver}_*_*.{diagnostic}.nc")
            diag_files = glob.glob(path)
            if diag_files:
                diag_file = base_path / sas_solver / age_max / f"SVATTRANSPORT_{tms}_{sas_solver}.{diagnostic}.nc"
                if not os.path.exists(diag_file):
                    click.echo(f'Merge {diagnostic} of {tm_structure} ...')
                    # initial diagnostic file
                    with h5netcdf.File(diag_file, 'w', decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f'RoGeR {tm_structure} transport model Monte Carlo simulations at Rietholzbach lysimeter site',
                            institution='University of Freiburg, Chair of Hydrology',
                            references='',
                            comment='First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).',
                            model_structure=f'SVAT {tm_structure} transport model with free drainage',
                            sas_solver=f'{sas_solver}',
                            roger_version=f'{roger.__version__}'
                        )
                        # collect dimensions
                        with h5netcdf.File(diag_files[0], 'r', decode_vlen_strings=False) as df:
                            dict_dim = {'x': nsamples, 'y': 1, 'Time': len(df.variables['Time']), 'ages': len(df.variables['ages']), 'nages': len(df.variables['nages']), 'n_sas_params': len(df.variables['n_sas_params'])}
                            time_obj = df.variables.get('Time')
                            time_origin = time_obj.attrs["time_origin"]
                            time_unit = time_obj.attrs["units"]
                            time = onp.array(df.variables.get('Time'))
                        f.dimensions = dict_dim
                        v = f.create_variable('x', ('x',), float, compression="gzip", compression_opts=1)
                        v.attrs['long_name'] = 'model run'
                        v.attrs['units'] = ''
                        v[:] = onp.arange(dict_dim["x"])
                        v = f.create_variable('y', ('y',), float, compression="gzip", compression_opts=1)
                        v.attrs['long_name'] = ''
                        v.attrs['units'] = ''
                        v[:] = onp.arange(dict_dim["y"])
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
                        v[:] = onp.arange(0, dict_dim["n_sas_params"])
                        v = f.create_variable('Time', ('Time',), float, compression="gzip", compression_opts=1)
                        v.attrs.update(time_origin=time_origin,
                                       units=time_unit)
                        v[:] = time

                    with h5netcdf.File(diag_file, 'a', decode_vlen_strings=False) as f:
                        for i, dfs in enumerate(diag_files):
                            x1 = x1x2[i]
                            x2 = x1x2[i+1]
                            file = base_path / sas_solver / age_max / f"SVATTRANSPORT_{tms}_{sas_solver}_{x1}_{x2}.{diagnostic}.nc"
                            with h5netcdf.File(file, 'r', decode_vlen_strings=False) as df:
                                for var_sim in list(df.variables.keys()):
                                    var_obj = df.variables.get(var_sim)
                                    if var_sim not in list(dict_dim.keys()) and ('Time', 'y', 'x') == var_obj.dimensions and var_obj.shape[0] > 2:
                                        try:
                                            v = f.create_variable(var_sim, ('Time', 'y', 'x'), float, compression="gzip", compression_opts=1)
                                        except ValueError:
                                            v = f.get(var_sim)
                                        vals = onp.array(var_obj)
                                        v[:, :, x1:x2] = vals
                                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                                       units=var_obj.attrs["units"])
                                        del var_obj, vals
                                    elif var_sim not in list(dict_dim.keys()) and ('Time', 'y', 'x') == var_obj.dimensions and var_obj.shape[0] <= 2:
                                        try:
                                            v = f.create_variable(var_sim, ('Time', 'y', 'x'), float, compression="gzip", compression_opts=1)
                                        except ValueError:
                                            v = f.get(var_sim)
                                        vals = onp.array(var_obj)
                                        v[:, :, x1:x2] = vals
                                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                                       units=var_obj.attrs["units"])
                                        del var_obj, vals
                                    elif var_sim not in list(dict_dim.keys()) and ('Time', 'n_sas_params', 'y', 'x') == var_obj.dimensions:
                                        try:
                                            v = f.create_variable(var_sim, ('Time', 'n_sas_params', 'y', 'x'), float, compression="gzip", compression_opts=1)
                                        except ValueError:
                                            v = f.get(var_sim)
                                        vals = onp.array(var_obj)
                                        v[:, :, :, x1:x2] = vals
                                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                                       units=var_obj.attrs["units"])
                                        del var_obj, vals
                                    elif var_sim not in list(dict_dim.keys()) and ('Time', 'ages', 'y', 'x') == var_obj.dimensions:
                                        try:
                                            v = f.create_variable(var_sim, ('Time', 'ages', 'y', 'x'), float, compression="gzip", compression_opts=1)
                                        except ValueError:
                                            v = f.get(var_sim)
                                        vals = onp.array(var_obj)
                                        v[:, :, :, x1:x2] = vals
                                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                                       units=var_obj.attrs["units"])
                                        del var_obj, vals
                                    elif var_sim not in list(dict_dim.keys()) and ('Time', 'nages', 'y', 'x') == var_obj.dimensions:
                                        try:
                                            v = f.create_variable(var_sim, ('Time', 'nages', 'y', 'x'), float, compression="gzip", compression_opts=1)
                                        except ValueError:
                                            v = f.get(var_sim)
                                        vals = onp.array(var_obj)
                                        v[:, :, :, x1:x2] = vals
                                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                                       units=var_obj.attrs["units"])
                                        del var_obj, vals

    # merge results into single file
    for tm_structure in tm_structures:
        tms = tm_structure.replace(" ", "_")
        path = str(base_path / sas_solver / age_max / f"SVATTRANSPORT_{tms}_{sas_solver}.*.nc")
        diag_files = glob.glob(path)
        states_tm_file = base_path / f"states_{tms}_saltelli.nc"
        if not os.path.exists(states_tm_file):
            click.echo(f'Merge output files of {tm_structure} into {states_tm_file.as_posix()}')
            with h5netcdf.File(states_tm_file, 'w', decode_vlen_strings=False) as f:
                f.attrs.update(
                    date_created=datetime.datetime.today().isoformat(),
                    title=f'RoGeR {tm_structure} model Saltelli simulations at Rietholzbach lysimeter site',
                    institution='University of Freiburg, Chair of Hydrology',
                    references='',
                    comment='First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).',
                    model_structure=f'SVAT {tm_structure} model with free drainage',
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
                            v[:] = onp.arange(0, dict_dim["n_sas_params"])
                            v = f.create_variable('Time', ('Time',), float, compression="gzip", compression_opts=1)
                            var_obj = df.variables.get('Time')
                            v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                           units=var_obj.attrs["units"])
                            v[:] = time
                        for var_sim in list(df.variables.keys()):
                            var_obj = df.variables.get(var_sim)
                            if var_sim not in list(dict_dim.keys()) and ('Time', 'y', 'x') == var_obj.dimensions and var_obj.shape[0] > 2:
                                v = f.create_variable(var_sim, ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                                vals = onp.array(var_obj)
                                v[:, :, :] = vals.swapaxes(0, 2)
                                v.attrs.update(long_name=var_obj.attrs["long_name"],
                                               units=var_obj.attrs["units"])
                                del var_obj, vals
                            elif var_sim not in list(dict_dim.keys()) and ('Time', 'y', 'x') == var_obj.dimensions and var_obj.shape[0] <= 2:
                                v = f.create_variable(var_sim, ('x', 'y'), float, compression="gzip", compression_opts=1)
                                vals = onp.array(var_obj)
                                v[:, :] = vals.swapaxes(0, 2)[:, :, 0]
                                v.attrs.update(long_name=var_obj.attrs["long_name"],
                                               units=var_obj.attrs["units"])
                                del var_obj, vals
                            elif var_sim not in list(dict_dim.keys()) and ('Time', 'n_sas_params', 'y', 'x') == var_obj.dimensions:
                                v = f.create_variable(var_sim, ('x', 'y', 'n_sas_params'), float, compression="gzip", compression_opts=1)
                                vals = onp.array(var_obj)
                                vals = vals.swapaxes(0, 3)
                                vals = vals.swapaxes(1, 2)
                                v[:, :, :] = vals[:, :, :, 0]
                                v.attrs.update(long_name=var_obj.attrs["long_name"],
                                               units=var_obj.attrs["units"])
                                del var_obj, vals
                            elif var_sim not in list(dict_dim.keys()) and ('Time', 'ages', 'y', 'x') == var_obj.dimensions:
                                v = f.create_variable(var_sim, ('x', 'y', 'Time', 'ages'), float, compression="gzip", compression_opts=1)
                                vals = onp.array(var_obj)
                                vals = vals.swapaxes(0, 3)
                                vals = vals.swapaxes(1, 2)
                                vals = vals.swapaxes(2, 3)
                                v[:, :, :, :] = vals
                                v.attrs.update(long_name=var_obj.attrs["long_name"],
                                               units=var_obj.attrs["units"])
                                del var_obj, vals
                            elif var_sim not in list(dict_dim.keys()) and ('Time', 'nages', 'y', 'x') == var_obj.dimensions:
                                v = f.create_variable(var_sim, ('x', 'y', 'Time', 'nages'), float, compression="gzip", compression_opts=1)
                                vals = onp.array(var_obj)
                                vals = vals.swapaxes(0, 3)
                                vals = vals.swapaxes(1, 2)
                                vals = vals.swapaxes(2, 3)
                                v[:, :, :, :] = vals
                                v.attrs.update(long_name=var_obj.attrs["long_name"],
                                               units=var_obj.attrs["units"])
                                del var_obj, vals

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

    for tm_structure in tm_structures:
        click.echo(f'Calculate metrics for {tm_structure} ...')
        tms = tm_structure.replace(" ", "_")

        # load hydrologic simulations
        states_hm_file = base_path / f"states_hm_best_for_{tms}.nc"
        ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")
        # assign date
        days_sim_hm = (ds_sim_hm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date_sim_hm = num2date(days_sim_hm, units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))

        # load transport simulations
        states_tm_file = base_path / f"states_{tms}_saltelli.nc"
        ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf", decode_times=False)
        # assign date
        date_sim_tm = num2date(ds_sim_tm['Time'].values, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))

        # DataFrame with sampled model parameters and the corresponding metrics
        nx = ds_sim_tm.dims['x']  # number of rows
        ny = ds_sim_tm.dims['y']  # number of columns
        df_params_metrics = pd.DataFrame(index=range(nx * ny))
        # sampled model parameters
        if tm_structure == "advection-dispersion":
            df_params_metrics.loc[:, 'b_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=2).values.flatten()
            df_params_metrics.loc[:, 'a_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'a_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=1).values.flatten()
        elif tm_structure == "time-variant advection-dispersion":
            df_params_metrics.loc[:, 'b1_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=3).values.flatten()
            df_params_metrics.loc[:, 'b2_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=3).values.flatten() + ds_sim_tm["sas_params_transp"].isel(n_sas_params=4).values.flatten()
            df_params_metrics.loc[:, 'a1_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=3).values.flatten()
            df_params_metrics.loc[:, 'a2_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=3).values.flatten() + ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=4).values.flatten()
            df_params_metrics.loc[:, 'a1_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=3).values.flatten()
            df_params_metrics.loc[:, 'a2_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=3).values.flatten() + ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=4).values.flatten()

        # compare observations and simulations
        ncol = 0
        idx = ds_sim_tm.Time.values  # time index
        d18O_perc_bs = onp.zeros((nx, 1, len(idx)))
        df_idx_bs = pd.DataFrame(index=date_obs, columns=['sol'])
        df_idx_bs.loc[:, 'sol'] = ds_obs['d18O_PERC'].isel(x=0, y=0).values
        idx_bs = df_idx_bs['sol'].dropna().index
        for nrow in range(nx):
            # calculate simulated oxygen-18 bulk sample
            df_perc_18O_obs = pd.DataFrame(index=date_obs, columns=['perc_obs', 'd18O_perc_obs'])
            df_perc_18O_obs.loc[:, 'perc_obs'] = ds_obs['PERC'].isel(x=0, y=0).values
            df_perc_18O_obs.loc[:, 'd18O_perc_obs'] = ds_obs['d18O_PERC'].isel(x=0, y=0).values
            sample_no = pd.DataFrame(index=idx_bs, columns=['sample_no'])
            sample_no = sample_no.loc['1997':'2007']
            sample_no['sample_no'] = range(len(sample_no.index))
            df_perc_18O_sim = pd.DataFrame(index=date_sim_tm, columns=['perc_sim', 'd18O_perc_sim'])
            df_perc_18O_sim['perc_sim'] = ds_sim_hm['q_ss'].isel(y=0).values
            df_perc_18O_sim['d18O_perc_sim'] = ds_sim_tm['C_iso_q_ss'].isel(x=nrow, y=ncol).values
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
            d18O_perc_bs[nrow, ncol, :] = df_perc_18O_sim.loc[:, 'd18O_sample'].values
            # calculate observed oxygen-18 bulk sample
            df_perc_18O_obs.loc[:, 'd18O_perc_bs'] = df_perc_18O_obs['d18O_perc_obs'].fillna(method='bfill', limit=14)

            perc_sample_sum_obs = df_perc_18O_sim.join(df_perc_18O_obs).groupby(['sample_no']).sum().loc[:, 'perc_obs']
            sample_no['perc_obs_sum'] = perc_sample_sum_obs.values
            df_perc_18O_sim = df_perc_18O_sim.join(sample_no['perc_obs_sum'])
            df_perc_18O_sim.loc[:, 'perc_obs_sum'] = df_perc_18O_sim.loc[:, 'perc_obs_sum'].fillna(method='bfill', limit=14)

            # join observations on simulations
            for sc, sc1 in zip([0, 1, 2, 3], ['', 'dry', 'normal', 'wet']):
                obs_vals = ds_obs['d18O_PERC'].isel(x=0, y=0).values
                sim_vals = d18O_perc_bs[nrow, ncol, :]
                df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
                df_obs.loc[:, 'obs'] = obs_vals
                df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
                if sc > 0:
                    df_rows = pd.DataFrame(index=df_eval.index).join(df_thetap)
                    rows = (df_rows['sc'].values == sc)
                    df_eval = df_eval.loc[rows, :]
                df_eval = df_eval.dropna()

                # calculate metrics
                var_sim = 'C_iso_q_ss'
                obs_vals = df_eval.loc[:, 'obs'].values
                sim_vals = df_eval.loc[:, 'sim'].values
                key_kge = f'KGE_{var_sim}{sc1}'
                df_params_metrics.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals, sim_vals)
                key_kge_alpha = f'KGE_alpha_{var_sim}{sc1}'
                df_params_metrics.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals, sim_vals)
                key_kge_beta = f'KGE_beta_{var_sim}{sc1}'
                df_params_metrics.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals, sim_vals)
                key_r = f'r_{var_sim}{sc1}'
                df_params_metrics.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals, sim_vals)
                # average age metrics
                vars_sim = ['tt25_transp', 'tt50_transp', 'tt75_transp', 'ttavg_transp',
                            'tt25_q_ss', 'tt50_q_ss', 'tt75_q_ss', 'ttavg_q_ss',
                            'rt25_s', 'rt50_s', 'rt75_s',  'rtavg_s']
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
            df_params_metrics = df_params_metrics.copy()

        # write to .txt
        file = base_path_results / f"params_metrics_{tms}.txt"
        df_params_metrics.to_csv(file, header=True, index=False, sep="\t")

        # write simulated bulk sample to output file
        ds_sim_tm = ds_sim_tm.load()
        ds_sim_tm = ds_sim_tm.close()
        states_tm_file = base_path / f"states_{tms}_saltelli.nc"
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
