from pathlib import Path
import os
import glob
import datetime
import h5netcdf
import xarray as xr
from cftime import num2date, date2num
import pandas as pd
from de import de
import numpy as onp
import click
import roger
import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs
import matplotlib as mpl
import seaborn as sns
mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
sns.set_style("ticks")


@click.option("-ns", "--nsamples", type=int, default=10000)
@click.option("-ss", "--split-size", type=int, default=1000)
@click.option("-tms", "--transport-model-structure", type=click.Choice(['complete-mixing', 'piston', 'preferential', 'advection-dispersion', 'time-variant_advection-dispersion', 'time-variant_preferential_+_advection-dispersion', 'time-variant', 'power', 'time-variant_power', 'time-variant-transp', 'older-preference', 'preferential_+_advection-dispersion']), default='complete-mixing')
@click.option("--sas-solver", type=click.Choice(['RK4', 'Euler', 'deterministic']), default='deterministic')
@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(nsamples, transport_model_structure, split_size, sas_solver, tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent
    age_max = "age_max_1500_days"
    nruns_hm = 100
    metric_name = "KGE_multi"
    metric_for_optimization = f"optimized_with_{metric_name}_hm{nruns_hm}"
    tms = transport_model_structure.replace("_", " ")
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
    base_path_results = base_path / "results" / sas_solver / age_max / metric_for_optimization
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
    base_path_figs = base_path / "figures" / sas_solver / age_max / metric_for_optimization
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    # merge diagnostics into single file
    x1x2 = onp.arange(0, nsamples, split_size).tolist()
    if nsamples not in x1x2:
        x1x2.append(nsamples)

    diagnostics = ['average',
                   'constant',
                   'maximum']
    for diagnostic in diagnostics:
        path = str(base_path / sas_solver / age_max / metric_for_optimization / f"SVATTRANSPORT_{transport_model_structure}_{sas_solver}_*.{diagnostic}.nc")
        diag_files = glob.glob(path)
        if diag_files:
            diag_file = base_path / sas_solver / age_max / metric_for_optimization / f"SVATTRANSPORT_{transport_model_structure}_{sas_solver}.{diagnostic}.nc"
            if not os.path.exists(diag_file):
                click.echo(f'Merge {diagnostic} of {tms} ...')
                # initial diagnostic file
                with h5netcdf.File(diag_file, 'w', decode_vlen_strings=False) as f:
                    f.attrs.update(
                        date_created=datetime.datetime.today().isoformat(),
                        title=f'RoGeR {tms} transport model Monte Carlo simulations at Rietholzbach lysimeter site',
                        institution='University of Freiburg, Chair of Hydrology',
                        references='',
                        comment='First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).',
                        model_structure=f'SVAT {tms} transport model with free drainage',
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
                        with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
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

    states_hm1_file = base_path / sas_solver / age_max / metric_for_optimization / f"states_hm{nruns_hm}_bootstrap_for_{sas_solver}.nc"
    if os.path.exists(states_hm1_file):
        states_hm_mc_file = base_path / sas_solver / age_max / metric_for_optimization / "states_hm_for_tm_mc.nc"
        n_repeat = int(nsamples / split_size)
        if not os.path.exists(states_hm_mc_file):
            click.echo('Repeat hydrologic simualtions ...')
            with h5netcdf.File(states_hm_mc_file, 'w', decode_vlen_strings=False) as f:
                f.attrs.update(
                  date_created=datetime.datetime.today().isoformat(),
                  title=f'RoGeR best {nruns_hm} monte carlo simulations (bootstrapped) optimized with {metric_name} at Rietholzbach lysimeter site',
                  institution='University of Freiburg, Chair of Hydrology',
                  references='',
                  comment='First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).',
                  model_structure='SVAT model with free drainage',
                  roger_version=f'{roger.__version__}'
                )
                with h5netcdf.File(states_hm1_file, 'r', decode_vlen_strings=False) as df:
                    # set dimensions with a dictionary
                    dict_dim = {'x': nsamples, 'y': 1, 'Time': len(df.variables['Time'])}
                    if not f.dimensions:
                        f.dimensions = dict_dim
                        v = f.create_variable('x', ('x',), float, compression="gzip", compression_opts=1)
                        v.attrs['long_name'] = 'Number of model run'
                        v.attrs['units'] = ''
                        v[:] = onp.arange(dict_dim["x"])
                        v = f.create_variable('y', ('y',), float, compression="gzip", compression_opts=1)
                        v.attrs['long_name'] = ''
                        v.attrs['units'] = ''
                        v[:] = onp.arange(dict_dim["y"])
                        v = f.create_variable('Time', ('Time',), float, compression="gzip", compression_opts=1)
                        var_obj = df.variables.get('Time')
                        v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                       units=var_obj.attrs["units"])
                        v[:] = onp.array(var_obj)
                    for var_sim in list(df.variables.keys()):
                        var_obj = df.variables.get(var_sim)
                        if var_sim not in list(f.dimensions.keys()) and ('x', 'y', 'Time') == var_obj.dimensions:
                            v = f.create_variable(var_sim, ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                            vals = onp.array(var_obj)
                            vals_rep = onp.repeat(vals, n_repeat, axis=0)
                            v[:, :, :] = vals_rep
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
                        elif var_sim not in list(f.dimensions.keys()) and ('x', 'y') == var_obj.dimensions:
                            v = f.create_variable(var_sim, ('x', 'y'), float, compression="gzip", compression_opts=1)
                            vals = onp.array(var_obj)
                            vals_rep = onp.repeat(vals, n_repeat, axis=0)
                            v[:, :] = vals_rep
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])

    # merge results into single file
    path = str(base_path / sas_solver / age_max / metric_for_optimization / f"SVATTRANSPORT_{transport_model_structure}_{sas_solver}.*.nc")
    diag_files = glob.glob(path)
    states_tm_file = base_path / sas_solver / age_max / metric_for_optimization / f"states_{transport_model_structure}_monte_carlo.nc"
    if not os.path.exists(states_tm_file):
        click.echo(f'Merge output files of {tms} into {states_tm_file.as_posix()}')
        with h5netcdf.File(states_tm_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title=f'RoGeR {tms} transport model Monte Carlo simulations at Rietholzbach lysimeter site',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment='First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).',
                model_structure=f'SVAT {tms} transport model with free drainage',
                sas_solver=f'{sas_solver}',
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

    # load hydrologic simulation
    if tms in ['complete-mixing', 'piston']:
        states_hm_file = base_path / sas_solver / age_max / metric_for_optimization / f"states_hm{nruns_hm}.nc"
    else:
        states_hm_file = base_path / sas_solver / age_max / metric_for_optimization / "states_hm_for_tm_mc.nc"
    ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")
    days_sim_hm = (ds_sim_hm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim_hm = num2date(days_sim_hm, units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))

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

    file = base_path_results / f"params_metrics_{transport_model_structure}.txt"
    if not os.path.exists(file):
        click.echo(f'Calculate metrics for {tms} ...')

        # load transport simulation
        states_tm_file = base_path / sas_solver / age_max / metric_for_optimization / f"states_{transport_model_structure}_monte_carlo.nc"
        ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
        days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))

        # DataFrame with sampled model parameters and the corresponding metrics
        df_params_metrics = pd.DataFrame(index=range(ds_sim_tm.dims['x']))
        df_params_metrics.loc[:, 'c1_mak'] = ds_sim_hm["c1_mak"].values.flatten()
        df_params_metrics.loc[:, 'c2_mak'] = ds_sim_hm["c2_mak"].values.flatten()
        df_params_metrics.loc[:, 'dmpv'] = ds_sim_hm["dmpv"].values.flatten()
        df_params_metrics.loc[:, 'lmpv'] = ds_sim_hm["lmpv"].values.flatten()
        df_params_metrics.loc[:, 'theta_ac'] = ds_sim_hm["theta_ac"].values.flatten()
        df_params_metrics.loc[:, 'theta_ufc'] = ds_sim_hm["theta_ufc"].values.flatten()
        df_params_metrics.loc[:, 'theta_pwp'] = ds_sim_hm["theta_pwp"].values.flatten()
        df_params_metrics.loc[:, 'ks'] = ds_sim_hm["ks"].values.flatten()
        # sampled model parameters
        if tms == "advection-dispersion":
            df_params_metrics.loc[:, 'a_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=2).values.flatten()
            df_params_metrics.loc[:, 'a_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=2).values.flatten()
            df_params_metrics.loc[:, 'a_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=2).values.flatten()
        elif tms == "older-preference":
            df_params_metrics.loc[:, 'a_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=2).values.flatten()
            df_params_metrics.loc[:, 'a_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=2).values.flatten()
            df_params_metrics.loc[:, 'a_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=2).values.flatten()
        elif tms == "time-variant advection-dispersion":
            df_params_metrics.loc[:, 'a_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b1_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=3).values.flatten()
            df_params_metrics.loc[:, 'b2_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=3).values.flatten() + ds_sim_tm["sas_params_transp"].isel(n_sas_params=4).values.flatten()
            df_params_metrics.loc[:, 'a1_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=3).values.flatten()
            df_params_metrics.loc[:, 'a2_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=3).values.flatten() + ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=4).values.flatten()
            df_params_metrics.loc[:, 'b_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=2).values.flatten()
            df_params_metrics.loc[:, 'a1_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=3).values.flatten()
            df_params_metrics.loc[:, 'a2_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=3).values.flatten() + ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=4).values.flatten()
            df_params_metrics.loc[:, 'b_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=2).values.flatten()
        elif tms == "time-variant preferential + advection-dispersion":
            df_params_metrics.loc[:, 'a_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b1_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=3).values.flatten()
            df_params_metrics.loc[:, 'b2_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=3).values.flatten() + ds_sim_tm["sas_params_transp"].isel(n_sas_params=4).values.flatten()
            df_params_metrics.loc[:, 'a_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b1_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=3).values.flatten()
            df_params_metrics.loc[:, 'b2_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=3).values.flatten() + ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=4).values.flatten()
            df_params_metrics.loc[:, 'a1_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=3).values.flatten()
            df_params_metrics.loc[:, 'a2_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=3).values.flatten() + ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=4).values.flatten()
            df_params_metrics.loc[:, 'b_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=2).values.flatten()
        elif tms == "time-variant":
            df_params_metrics.loc[:, 'c_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=4).values.flatten()
            df_params_metrics.loc[:, 'c_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=4).values.flatten()
            df_params_metrics.loc[:, 'a_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=2).values.flatten()
        elif tms == "time-variant-transp":
            df_params_metrics.loc[:, 'a_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b1_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=3).values.flatten()
            df_params_metrics.loc[:, 'b2_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=3).values.flatten() + ds_sim_tm["sas_params_transp"].isel(n_sas_params=4).values.flatten()
            df_params_metrics.loc[:, 'a_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=2).values.flatten()
            df_params_metrics.loc[:, 'a_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=2).values.flatten()
        elif tms == "preferential":
            df_params_metrics.loc[:, 'a_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=2).values.flatten()
            df_params_metrics.loc[:, 'a_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=2).values.flatten()
            df_params_metrics.loc[:, 'a_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=2).values.flatten()
        elif tms == "preferential + advection-dispersion":
            df_params_metrics.loc[:, 'a_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=2).values.flatten()
            df_params_metrics.loc[:, 'a_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=2).values.flatten()
            df_params_metrics.loc[:, 'a_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'b_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=2).values.flatten()
        elif tms == "power":
            df_params_metrics.loc[:, 'k_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'k_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=1).values.flatten()
            df_params_metrics.loc[:, 'k_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=1).values.flatten()
        elif tms == "time-variant power":
            df_params_metrics.loc[:, 'k1_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=3).values.flatten()
            df_params_metrics.loc[:, 'k2_transp'] = ds_sim_tm["sas_params_transp"].isel(n_sas_params=3).values.flatten() + ds_sim_tm["sas_params_transp"].isel(n_sas_params=4).values.flatten()
            df_params_metrics.loc[:, 'k1_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=3).values.flatten()
            df_params_metrics.loc[:, 'k2_q_rz'] = ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=3).values.flatten() + ds_sim_tm["sas_params_q_rz"].isel(n_sas_params=4).values.flatten()
            df_params_metrics.loc[:, 'k1_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=3).values.flatten()
            df_params_metrics.loc[:, 'k2_q_ss'] = ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=3).values.flatten() + ds_sim_tm["sas_params_q_ss"].isel(n_sas_params=4).values.flatten()

        # compare observations and simulations
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

            perc_sample_sum_obs = df_perc_18O_sim.join(df_perc_18O_obs).groupby(['sample_no']).sum().loc[:, 'perc_obs']
            sample_no['perc_obs_sum'] = perc_sample_sum_obs.values
            df_perc_18O_sim = df_perc_18O_sim.join(sample_no['perc_obs_sum'])
            df_perc_18O_sim.loc[:, 'perc_obs_sum'] = df_perc_18O_sim.loc[:, 'perc_obs_sum'].fillna(method='bfill', limit=14)

            # join observations on simulations
            for sc, sc1 in zip([0, 1, 2, 3], ['', 'dry', 'normal', 'wet']):
                obs_vals = ds_obs['d18O_PERC'].isel(x=0, y=0).values
                sim_vals = d18O_perc_bs[nrow, 0, :]
                df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
                df_obs.loc[:, 'obs'] = obs_vals
                df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)

                if sc > 0:
                    df_rows = pd.DataFrame(index=df_eval.index).join(df_thetap)
                    rows = (df_rows['sc'].values == sc)
                    df_eval = df_eval.loc[rows, :]
                df_eval = df_eval.dropna()
                # calculate metrics
                if len(df_eval.index) > 10:
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
                    # add offset since diagnostic efficiency requires positive values
                    offset = onp.nanmin(df_eval.values) * (-1) + 1
                    obs_vals = df_eval.loc[:, 'obs'].values + offset
                    sim_vals = df_eval.loc[:, 'sim'].values + offset
                    # share of observations with zero values
                    key_p0 = 'p0_' + var_sim + f'{sc1}'
                    df_params_metrics.loc[nrow, key_p0] = 0
                    # mean absolute relative error
                    key_mare = 'MARE_' + var_sim + f'{sc1}'
                    df_params_metrics.loc[nrow, key_mare] = eval_utils.calc_mare(obs_vals, sim_vals)
                    # mean relative bias
                    key_brel_mean = 'brel_mean_' + var_sim + f'{sc1}'
                    brel_mean = de.calc_brel_mean(obs_vals, sim_vals)
                    df_params_metrics.loc[nrow, key_brel_mean] = brel_mean
                    # residual relative bias
                    brel_res = de.calc_brel_res(obs_vals, sim_vals)
                    # area of relative residual bias
                    key_b_area = 'b_area_' + var_sim + f'{sc1}'
                    b_area = de.calc_bias_area(brel_res)
                    df_params_metrics.loc[nrow, key_b_area] = b_area
                    # temporal correlation
                    key_temp_cor = 'temp_cor_' + var_sim + f'{sc1}'
                    temp_cor = de.calc_temp_cor(obs_vals, sim_vals)
                    df_params_metrics.loc[nrow, key_temp_cor] = temp_cor
                    # diagnostic efficiency
                    key_de = 'DE_' + var_sim + f'{sc1}'
                    df_params_metrics.loc[nrow, key_de] = de.calc_de(obs_vals, sim_vals)
                    # relative bias
                    brel = de.calc_brel(obs_vals, sim_vals)
                    # total bias
                    key_b_tot = 'b_tot_' + var_sim + f'{sc1}'
                    b_tot = de.calc_bias_tot(brel)
                    df_params_metrics.loc[nrow, key_b_tot] = b_tot
                    # bias of lower exceedance probability
                    key_b_hf = 'b_hf_' + var_sim + f'{sc1}'
                    b_hf = de.calc_bias_hf(brel)
                    df_params_metrics.loc[nrow, key_b_hf] = b_hf
                    # error contribution of higher exceedance probability
                    key_err_hf = 'err_hf_' + var_sim + f'{sc1}'
                    err_hf = de.calc_err_hf(b_hf, b_tot)
                    df_params_metrics.loc[nrow, key_err_hf] = err_hf
                    # bias of higher exceedance probability
                    key_b_lf = 'b_lf_' + var_sim + f'{sc1}'
                    b_lf = de.calc_bias_lf(brel)
                    df_params_metrics.loc[nrow, key_b_lf] = b_lf
                    # error contribution of lower exceedance probability
                    key_err_lf = 'err_lf_' + var_sim + f'{sc1}'
                    err_lf = de.calc_err_hf(b_lf, b_tot)
                    df_params_metrics.loc[nrow, key_err_lf] = err_lf
                    # direction of bias
                    key_b_dir = 'b_dir_' + var_sim + f'{sc1}'
                    b_dir = de.calc_bias_dir(brel_res)
                    df_params_metrics.loc[nrow, key_b_dir] = b_dir
                    # slope of bias
                    key_b_slope = 'b_slope_' + var_sim + f'{sc1}'
                    b_slope = de.calc_bias_slope(b_area, b_dir)
                    df_params_metrics.loc[nrow, key_b_slope] = b_slope
                    # (y, x) trigonometric inverse tangent
                    key_phi = 'phi_' + var_sim + f'{sc1}'
                    df_params_metrics.loc[nrow, key_phi] = de.calc_phi(brel_mean, b_slope)

                else:
                    var_sim = 'C_iso_q_ss'
                    key_kge = f'KGE_{var_sim}{sc1}'
                    df_params_metrics.loc[nrow, key_kge] = onp.nan
                    key_kge_alpha = f'KGE_alpha_{var_sim}{sc1}'
                    df_params_metrics.loc[nrow, key_kge_alpha] = onp.nan
                    key_kge_beta = f'KGE_beta_{var_sim}{sc1}'
                    df_params_metrics.loc[nrow, key_kge_beta] = onp.nan
                    key_r = f'r_{var_sim}{sc1}'
                    df_params_metrics.loc[nrow, key_r] = onp.nan

            # avoid defragmentation of DataFrame
            click.echo(f'{nrow}')
            df_params_metrics = df_params_metrics.copy()

        # write to .txt
        file = base_path_results / f"params_metrics_{transport_model_structure}.txt"
        df_params_metrics.to_csv(file, header=True, index=False, sep="\t")

        # write simulated bulk sample to output file
        ds_sim_tm = ds_sim_tm.load()
        ds_sim_tm = ds_sim_tm.close()
        states_tm_file = base_path / sas_solver / age_max / metric_for_optimization / f"states_{transport_model_structure}_monte_carlo.nc"
        with h5netcdf.File(states_tm_file, 'a', decode_vlen_strings=False) as f:
            try:
                v = f.create_variable('d18O_perc_bs', ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                v.attrs.update(long_name="bulk sample of d18O in percolation",
                               units="permil")
                v[:, :, :] = d18O_perc_bs
            except ValueError:
                v = f.get('d18O_perc_bs')
                v[:, :, :] = d18O_perc_bs

    # dotty plots
    click.echo(f'Dotty plots for {tms} ...')
    file = base_path_results / f"params_metrics_{transport_model_structure}.txt"
    df_params_metrics = pd.read_csv(file, header=0, index_col=False, sep="\t")
    df_metrics = df_params_metrics.loc[:, ['KGE_C_iso_q_ss']]
    if tms == "complete-mixing":
        df_params = df_params_metrics.loc[:, ['c1_mak', 'c2_mak', 'dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks']]
    elif tms == "piston":
        df_params = df_params_metrics.loc[:, ['c1_mak', 'c2_mak', 'dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks']]
    elif tms == "advection-dispersion":
        df_params = df_params_metrics.loc[:, ['a_transp', 'b_transp', 'a_q_rz', 'b_q_rz', 'a_q_ss', 'b_q_ss']]
    elif tms == "older-preference":
        df_params = df_params_metrics.loc[:, ['a_transp', 'b_transp', 'a_q_rz', 'b_q_rz', 'a_q_ss', 'b_q_ss']]
    elif tms == "time-variant advection-dispersion":
        df_params = df_params_metrics.loc[:, ['a_transp', 'b1_transp', 'b2_transp', 'a1_q_rz', 'a2_q_rz', 'b_q_rz', 'a1_q_ss', 'a2_q_ss', 'b_q_ss']]
    elif tms == "time-variant preferential + advection-dispersion":
        df_params = df_params_metrics.loc[:, ['a_transp', 'b1_transp', 'b2_transp', 'a_q_rz', 'b1_q_rz', 'b2_q_rz', 'a1_q_ss', 'a2_q_ss', 'b_q_ss']]
    elif tms == "time-variant-transp":
        df_params = df_params_metrics.loc[:, ['a_transp', 'b1_transp', 'b2_transp', 'a_q_rz', 'b_q_rz', 'a_q_ss', 'b_q_ss']]
    elif tms == "time-variant":
        df_params = df_params_metrics.loc[:, ['c_transp', 'c_q_rz', 'a_q_ss', 'b_q_ss']]
    elif tms == "preferential":
        df_params = df_params_metrics.loc[:, ['a_transp', 'b_transp', 'a_q_rz', 'b_q_rz', 'a_q_ss', 'b_q_ss']]
    elif tms == "preferential + advection-dispersion":
        df_params = df_params_metrics.loc[:, ['a_transp', 'b_transp', 'a_q_rz', 'b_q_rz', 'a_q_ss', 'b_q_ss']]
    elif tms == "power":
        df_params = df_params_metrics.loc[:, ['k_transp', 'k_q_rz', 'k_q_ss']]
    elif tms == "time-variant power":
        df_params = df_params_metrics.loc[:, ['k1_transp', 'k2_transp', 'k1_q_rz', 'k2_q_rz', 'k1_q_ss', 'k2_q_ss']]
    # select best model run
    idx_best = df_params_metrics['KGE_C_iso_q_ss'].idxmax()
    nrow = len(df_metrics.columns)
    ncol = len(df_params.columns)
    fig, ax1 = plt.subplots(nrow, ncol, sharey=True, figsize=(ncol*3.5, 3.5))
    if ncol > 1:
        ax = ax1.reshape(nrow, ncol)
        for i in range(nrow):
            for j in range(ncol):
                y = df_metrics.iloc[:, i]
                x = df_params.iloc[:, j]
                ax[i, j].scatter(x, y, s=4, c='grey', alpha=0.5)
                ax[i, j].set_xlabel('')
                ax[i, j].set_ylabel('')
                ax[i, j].set_ylim(-1, 1)
                # best model run
                y_best = df_metrics.iloc[idx_best, i]
                x_best = df_params.iloc[idx_best, j]
                ax[i, j].scatter(x_best, y_best, s=12, c='red', alpha=0.8)

        for j in range(ncol):
            xlabel = labs._LABS[df_params.columns[j]]
            ax[-1, j].set_xlabel(xlabel)

        ax[0, 0].set_ylabel(r'$KGE_{\delta^{18}O_{PERC}}$ [-]')

        fig.tight_layout()
        file = base_path_figs / f"dotty_plots_{transport_model_structure}.png"
        fig.savefig(file, dpi=250)
        plt.close('all')
    else:
        ax = ax1
        y = df_metrics.iloc[:, 0]
        x = df_params.iloc[:, 0]
        ax.scatter(x, y, s=4, c='grey', alpha=0.5)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_ylim(-1, 1)
        # best model run
        y_best = df_metrics.iloc[idx_best, 0]
        x_best = df_params.iloc[idx_best, 0]
        ax.scatter(x_best, y_best, s=12, c='red', alpha=0.8)
        xlabel = labs._LABS[df_params.columns[0]]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$KGE_{\delta^{18}O_{PERC}}$ [-]')
        fig.tight_layout()
        file = base_path_figs / f"dotty_plots_{transport_model_structure}.png"
        fig.savefig(file, dpi=250)
        plt.close('all')

    # write SAS parameters of best model run
    click.echo(f'Write SAS params of best {tms} simulation ...')
    states_tm_file = base_path / sas_solver / age_max / metric_for_optimization / f"states_{transport_model_structure}_monte_carlo.nc"
    ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
    params_tm_file = base_path / sas_solver / age_max / metric_for_optimization / f"sas_params_{transport_model_structure}.nc"
    with h5netcdf.File(params_tm_file, 'w', decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title='RoGeR SAS parameters of best monte carlo simulation at Rietholzbach Lysimeter site',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            model_structure=f'SVAT {tms} model with free drainage',
            sas_solver=f'{sas_solver}',
        )
        dict_dim = {'x': 1, 'y': 1, 'n_sas_params': 8}
        if not f.dimensions:
            f.dimensions = dict_dim
            v = f.create_variable('x', ('x',), float, compression="gzip", compression_opts=1)
            v.attrs['long_name'] = 'Zonal coordinate'
            v.attrs['units'] = 'meters'
            v[:] = onp.arange(dict_dim["x"])
            v = f.create_variable('y', ('y',), float, compression="gzip", compression_opts=1)
            v.attrs['long_name'] = 'Meridonial coordinate'
            v.attrs['units'] = 'meters'
            v[:] = onp.arange(dict_dim["y"])
            v = f.create_variable('n_sas_params', ('n_sas_params',), float, compression="gzip", compression_opts=1)
            v.attrs['long_name'] = 'Number of SAS parameters'
            v.attrs['units'] = ' '
            v[:] = onp.arange(dict_dim["n_sas_params"])

        try:
            v = f.create_variable('sas_params_transp', ('x', 'y', 'n_sas_params'), float, compression="gzip", compression_opts=1)
        except ValueError:
            v = f.get('sas_params_transp')
        v[:, :, :] = ds_sim_tm["sas_params_transp"].isel(x=idx_best)
        v.attrs.update(long_name="SAS parameters of transpiration",
                        units=" ")
        try:
            v = f.create_variable('sas_params_q_rz', ('x', 'y', 'n_sas_params'), float, compression="gzip", compression_opts=1)
        except ValueError:
            v = f.get('sas_params_q_rz')
        v[:, :, :] = ds_sim_tm["sas_params_q_rz"].isel(x=idx_best)
        v.attrs.update(long_name="SAS parameters of root zone percolation",
                        units=" ")
        try:
            v = f.create_variable('sas_params_q_ss', ('x', 'y', 'n_sas_params'), float, compression="gzip", compression_opts=1)
        except ValueError:
            v = f.get('sas_params_q_ss')
        v[:, :, :] = ds_sim_tm["sas_params_q_ss"].isel(x=idx_best)
        v.attrs.update(long_name="SAS parameters of subsoil percolation",
                        units=" ")
    ds_sim_tm = ds_sim_tm.load()
    ds_sim_tm = ds_sim_tm.close()

    # write states of best transport simulation
    click.echo(f'Write states of best {tms} simulation ...')
    states_tm_mc_file = base_path / sas_solver / age_max / metric_for_optimization / f"states_{transport_model_structure}_monte_carlo.nc"
    states_tm_file = base_path / sas_solver / age_max / metric_for_optimization / f"states_best_{transport_model_structure}.nc"
    with h5netcdf.File(states_tm_file, 'w', decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title='RoGeR best transport model Monte Carlo simulations at Rietholzbach lysimeter site',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment='First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).',
            model_structure=f'SVAT {tms} model with free drainage',
            sas_solver=f'{sas_solver}',
        )
        # collect dimensions
        with h5netcdf.File(states_tm_mc_file, 'r', decode_vlen_strings=False) as df:
            f.attrs.update(
                roger_version=df.attrs['roger_version']
            )
            # set dimensions with a dictionary
            dict_dim = {'x': 1, 'y': 1, 'Time': len(df.variables['Time']), 'ages': len(df.variables['ages']), 'nages': len(df.variables['nages']), 'n_sas_params': len(df.variables['n_sas_params'])}
            time = onp.array(df.variables.get('Time'))
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
                if var_sim not in list(dict_dim.keys()) and ('x', 'y', 'Time') == var_obj.dimensions:
                    v = f.create_variable(var_sim, ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                    vals = onp.array(var_obj)
                    v[:, :, :] = vals[idx_best, :, :]
                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                    units=var_obj.attrs["units"])
                    del var_obj, vals
                elif var_sim not in list(dict_dim.keys()) and ('x', 'y') == var_obj.dimensions:
                    v = f.create_variable(var_sim, ('x', 'y'), float)
                    vals = onp.array(var_obj)
                    v[:, :] = vals[idx_best, :]
                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                    units=var_obj.attrs["units"])
                    del var_obj, vals
                elif var_sim not in list(dict_dim.keys()) and ('x', 'y', 'n_sas_params') == var_obj.dimensions:
                    v = f.create_variable(var_sim, ('x', 'y', 'n_sas_params'), float, compression="gzip", compression_opts=1)
                    vals = onp.array(var_obj)
                    v[:, :, :] = vals[idx_best, :, :]
                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                    units=var_obj.attrs["units"])
                    del var_obj, vals
                elif var_sim not in list(dict_dim.keys()) and ('x', 'y', 'Time', 'ages') == var_obj.dimensions:
                    v = f.create_variable(var_sim, ('x', 'y', 'Time', 'ages'), float, compression="gzip", compression_opts=1)
                    vals = onp.array(var_obj)
                    v[:, :, :, :] = vals[idx_best, :, :, :]
                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                    units=var_obj.attrs["units"])
                    del var_obj, vals
                elif var_sim not in list(dict_dim.keys()) and ('x', 'y', 'Time', 'nages') == var_obj.dimensions:
                    v = f.create_variable(var_sim, ('x', 'y', 'Time', 'nages'), float, compression="gzip", compression_opts=1)
                    vals = onp.array(var_obj)
                    v[:, :, :, :] = vals[idx_best, :, :, :]
                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                    units=var_obj.attrs["units"])
                    del var_obj, vals

    # write hydrologic states corresponding to best transport simulation
    click.echo(f'Write states of best hydrologic simulation corresponding to {tms} ...')

    # write states of best hydrologic simulation corresponding to best transport simulation
    states_hm_mc_file = base_path / sas_solver / age_max / metric_for_optimization / "states_hm_for_tm_mc.nc"
    ds_hm_for_tm_mc = xr.open_dataset(states_hm_mc_file, engine="h5netcdf")
    ds_hm_best = ds_sim_hm.loc[dict(x=idx_best)]
    ds_hm_best.attrs['title'] = f'Best hydrologic simulation corresponding to best {tms} oxygen-18 simulation'
    days = date2num(ds_hm_best["Time"].values.astype('M8[ms]').astype('O'), units=f"days since {ds_hm_for_tm_mc['Time'].attrs['time_origin']}", calendar='standard')
    ds_hm_best = ds_hm_best.assign_coords(Time=("Time", days))
    ds_hm_best.Time.attrs['units'] = "days"
    ds_hm_best.Time.attrs['time_origin'] = ds_hm_for_tm_mc['Time'].attrs['time_origin']
    file = base_path / sas_solver / age_max / metric_for_optimization / f"states_hm_best_for_{transport_model_structure}.nc"
    ds_hm_best.to_netcdf(file, engine="h5netcdf")
    return


if __name__ == "__main__":
    main()
