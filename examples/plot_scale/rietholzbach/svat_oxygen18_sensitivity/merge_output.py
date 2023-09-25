from pathlib import Path
import os
import glob
import h5netcdf
import datetime
import numpy as onp
import click
import roger
import matplotlib as mpl
import seaborn as sns
mpl.use("agg")
sns.set_style("ticks")


@click.option("-ss", "--split-size", type=int, default=1000)
@click.option("-tms", "--transport-model-structure", type=click.Choice(['complete-mixing', 'piston', 'advection-dispersion-power', 'time-variant_advection-dispersion-power']), default='advection-dispersion-power')
@click.option("--sas-solver", type=click.Choice(['RK4', 'Euler', 'deterministic']), default='deterministic')
@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(split_size, transport_model_structure, sas_solver, tmp_dir):
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

    # merge diagnostics into single file
    diagnostics = ['average',
                   'constant',
                   'maximum']
    if tms in ['complete-mixing', 'piston']:
        nsamples = 1024 * 10
    elif tms in ['advection-dispersion-power']:
        nsamples = 1024 * 13
    elif tms in ['time-variant advection-dispersion-power']:
        nsamples = 1024 * 16
    x1x2 = onp.arange(0, nsamples, split_size).tolist()
    if nsamples not in x1x2:
        x1x2.append(nsamples)
    for diagnostic in diagnostics:
        path = str(base_path_output / f"SVATTRANSPORT_{transport_model_structure}_{sas_solver}_*_*.{diagnostic}.nc")
        diag_files = glob.glob(path)
        if diag_files:
            diag_file = base_path_output / f"SVATTRANSPORT_{transport_model_structure}_{sas_solver}.{diagnostic}.nc"
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
                        file = base_path_output / f"SVATTRANSPORT_{transport_model_structure}_{sas_solver}_{x1}_{x2}.{diagnostic}.nc"
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
    path = str(base_path_output / f"SVATTRANSPORT_{transport_model_structure}_{sas_solver}.*.nc")
    diag_files = glob.glob(path)
    states_tm_file = base_path_output / f"states_{transport_model_structure}_saltelli.nc"
    if not os.path.exists(states_tm_file):
        click.echo(f'Merge output files of {tms} into {states_tm_file.as_posix()}')
        with h5netcdf.File(states_tm_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title=f'RoGeR {tms} model Saltelli simulations at Rietholzbach lysimeter site',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment='First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).',
                model_structure=f'SVAT {tms} model with free drainage',
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
    return


if __name__ == "__main__":
    main()
