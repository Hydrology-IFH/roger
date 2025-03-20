from pathlib import Path
import os
import datetime
import h5netcdf
from cftime import num2date
import pandas as pd
import numpy as onp
import click
import roger


@click.option("--nruns", type=int, default=100)
@click.option("-rs", "--resample-size", type=int, default=10000)
@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir, resample_size, nruns):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path("/Volumes/LaCie/roger/examples/plot_scale/reckenholz")
        # base_path = Path(__file__).parent.parent

    # directory of results
    base_path_output = base_path / "output" / "svat_crop_monte_carlo"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)

    metric_name = "E_multi"
    lys_experiments = ["lys2", "lys3", "lys8"]
    for lys_experiment in lys_experiments:
        output_file = base_path_output / f"SVATCROP_{lys_experiment}_best_{nruns}_simulations.nc"
        with h5netcdf.File(output_file, 'r', decode_vlen_strings=False) as df:
            n_repeat = int(resample_size / df.dims["x"].size)
        if n_repeat <= 1:
            n_repeat = 1
        # write states of best model run
        bootstrap_file = base_path_output / f"SVATCROP_{lys_experiment}_bootstrap.nc"
        with h5netcdf.File(bootstrap_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title=f'RoGeR best {nruns} hydrologic monte carlo simulations (bootstrapped) optimized with {metric_name} at Reckenholz lysimeter ({lys_experiment})',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment='First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).',
            model_structure='SVAT model with free drainage',
            roger_version=f'{roger.__version__}'
            )
            with h5netcdf.File(output_file, 'r', decode_vlen_strings=False) as df:
                # set dimensions with a dictionary
                dict_dim = {'x': resample_size, 'y': 1, 'Time': len(df.variables['Time'])}
                time = onp.array(df.variables.get("Time"))
                time_origin = df.variables['Time'].attrs['time_origin']
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
                        vals_rep = onp.repeat(vals, n_repeat, axis=0)[:resample_size, :, :]
                        v[:, :, :] = vals_rep
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                    units=var_obj.attrs["units"])
                    elif var_sim not in list(f.dimensions.keys()) and ('x', 'y') == var_obj.dimensions:
                        v = f.create_variable(var_sim, ('x', 'y'), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        vals_rep = onp.repeat(vals, n_repeat, axis=0)[:resample_size, :]
                        v[:, :] = vals_rep
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                    units=var_obj.attrs["units"])
            # add year and day of year for nitrate transport model
            dates1 = num2date(
                time,
                units=f"days since {time_origin}",
                calendar="standard",
                only_use_cftime_datetimes=False,
            )
            dates = pd.to_datetime(dates1)
            vals = onp.array(dates.year)
            v = f.create_variable(
                "year", ("Time",), float, compression="gzip", compression_opts=1
            )
            v[:] = onp.array(dates.year)
            v.attrs.update(long_name="Year", units="")
            vals = onp.array(dates.year)
            v = f.create_variable(
                "doy", ("Time",), float, compression="gzip", compression_opts=1
            )
            v[:] = onp.array(dates.day_of_year)
            v.attrs.update(long_name="Day of year", units="")

    return


if __name__ == "__main__":
    main()
