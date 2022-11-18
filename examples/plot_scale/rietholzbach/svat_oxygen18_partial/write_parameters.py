from pathlib import Path
import h5netcdf
import datetime
import yaml
import numpy as onp
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent
    file_path = base_path / "param_bounds.yml"
    with open(file_path, 'r') as file:
        bounds = yaml.safe_load(file)

    # initialize parameter file
    params_file = base_path / "parameters.nc"
    with h5netcdf.File(params_file, 'w', decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title='Parameters for partial parameter dependency',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment='',
        )

    transport_models = ['preferential', 'advection-dispersion', 'power', 'older-preference', 'preferential + advection-dispersion', 'time-variant', 'time-variant-transp']
    for tm in transport_models:
        n_params = len(bounds[tm].keys())
        args = []
        for key in bounds[tm].keys():
            arr = onp.linspace(bounds[tm][key][0], bounds[tm][key][1], num=bounds[tm][key][2])
            args.append(arr)
        params = onp.array(onp.meshgrid(*args)).T.reshape(-1, n_params)

        # write sampled parameters to file
        with h5netcdf.File(params_file, 'a', decode_vlen_strings=False) as f:
            dict_dim = {'x': params.shape[0], 'y': 1}
            f.create_group(f"{tm}")
            f.groups[f"{tm}"].dimensions = dict_dim
            v = f.groups[f"{tm}"].create_variable('x', ('x',), float, compression="gzip", compression_opts=1)
            v.attrs['long_name'] = 'model run'
            v.attrs['units'] = ''
            v[:] = onp.arange(dict_dim["x"])
            v = f.groups[f"{tm}"].create_variable('y', ('y',), float, compression="gzip", compression_opts=1)
            v.attrs['long_name'] = ''
            v.attrs['units'] = ''
            v[:] = onp.arange(dict_dim["y"])
            for i, var_sim in enumerate(bounds[tm].keys()):
                v = f.groups[f"{tm}"].create_variable(var_sim, ('x', 'y'), float, compression="gzip", compression_opts=1)
                v[:, 0] = params[:, i]
                v.attrs.update(units='-')
    return


if __name__ == "__main__":
    main()
