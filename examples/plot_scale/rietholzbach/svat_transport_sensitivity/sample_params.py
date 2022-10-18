from pathlib import Path
import h5netcdf
import datetime
import yaml
from SALib.sample import saltelli
import numpy as onp
import click


@click.option("-ns", "--nsamples", type=int, default=1024)
@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(nsamples, tmp_dir):
    base_path = Path(__file__).parent
    file_path = base_path / "param_bounds.yml"
    with open(file_path, 'r') as file:
        bounds = yaml.safe_load(file)

    # initialize parameter file
    params_file = base_path / "params_saltelli.nc"
    with h5netcdf.File(params_file, 'w', decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title='Transport model parameters of sensitivity analysis at Rietholzbach lysimeter site',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment='',
        )

    transport_models = list(bounds.keys())
    for tm in transport_models:
        params = saltelli.sample(bounds[tm], nsamples, calc_second_order=False)
        nrows = params.shape[0]

        # write sampled parameters to file
        with h5netcdf.File(params_file, 'a', decode_vlen_strings=False) as f:
            dict_dim = {'x': nrows, 'y': 1}
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
            for i, var_sim in enumerate(bounds[tm]['names']):
                v = f.groups[f"{tm}"].create_variable(var_sim, ('x', 'y'), float, compression="gzip", compression_opts=1)
                v[:, 0] = params[:, i]
                v.attrs.update(units="-")

    return


if __name__ == "__main__":
    main()
