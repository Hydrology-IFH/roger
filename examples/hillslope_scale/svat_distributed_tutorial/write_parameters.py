from pathlib import Path
import h5netcdf
import datetime
import yaml
import numpy as onp
import click
import roger


_UNITS = {'z_soil': 'mm',
          'dmpv': '1/m2',
          'lmpv': 'mm',
          'theta_ac': '-',
          'theta_ufc': '-',
          'theta_pwp': '-',
          'ks': 'mm/hour',
          }


@click.option("-nx", "--nrows", type=int, default=12)
@click.option("-ny", "--ncols", type=str, default=24)
@click.command("main")
def main(nrows, ncols):
    base_path = Path(__file__).parent
    file_param_bounds = base_path / "param_bounds.yml"
    with open(file_param_bounds, 'r') as file:
        bounds = yaml.safe_load(file)

    # write parameters to netcdf
    RNG = onp.random.default_rng(42)
    file_params = base_path / "parameters.nc"
    with h5netcdf.File(file_params, 'w', decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title='RoGeR model parameters',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment='',
            model_structure='SVAT model with free drainage',
            roger_version=f'{roger.__version__}'
        )
        dict_dim = {'x': nrows, 'y': ncols}
        f.dimensions = dict_dim
        v = f.create_variable('x', ('x',), float, compression="gzip", compression_opts=1)
        v.attrs['long_name'] = 'x'
        v.attrs['units'] = 'm'
        v[:] = onp.arange(dict_dim["x"]) * 5
        v = f.create_variable('y', ('y',), float, compression="gzip", compression_opts=1)
        v.attrs['long_name'] = 'y'
        v.attrs['units'] = 'm'
        v[:] = onp.arange(dict_dim["y"]) * 5
        for i, param in enumerate(bounds.keys()):
            v = f.create_variable(param, ('x', 'y'), float, compression="gzip", compression_opts=1)
            v[:, :] = RNG.uniform(bounds[param][0], bounds[param][1], size=dict_dim['x'] * dict_dim['y']).reshape((dict_dim['x'], dict_dim['y']))
            v.attrs.update(units=_UNITS[param])
    return


if __name__ == "__main__":
    main()
