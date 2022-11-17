from pathlib import Path
import h5netcdf
import datetime
import numpy as onp
import yaml
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent
    file_path = base_path / "param_bounds.yml"
    with open(file_path, 'r') as file:
        bounds = yaml.safe_load(file)
    lu_id = onp.array([8])
    z_soil = onp.linspace(bounds["z_soil"][0], bounds["z_soil"][1], num=bounds["z_soil"][2])
    dmpv = onp.linspace(bounds["dmpv"][0], bounds["dmpv"][1], num=bounds["dmpv"][2])
    lmpv = onp.linspace(bounds["lmpv"][0], bounds["lmpv"][1], num=bounds["lmpv"][2])
    theta_ac = onp.linspace(bounds["theta_ac"][0], bounds["theta_ac"][1], num=bounds["theta_ac"][2])
    theta_ufc = onp.linspace(bounds["theta_ufc"][0], bounds["theta_ufc"][1], num=bounds["theta_ufc"][2])
    theta_pwp = onp.linspace(bounds["theta_pwp"][0], bounds["theta_pwp"][1], num=bounds["theta_pwp"][2])
    ks = onp.linspace(bounds["ks"][0], bounds["ks"][1], num=bounds["ks"][2])
    params = onp.array(onp.meshgrid(lu_id, z_soil, dmpv, lmpv, theta_ac, theta_ufc, theta_pwp, ks)).T.reshape(-1, 8)

    # initialize parameter file
    params_file = base_path / "parameters.nc"
    with h5netcdf.File(params_file, 'w', decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title='Parameters for continous development',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment='',
        )

        dict_dim = {'x': params.shape[0], 'y': 1}
        f.dimensions = dict_dim
        v = f.create_variable('x', ('x',), float, compression="gzip", compression_opts=1)
        v.attrs['long_name'] = 'model run'
        v.attrs['units'] = ''
        v[:] = onp.arange(dict_dim["x"])
        v = f.create_variable('y', ('y',), float, compression="gzip", compression_opts=1)
        v.attrs['long_name'] = ''
        v.attrs['units'] = ''
        v[:] = onp.arange(dict_dim["y"])
        v = f.create_variable('lu_id', ('x', 'y'), int, compression="gzip", compression_opts=1)
        v[:, 0] = params[:, 0].astype(int)
        v.attrs.update(units='-')
        v = f.create_variable('z_soil', ('x', 'y'), float, compression="gzip", compression_opts=1)
        v[:, 0] = params[:, 1]
        v.attrs.update(units='mm')
        v = f.create_variable('dmpv', ('x', 'y'), int, compression="gzip", compression_opts=1)
        v[:, 0] = params[:, 2].astype(int)
        v.attrs.update(units='1/m2')
        v = f.create_variable('lmpv', ('x', 'y'), int, compression="gzip", compression_opts=1)
        v[:, 0] = params[:, 3].astype(int)
        v.attrs.update(units='mm')
        v = f.create_variable('theta_ac', ('x', 'y'), float, compression="gzip", compression_opts=1)
        v[:, 0] = params[:, 4]
        v.attrs.update(units='-')
        v = f.create_variable('theta_ufc', ('x', 'y'), float, compression="gzip", compression_opts=1)
        v[:, 0] = params[:, 5]
        v.attrs.update(units='-')
        v = f.create_variable('theta_pwp', ('x', 'y'), float, compression="gzip", compression_opts=1)
        v[:, 0] = params[:, 6]
        v.attrs.update(units='-')
        v = f.create_variable('ks', ('x', 'y'), float, compression="gzip", compression_opts=1)
        v[:, 0] = params[:, 7]
        v.attrs.update(units='-')

    return


if __name__ == "__main__":
    main()
