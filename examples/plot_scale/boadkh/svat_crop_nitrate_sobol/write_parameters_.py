from pathlib import Path
import yaml
from SALib import ProblemSpec
import numpy as onp
import pandas as pd
import datetime
import h5netcdf
import click


_UNITS = {
    "alpha_transp": "-",
    "alpha_q": "-",
    "km_denit": "kg N/ha/year",
    "km_nit": "kg N/ha/year",
    "kmin": "kg N/ha/year",
    "kfix": "kg N/ha/year",
    "dmax_denit": "kg N/ha/year",
    "dmax_nit": "kg N/ha/year",
    "phi_soil_temp": "day of year",
    "clay": "-",
    "z_soil": "mm",
    "c_fert": "-",
}

@click.option("-ns", "--nsamples", type=int, default=2**9)
@click.command("main")
def main(nsamples):
    base_path = Path(__file__).parent
    # load parameter boundaries
    file_param_bounds = base_path / "param_bounds_.yml"
    with open(file_param_bounds, "r") as file:
        dict_bounds = yaml.safe_load(file)
        list_names = [key for key in dict_bounds.keys()]
        list_param_bounds = [dict_bounds[key] for key in dict_bounds.keys()]
        bounds = {'names': list_names, 'bounds': list_param_bounds}

    # generate parameter samples using a Sobol' sequence
    sp = ProblemSpec(bounds)
    params = sp.sample_sobol(nsamples, calc_second_order=False).samples
    nrows = params.shape[0]
    ncols = 1

    # write parameters to csv
    df_params = pd.DataFrame(index=range(nrows * ncols))
    for i, param in enumerate(bounds["names"]):
        values = params[:, i].reshape(nrows, ncols).astype(onp.float32)

        # write parameters to dataframe
        df_params.loc[:, param] = values.flatten()


    param_names = ["alpha_transp", "alpha_q", "km_denit", "km_nit", "kmin", "dmax_denit", "dmax_nit", "kfix", "c_fert"]
    df_params = df_params.loc[:, param_names]

    # write parameters to csv
    df_params.columns = [
        ["[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[-]"],
        param_names,
    ]
    df_params.to_csv(base_path / "parameters.csv", index=False, sep=";")

    # write parameters to netcdf
    params_file = base_path / "parameters.nc"
    with h5netcdf.File(params_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title="RoGeR SAS and nitrate parameters for Baden-Wuerttemberg, Germany",
        institution="University of Freiburg, Chair of Hydrology",
        references="",
        comment="",
        )
        dict_dim = {"x": len(df_params.index), "y": 1, 'scalar': 1}
        f.dimensions = dict_dim
        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
        v.attrs["long_name"] = "model run"
        v.attrs["units"] = "-"
        v[:] = onp.arange(dict_dim["x"])
        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
        v.attrs["long_name"] = "y"
        v.attrs["units"] = "-"
        v[:] = onp.arange(dict_dim["y"])
        v = f.create_variable('cell_width', ('scalar',), float)
        v.attrs['long_name'] = 'Cell width'
        v.attrs['units'] = 'm'
        v[:] = 1
        v = f.create_variable('x_origin', ('scalar',), float)
        v.attrs['long_name'] = 'Origin of x-direction'
        v.attrs['units'] = 'm'
        v[:] = 0
        v = f.create_variable('y_origin', ('scalar',), float)
        v.attrs['long_name'] = 'Origin of y-direction'
        v.attrs['units'] = 'm'
        v[:] = 0
        for i, key in enumerate(param_names):
            v = f.create_variable(
                key, ("x", "y"), onp.float32, compression="gzip", compression_opts=1
            )
            vals = df_params.iloc[:, i].values.flatten().reshape(nrows, ncols)
            v[:, :] = vals
            v.attrs.update(long_name=key, units=_UNITS[key])

    # write clay content to netcdf
    csv_file = base_path / "clay.csv"
    df_clay = pd.read_csv(csv_file, sep=";", skiprows=1)
    clay_file = base_path / "clay.nc"
    with h5netcdf.File(clay_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title="Clay parameter for Baden-Wuerttemberg, Germany",
        institution="University of Freiburg, Chair of Hydrology",
        references="",
        comment="",
        )
        dict_dim = {"x": len(df_clay.index), "y": 1, 'scalar': 1}
        f.dimensions = dict_dim
        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
        v.attrs["long_name"] = "model run"
        v.attrs["units"] = "-"
        v[:] = onp.arange(dict_dim["x"])
        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
        v.attrs["long_name"] = "y"
        v.attrs["units"] = "-"
        v[:] = onp.arange(dict_dim["y"])
        v = f.create_variable('cell_width', ('scalar',), float)
        v.attrs['long_name'] = 'Cell width'
        v.attrs['units'] = 'm'
        v[:] = 1
        v = f.create_variable('x_origin', ('scalar',), float)
        v.attrs['long_name'] = 'Origin of x-direction'
        v.attrs['units'] = 'm'
        v[:] = 0
        v = f.create_variable('y_origin', ('scalar',), float)
        v.attrs['long_name'] = 'Origin of y-direction'
        v.attrs['units'] = 'm'
        v[:] = 0
        v = f.create_variable(
            "clay", ("x", "y"), onp.float32, compression="gzip", compression_opts=1
        )
        vals = df_clay.loc[:, "clay"].values.astype(float).flatten().reshape(len(df_clay.index), 1)
        v[:, :] = vals
        v.attrs.update(long_name=key, units=_UNITS["clay"])


    # write soil depth to netcdf
    csv_file = base_path / "z_soil.csv"
    df_zsoil = pd.read_csv(csv_file, sep=";", skiprows=1)
    zsoil_file = base_path / "z_soil.nc"
    with h5netcdf.File(zsoil_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title="Soil depth parameter for Baden-Wuerttemberg, Germany",
        institution="University of Freiburg, Chair of Hydrology",
        references="",
        comment="",
        )
        dict_dim = {"x": len(df_zsoil.index), "y": 1, 'scalar': 1}
        f.dimensions = dict_dim
        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
        v.attrs["long_name"] = "model run"
        v.attrs["units"] = "-"
        v[:] = onp.arange(dict_dim["x"])
        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
        v.attrs["long_name"] = "y"
        v.attrs["units"] = "-"
        v[:] = onp.arange(dict_dim["y"])
        v = f.create_variable('cell_width', ('scalar',), float)
        v.attrs['long_name'] = 'Cell width'
        v.attrs['units'] = 'm'
        v[:] = 1
        v = f.create_variable('x_origin', ('scalar',), float)
        v.attrs['long_name'] = 'Origin of x-direction'
        v.attrs['units'] = 'm'
        v[:] = 0
        v = f.create_variable('y_origin', ('scalar',), float)
        v.attrs['long_name'] = 'Origin of y-direction'
        v.attrs['units'] = 'm'
        v[:] = 0
        v = f.create_variable(
            "z_soil", ("x", "y"), onp.float32, compression="gzip", compression_opts=1
        )
        vals = df_zsoil.loc[:, "z_soil"].values.astype(float).flatten().reshape(len(df_clay.index), 1)
        v[:, :] = vals
        v.attrs.update(long_name=key, units=_UNITS["z_soil"])
    return


if __name__ == "__main__":
    main()
