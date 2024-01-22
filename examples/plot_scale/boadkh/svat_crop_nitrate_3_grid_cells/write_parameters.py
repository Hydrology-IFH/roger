from pathlib import Path
import numpy as onp
import pandas as pd
import datetime
import h5netcdf
import click


_UNITS = {
    "alpha_transp": "-",
    "alpha_q": "-",
    "c2_transp": "-",
    "c2_q_rz": "-",
    "c2_q_ss": "-",
    "km_denit": "kg N/ha/year",
    "km_nit": "kg N/ha/year",
    "kmin": "kg N/ha/year",
    "dmax_denit": "kg N/ha/year",
    "dmax_nit": "kg N/ha/year",
    "phi_soil_temp": "day of year",
    "clay": "-",
    "soil_fertility": "",
    "z_soil": "mm",
}

_c = 0.1
_VALS = {
    "alpha_transp": [0.95, 0.95, 0.95],
    "alpha_q": [0.7, 0.7, 0.7],
    "c2_transp": [0.6, 0.6, 0.6],
    "c2_q_rz": [1.5, 1.5, 1.5],
    "c2_q_ss": [1.5, 1.5, 1.5],
    "km_denit": [2.5, 9, 18.7],
    "km_nit": [_c*2.5, _c*9, _c*18.7],
    "kmin": [_c*2.5, _c*9, _c*18.7],
    "dmax_denit": [10, 40, 100],
    "dmax_nit": [_c*10, _c*40, _c*100],
    "phi_soil_temp": [91, 91, 91],
}

_X1 = [0, 1, 2]
_X2 = [1, 2, 3]

@click.command("main")
def main():
    base_path = Path(__file__).parent

    # write parameters to csv
    df_params = pd.DataFrame(index=range(_X2[-1]))
    for param in _VALS.keys():
        values = onp.zeros(_X2[-1])
        for i in range(len(_X2)):
            x1 = _X1[i]
            x2 = _X2[i]
            values[x1:x2] = _VALS[param][i]
            # write parameters to dataframe
        df_params.loc[:, param] = values.flatten()

    param_names = ["alpha_transp", "alpha_q", "c2_transp", "c2_q_rz", "c2_q_ss", "km_denit", "km_nit", "kmin", "dmax_denit", "dmax_nit", "phi_soil_temp"]
    df_params = df_params.loc[:, param_names]

    # write parameters to csv
    df_params.columns = [
        ["[-]", "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[day of year]"], 
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
            vals = df_params.iloc[:, i].values.flatten().reshape(_X2[-1], 1)
            v[:, :] = vals
            v.attrs.update(long_name=key, units=_UNITS[key])

    # write soil fertility to netcdf
    csv_file = base_path / "soil_fertility.csv"
    df_soil_fertility = pd.read_csv(csv_file, sep=";", skiprows=1)
    soil_fertility_file = base_path / "soil_fertility.nc"
    with h5netcdf.File(soil_fertility_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title="Soil fertility parameter for Baden-Wuerttemberg, Germany",
        institution="University of Freiburg, Chair of Hydrology",
        references="",
        comment="",
        )
        dict_dim = {"x": len(df_soil_fertility.index), "y": 1, 'scalar': 1}
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
            "soil_fertility", ("x", "y"), onp.float32, compression="gzip", compression_opts=1
        )
        vals = df_soil_fertility.loc[:, "soil_fertility"].values.astype(float).flatten().reshape(len(df_soil_fertility.index), 1)
        v[:, :] = vals
        v.attrs.update(long_name=key, units=_UNITS["soil_fertility"])

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
