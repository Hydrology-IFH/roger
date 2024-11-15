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
    "kfix": "kg N/ha/year",
    "kngl": "kg N/ha/year",
    "dmax_denit": "kg N/ha/year",
    "dmax_nit": "kg N/ha/year",
    "phi_soil_temp": "day of year",
    "clay": "-",
    "soil_fertility": "",
    "z_soil": "mm",
}

_VALS = {
    "alpha_transp": 0.55,
    "alpha_q": 0.55,
    "c2_transp": 0.6,
    "c2_q_rz": 1.5,
    "c2_q_ss": 1.5,
    "km_denit": [2.5, 18.7],
    "km_nit": [2.5, 12],
    "kmin": [10., 30.],
    "kfix": 40.,
    "kngl": 75.,
    "dmax_denit": [10., 40.],
    "dmax_nit": [10., 60.],
    "phi_soil_temp": 91.,
}


@click.command("main")
def main():
    base_path = Path(__file__).parent

    soil_fertility = 3.
    c_soil_fertility = 1.
    zsoil = 1350.
    clay = 0.2822544050076512
    c_clay = clay / (0.4 - 0.02)

    # write parameters to csv
    df_params = pd.DataFrame(index=range(3))
    df_params.loc[:, "alpha_transp"] = _VALS["alpha_transp"]
    df_params.loc[:, "alpha_q"] = _VALS["alpha_q"]
    df_params.loc[:, "c2_transp"] = _VALS["c2_transp"]
    df_params.loc[:, "c2_q_rz"] = _VALS["c2_q_rz"]
    df_params.loc[:, "c2_q_ss"] = _VALS["c2_q_ss"]
    df_params.loc[:, "phi_soil_temp"] = _VALS["phi_soil_temp"]
    df_params.loc[:, "km_denit"] = _VALS["km_denit"][0] + (_VALS["km_denit"][1] - _VALS["km_denit"][0]) * (1 - c_clay) 
    df_params.loc[:, "dmax_denit"] = _VALS["dmax_denit"][0] + (_VALS["dmax_denit"][1] - _VALS["dmax_denit"][0]) * c_clay
    for i in range(3):
        if i == 0:
            soil_fertility = 1.
            c_soil_fertility = 1/3
        elif i == 1:
            soil_fertility = 2.
            c_soil_fertility = 2/3
        elif i == 2:
            soil_fertility = 3.
            c_soil_fertility = 1.
        
        df_params.loc[df_params.index[i], "km_nit"] = _VALS["km_nit"][0] + (_VALS["km_nit"][1] - _VALS["km_nit"][0]) * (1 - c_clay) * c_soil_fertility
        df_params.loc[df_params.index[i], "dmax_nit"] = _VALS["dmax_nit"][0] + (_VALS["dmax_nit"][1] - _VALS["dmax_nit"][0]) * (1 - c_clay) * c_soil_fertility
        df_params.loc[df_params.index[i], "kmin"] = _VALS["kmin"][0] + (_VALS["kmin"][1] - _VALS["kmin"][0]) * c_soil_fertility
        df_params.loc[df_params.index[i], "kfix"] = _VALS["kfix"] * c_soil_fertility
        df_params.loc[:, "soil_fertility"] = soil_fertility
    df_params.loc[:, "kngl"] = _VALS["kngl"] * (1 - c_clay)
    df_params.loc[:, "clay"] = clay
    df_params.loc[:, "z_soil"] = zsoil

    param_names = ["alpha_transp", "alpha_q", "c2_transp", "c2_q_rz", "c2_q_ss", "km_denit", "km_nit", "kmin", "kfix", "kngl", "dmax_denit", "dmax_nit", "phi_soil_temp", "clay", "soil_fertility", "z_soil"]
    df_params = df_params.loc[:, param_names]

    # write parameters to csv
    df_params.columns = [
        ["[-]", "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[day of year]", "[-]", "", "[mm]"], 
        param_names,
    ]
    df_params.to_csv(base_path / "parameters.csv", index=False, sep=";")

    # write parameters to netcdf
    params_file = base_path / "parameters.nc"
    with h5netcdf.File(params_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title="RoGeR SAS and nitrate reference parameters for Reckenholz lysimeters, Switzerland",
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
            vals = df_params.iloc[:, i].values.flatten().reshape(3, 1)
            v[:, :] = vals
            v.attrs.update(long_name=key, units=_UNITS[key])
    return


if __name__ == "__main__":
    main()
