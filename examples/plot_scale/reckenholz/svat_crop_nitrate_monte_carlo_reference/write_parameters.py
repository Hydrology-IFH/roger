from pathlib import Path
import yaml
import numpy as onp
import pandas as pd
import click
import datetime
import h5netcdf

_UNITS = {
    "alpha_transp": "-",
    "alpha_q": "-",
    "k_transp": "-",
    "k_q_rz": "-",
    "k_q_ss": "-",
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
    "alpha_q": 0.7,
    "k_transp": 0.6,
    "k_q_rz": 1.5,
    "k_q_ss": 1.5,
    "c2_transp": 0.6,
    "c2_q_rz": 1.5,
    "c2_q_ss": 1.5,
    "km_denit": [2.5, 18.7],
    "km_nit": [2.5, 12],
    "kmin": [10, 30],
    "kfix": 40,
    "kngl": 75,
    "dmax_denit": [10, 40],
    "dmax_nit": [10, 60],
    "phi_soil_temp": 91,
}


@click.option("-ns", "--nsamples", type=int, default=3)
@click.command("main")
def main(nsamples):
    base_path = Path(__file__).parent

    zsoil = 1350
    soil_fertility = 3
    clay = 0.2822544050076512
    c_clay = 0.2822544050076512 / (0.4 - 0.02)
    phi_soil_temp = 91

    # load parameter boundaries
    file_param_bounds = base_path / "param_bounds.yml"
    with open(file_param_bounds, "r") as file:
        bounds = yaml.safe_load(file)

    nrows = nsamples
    ncols = 1

    # write parameters to csv
    df_params = pd.DataFrame(index=range(nrows * ncols))
    RNG = onp.random.default_rng(42)
    for i, param in enumerate(bounds.keys()):
        # generate random values
        values = (
            RNG.uniform(bounds[param][0], bounds[param][1], size=nrows).reshape((nrows, ncols)).astype(onp.float32)
        )
        # write parameters to dataframe
        df_params.loc[:, param] = values.flatten()

    # write parameters to csv
    df_params.loc[:, "phi_soil_temp"] = phi_soil_temp
    df_params.loc[:, "clay"] = clay
    df_params.loc[:, "soil_fertility"] = soil_fertility
    df_params.loc[:, "z_soil"] = zsoil
    df_params.loc[:, "kfix"] = 40.
    df_params.loc[:, "kngl"] = 75. * (1 - c_clay)

    param_names = ["alpha_transp", "alpha_q", "km_denit", "km_nit", "kmin", "kfix", "kngl", "dmax_denit", "dmax_nit", "phi_soil_temp", "clay", "soil_fertility", "z_soil"]
    df_params = df_params.loc[:, param_names]

    # write parameters to csv
    df_params.columns = [
        ["[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[day of year]", "[-]", "", "[mm]"], 
        param_names,
    ]
    df_params.columns = [
        [_UNITS[key] for key in param_names],
        [key for key in param_names],
    ]
    df_params.to_csv(base_path / "parameters.csv", index=False, sep=";")

    # write parameters to netcdf
    params_file = base_path / "parameters.nc"
    with h5netcdf.File(params_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title="RoGeR SAS and nitrate parameters for Reckenholz lysimeters, Switzerland",
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
            vals = df_params.iloc[:, i].values.flatten().reshape(nrows, 1)
            v[:, :] = vals
            v.attrs.update(long_name=key, units=_UNITS[key])    
    return


if __name__ == "__main__":
    main()
