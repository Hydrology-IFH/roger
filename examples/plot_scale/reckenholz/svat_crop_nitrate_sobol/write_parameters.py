from pathlib import Path
import yaml
from SALib import ProblemSpec
import numpy as onp
import pandas as pd
import click
import h5netcdf
import datetime


_UNITS = {
    "alpha_transp": "-",
    "alpha_q": "-",
    "km_denit": "-",
    "dmax_denit": "kg N/ha/a",
    "km_nit": "-",
    "dmax_nit": "kg N/ha/a",
    "kmin": "-",
    "kfix": "kg N/ha/year",
    "kngl": "kg N/ha/year",
    "phi_soil_temp": "day of year",
    "clay": "-",
    "soil_fertility": "",
    "z_soil": "mm",
    "k_transp": "-",
    "k_q": "-",
}

@click.option("-ns", "--nsamples", type=int, default=2**9)
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
        dict_bounds = yaml.safe_load(file)
        list_names = [key for key in dict_bounds.keys()]
        list_param_bounds = [dict_bounds[key] for key in dict_bounds.keys()]
        bounds = {'names': list_names, 'bounds': list_param_bounds}

    # generate salteilli parameter samples
    nsamples = nsamples
    bounds["outputs"] = ["Y"]
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

    # write parameters to csv
    df_params.loc[:, "phi_soil_temp"] = phi_soil_temp
    df_params.loc[:, "clay"] = clay
    df_params.loc[:, "soil_fertility"] = soil_fertility
    df_params.loc[:, "z_soil"] = zsoil
    df_params.loc[:, "kfix"] = 40.
    df_params.loc[:, "kngl"] = 75. * (1 - c_clay)

    param_names = ["alpha_transp", "alpha_q", "k_transp", "k_q_rz", "k_q_ss", "km_denit", "km_nit", "kmin", "kfix", "kngl", "dmax_denit", "dmax_nit", "phi_soil_temp", "clay", "soil_fertility", "z_soil"]
    df_params = df_params.loc[:, param_names]

    df_params.columns = [
        ["[-]", "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[day of year]", "[-]", "", "[mm]"], 
        param_names,
    ]

    df_params.columns = [
        [_UNITS[key] for key in param_names],
        [key for key in param_names],
    ]
    df_params.to_csv(base_path / f"parameters.csv", index=False, sep=";")

    # write parameters to netcdf
    params_file = base_path / f"parameters.nc"
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
