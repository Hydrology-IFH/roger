from pathlib import Path
import datetime
import yaml
from SALib import ProblemSpec
import numpy as onp
import pandas as pd
import h5netcdf
import click

def calc_clay_content(theta_pwp, theta_fc, theta_sat):

    # calculate pore-size distribution index
    lambda_bc = (
                onp.log(theta_fc / theta_sat)
                - onp.log(theta_pwp/ theta_sat)
            ) / (onp.log(15850) - onp.log(63))

    # calculate bubbling pressure
    ha = ((theta_pwp / theta_sat) ** (1.0 / lambda_bc) * (-15850))

    # calculate soil water content at pF = 6
    theta_6 = ((ha / (-(10**6))) ** lambda_bc * theta_sat)

    # calculate clay content
    clay = (0.71 * (theta_6 - 0.01) / 0.3)

    return clay


_UNITS = {
    "dmpv": "1/m2",
    "lmpv": "mm",
    "theta_eff": "-",
    "frac_lp": "-",
    "theta_pwp": "-",
    "theta_ufc": "-",
    "theta_ac": "-",
    "ks": "mm/hour",
    "c_canopy": "-",
    "c_root": "-",
    "alpha_transp": "-",
    "alpha_q": "-",
    "k_transp": "-",
    "k_q": "-",
    "km_denit": "kg N/ha/year",
    "km_nit": "kg N/ha/year",
    "kmin": "kg N/ha/year",
    "dmax_denit": "kg N/ha/year",
    "dmax_nit": "kg N/ha/year",
    "clay": "-",
}

@click.option("-ns", "--nsamples", type=int, default=2**9)
@click.command("main")
def main(nsamples):
    base_path = Path(__file__).parent
    # load parameter boundaries
    file_param_bounds = base_path / "param_bounds.yml"
    with open(file_param_bounds, "r") as file:
        bounds = yaml.safe_load(file)

    # generate parameter samples using a Sobol' sequence
    sp = ProblemSpec(bounds)
    params = sp.sample_sobol(nsamples, calc_second_order=False).samples
    nrows = params.shape[0]
    ncols = 1

    # write parameters to csv
    df_params = pd.DataFrame(index=range(nrows * ncols))
    for i, param in enumerate(bounds["names"]):
        if param in ["lmpv", "dmpv"]:
            values = params[:, i].reshape(nrows, ncols).astype(onp.int32)
        elif param in ["slope"]:
            values = onp.round(params[:, i].reshape((nrows, ncols)).astype(onp.float32), 2)
        else:
            values = params[:, i].reshape(nrows, ncols).astype(onp.float32)

        # write parameters to dataframe
        df_params.loc[:, param] = values.flatten()

    # set constant parameters
    # set air capacity and usable field capacity
    df_params.loc[:, "theta_ac"] = df_params.loc[:, "frac_lp"] * df_params.loc[:, "theta_eff"]
    df_params.loc[:, "theta_ufc"] = (1 - df_params.loc[:, "frac_lp"]) * df_params.loc[:, "theta_eff"]

    theta_pwp = df_params.loc[:, "theta_pwp"].values
    theta_fc = df_params.loc[:, "theta_pwp"].values + df_params.loc[:, "theta_ufc"].values
    theta_sat = df_params.loc[:, "theta_pwp"].values + df_params.loc[:, "theta_ufc"].values + df_params.loc[:, "theta_ac"].values
    df_params.loc[:, "clay"] = calc_clay_content(theta_pwp, theta_fc, theta_sat)

    param_names = ["dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "c_canopy", "c_root", "alpha_transp", "alpha_q", "k_transp", "k_q", "km_denit", "km_nit", "kmin", "dmax_denit", "dmax_nit", "clay", "frac_lp", "theta_eff"]
    df_params = df_params.loc[:, param_names]

    # write parameters to csv
    df_params.columns = [
        ["[1/m2]", "[mm]", "[-]", "[-]", "[-]", "[mm/hour]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[-]", "[-]", "[-]"],
        ["dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "c_canopy", "c_root", "alpha_transp", "alpha_q", "k_transp", "k_q", "km_denit", "km_nit", "kmin", "dmax_denit", "dmax_nit", "clay", "theta_eff", "frac_lp"],]
    df_params.to_csv(base_path / "parameters.csv", index=False, sep=";")

    # write parameters to netcdf
    params_file = base_path / "parameters.nc"
    with h5netcdf.File(params_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title="RoGeR SAS and nitrate parameters using Sobol' sampling method at Reckenholz lysimeter site, Switzerland",
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
            vals = df_params.iloc[:, i].values.flatten()
            v[:, 0] = vals
            v.attrs.update(long_name=key, units=_UNITS[key])
    return


if __name__ == "__main__":
    main()
