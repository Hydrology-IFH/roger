from pathlib import Path
import h5netcdf
import datetime
import yaml
import numpy as onp
import pandas as pd
import click
import roger


_UNITS = {
    "z_soil": "mm",
    "slope": "-",
    "dmph": "1/m2",
    "dmpv": "1/m2",
    "lmpv": "mm",
    "theta_ac": "-",
    "theta_ufc": "-",
    "theta_pwp": "-",
    "ks": "mm/hour",
    "kf": "mm/hour",
}


@click.option("-nx", "--nrows", type=int, default=12)
@click.option("-ny", "--ncols", type=int, default=24)
@click.command("main")
def main(nrows, ncols):
    base_path = Path(__file__).parent

    file_param_bounds = base_path / "param_bounds.yml"
    with open(file_param_bounds, "r") as file:
        bounds = yaml.safe_load(file)

    # write parameters to netcdf and csv
    df_params = pd.DataFrame(index=range(nrows * ncols))
    RNG = onp.random.default_rng(42)
    file_params = base_path / "parameters.nc"
    with h5netcdf.File(file_params, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title="RoGeR model parameters",
            institution="University of Freiburg, Chair of Hydrology",
            references="",
            comment="",
            model_structure="SVAT model with free drainage",
            roger_version=f"{roger.__version__}",
        )
        dict_dim = {"x": nrows, "y": ncols}
        f.dimensions = dict_dim
        v = f.create_variable("x", ("x",), onp.float32, compression="gzip", compression_opts=1)
        v.attrs["long_name"] = "x"
        v.attrs["units"] = "m"
        v[:] = onp.arange(dict_dim["x"]) * 5
        v = f.create_variable("y", ("y",), onp.float32, compression="gzip", compression_opts=1)
        v.attrs["long_name"] = "y"
        v.attrs["units"] = "m"
        v[:] = onp.arange(dict_dim["y"]) * 5
        df_params.loc[:, "lu_id"] = 8
        for i, param in enumerate(bounds.keys()):
            # generate random values
            if param in ["z_soil", "lmpv", "dmpv", "dmph"]:
                values = (
                    RNG.uniform(bounds[param][0], bounds[param][1], size=dict_dim["x"] * dict_dim["y"])
                    .reshape((dict_dim["x"], dict_dim["y"]))
                    .astype(onp.int32)
                )
            elif param in ["slope"]:
                values = onp.round(
                    RNG.uniform(bounds[param][0], bounds[param][1], size=dict_dim["x"] * dict_dim["y"])
                    .reshape((dict_dim["x"], dict_dim["y"]))
                    .astype(onp.float32),
                    2,
                )
            else:
                values = (
                    RNG.uniform(bounds[param][0], bounds[param][1], size=dict_dim["x"] * dict_dim["y"])
                    .reshape((dict_dim["x"], dict_dim["y"]))
                    .astype(onp.float32)
                )
            # write parameters to netcdf
            v = f.create_variable(param, ("x", "y"), onp.float32, compression="gzip", compression_opts=1)
            v[:, :] = values
            v.attrs.update(units=_UNITS[param])
            # write parameters to dataframe
            df_params.loc[:, param] = values.flatten()

    df_params.loc[:, 'prec_weight'] = 1
    df_params.loc[:, 'ta_weight'] = 1
    df_params.loc[:, 'pet_weight'] = 1

    # write parameters to csv
    df_params.columns = [
        ["", "[mm]", "[-]", "[1/m2]", "[1/m2]", "[mm]", "[-]", "[-]", "[-]", "[mm/hour]", "[mm/hour]", "[-]", "[-]", "[-]"],
        ["lu_id", "z_soil", "slope", "dmph", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "kf", "prec_weight", "ta_weight", "pet_weight"],
    ]
    df_params.to_csv(base_path / "parameters.csv", index=False, sep=";")
    return


if __name__ == "__main__":
    main()
