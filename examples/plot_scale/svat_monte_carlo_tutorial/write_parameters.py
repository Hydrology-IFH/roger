from pathlib import Path
import yaml
import numpy as onp
import pandas as pd
import click


_UNITS = {
    "dmpv": "1/m2",
    "lmpv": "mm",
    "theta_eff": "-",
    "frac_lp": "-",
    "theta_pwp": "-",
    "ks": "mm/hour",
}


@click.command("main")
def main():
    base_path = Path(__file__).parent
    # load configuration file
    file_config = base_path / "config.yml"
    with open(file_config, "r") as file:
        config = yaml.safe_load(file)
    # load parameter boundaries
    file_param_bounds = base_path / "param_bounds.yml"
    with open(file_param_bounds, "r") as file:
        bounds = yaml.safe_load(file)

    nrows = config["nx"]
    ncols = 1

    # write parameters to csv
    df_params = pd.DataFrame(index=range(nrows * ncols))
    RNG = onp.random.default_rng(42)
    for i, param in enumerate(bounds.keys()):
        # generate random values
        if param in ["lmpv", "dmpv"]:
            values = (
                RNG.uniform(bounds[param][0], bounds[param][1], size=nrows).reshape((nrows, ncols)).astype(onp.int32)
            )
        elif param in ["slope"]:
            values = onp.round(
                RNG.uniform(bounds[param][0], bounds[param][1], size=nrows).reshape((nrows, ncols)).astype(onp.float32),
                2,
            )
        else:
            values = (
                RNG.uniform(bounds[param][0], bounds[param][1], size=nrows).reshape((nrows, ncols)).astype(onp.float32)
            )
        # write parameters to dataframe
        df_params.loc[:, param] = values.flatten()

    # set constant parameters
    df_params.loc[:, "lu_id"] = config["lu_id"]
    df_params.loc[:, "z_soil"] = config["z_soil"]
    df_params.loc[:, "kf"] = config["kf"]
    # set air capacity and usable field capacity
    df_params.loc[:, "theta_ac"] = df_params.loc[:, "frac_lp"] * df_params.loc[:, "theta_eff"]
    df_params.loc[:, "theta_ufc"] = (1 - df_params.loc[:, "frac_lp"]) * df_params.loc[:, "theta_eff"]

    df_params = df_params.loc[:, ["lu_id", "z_soil", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "kf"]]

    # write parameters to csv
    df_params.columns = [
        ["", "[mm]", "[1/m2]", "[mm]", "[-]", "[-]", "[-]", "[mm/hour]", "[mm/hour]"],
        ["lu_id", "z_soil", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "kf"],
    ]
    df_params.to_csv(base_path / "parameters.csv", index=False, sep=";")
    return


if __name__ == "__main__":
    main()
