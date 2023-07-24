from pathlib import Path
import pandas as pd
import yaml
from SALib import ProblemSpec
import numpy as onp
import click


_UNITS = {
    "c_root": "-",
    "c_int": "-",
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
    file_path = base_path / "param_bounds.yml"
    with open(file_path, "r") as file:
        bounds = yaml.safe_load(file)

    # generate salteilli parameter samples
    nsamples = config["nsamples"]
    bounds["outputs"] = ["Y"]
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
    df_params.loc[:, "lu_id"] = config["lu_id"]
    df_params.loc[:, "z_soil"] = config["z_soil"]
    df_params.loc[:, "kf"] = config["kf"]
    # set air capacity and usable field capacity (-)
    df_params.loc[:, "theta_ac"] = df_params.loc[:, "frac_lp"] * df_params.loc[:, "theta_eff"]
    df_params.loc[:, "theta_ufc"] = (1 - df_params.loc[:, "frac_lp"]) * df_params.loc[:, "theta_eff"]
    df_params.loc[:, "z_root"] = df_params.loc[:, "c_root"] * config["z_root"]
    # set storaga capacity of lower interception storage (mm)
    df_params.loc[:, "S_int_ground_tot"] = df_params.loc[:, "c_int"] * config["z_root"]

    df_params = df_params.loc[
        :,
        [
            "lu_id",
            "z_soil",
            "dmpv",
            "lmpv",
            "theta_ac",
            "theta_ufc",
            "theta_pwp",
            "ks",
            "kf",
            "theta_eff",
            "frac_lp",
            "z_root",
            "c_root",
            "S_int_ground_tot",
            "c_int",
        ],
    ]

    # write parameters to csv
    df_params.columns = [
        [
            "",
            "[mm]",
            "[1/m2]",
            "[mm]",
            "[-]",
            "[-]",
            "[-]",
            "[mm/hour]",
            "[mm/hour]",
            "[-]",
            "[-]",
            "[mm]",
            "[-]",
            "[mm]",
            "[-]",
        ],
        [
            "lu_id",
            "z_soil",
            "dmpv",
            "lmpv",
            "theta_ac",
            "theta_ufc",
            "theta_pwp",
            "ks",
            "kf",
            "theta_eff",
            "frac_lp",
            "z_root",
            "c_root",
            "S_int_ground_tot",
            "c_int",
        ],
    ]
    df_params.to_csv(base_path / "parameters.csv", index=False, sep=";")
    return


if __name__ == "__main__":
    main()
