from pathlib import Path
import yaml
from SALib import ProblemSpec
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
    "c_canopy": "-",
    "c_root": "-",
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
    df_params.loc[:, "z_soil"] = 1350
    df_params.loc[:, "kf"] = 2500
    # set air capacity and usable field capacity
    df_params.loc[:, "theta_ac"] = df_params.loc[:, "frac_lp"] * df_params.loc[:, "theta_eff"]
    df_params.loc[:, "theta_ufc"] = (1 - df_params.loc[:, "frac_lp"]) * df_params.loc[:, "theta_eff"]

    df_params = df_params.loc[:, ["z_soil", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "kf", "c_canopy", "c_root"]]

    # write parameters to csv
    df_params.columns = [
        ["[mm]", "[1/m2]", "[mm]", "[-]", "[-]", "[-]", "[mm/hour]", "[mm/hour]", "[-]", "[-]"],
        ["z_soil", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "kf", "c_canopy", "c_root"],
    ]
    df_params.to_csv(base_path / "parameters.csv", index=False, sep=";")
    return


if __name__ == "__main__":
    main()
