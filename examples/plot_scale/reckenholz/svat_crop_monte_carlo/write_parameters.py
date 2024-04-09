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
    "c_canopy_growth": "-",
    "c_root_growth": "-",
    "c_basal_crop_coeff": "-",
    "c_pet": "-",
    "zroot_to_zsoil_max": "-",
}

@click.option("-ns", "--nsamples", type=int, default=10000)
@click.command("main")
def main(nsamples):
    base_path = Path(__file__).parent
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
        if param in ["lmpv", "dmpv"]:
            values = (
                RNG.uniform(bounds[param][0], bounds[param][1], size=nrows).reshape((nrows, ncols)).astype(onp.int32)
            )
        else:
            values = (
                RNG.uniform(bounds[param][0], bounds[param][1], size=nrows).reshape((nrows, ncols)).astype(onp.float32)
            )
        # write parameters to dataframe
        df_params.loc[:, param] = values.flatten()

    # set constant parameters
    # df_params.loc[:, "z_soil"] = 1350
    df_params.loc[:, "kf"] = 2500
    # set air capacity and usable field capacity
    df_params.loc[:, "theta_ac"] = df_params.loc[:, "frac_lp"] * df_params.loc[:, "theta_eff"]
    df_params.loc[:, "theta_ufc"] = (1 - df_params.loc[:, "frac_lp"]) * df_params.loc[:, "theta_eff"]

    df_params = df_params.loc[:, ["dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "kf", "c_canopy_growth", "c_root_growth", "c_basal_crop_coeff", "c_pet", "zroot_to_zsoil_max"]]

    # write parameters to csv
    df_params.columns = [
        ["[1/m2]", "[mm]", "[-]", "[-]", "[-]", "[mm/hour]", "[mm/hour]", "[-]", "[-]", "[-]", "[-]", "[-]"],
        ["dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "kf", "c_canopy_growth", "c_root_growth", "c_basal_crop_coeff", "c_pet", "zroot_to_zsoil_max"],
    ]
    df_params.to_csv(base_path / "parameters.csv", index=False, sep=";")
    return


if __name__ == "__main__":
    main()
