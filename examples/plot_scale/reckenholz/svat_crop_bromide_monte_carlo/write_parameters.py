from pathlib import Path
import yaml
import numpy as onp
import pandas as pd
import click


_UNITS = {
    "alpha_transp": "-",
    "alpha_q": "-",
    "k_transp": "-",
    "k_q_rz": "-",
    "k_q_ss": "-",
    "c1_transp": "-",
    "c2_transp": "-",
    "c1_q_rz": "-",
    "c2_q_rz": "-",
    "c1_q_ss": "-",
    "c2_q_ss": "-",
}
@click.option("-ns", "--nsamples", type=int, default=10000)
@click.command("main")
def main(nsamples):
    base_path = Path(__file__).parent
    for transport_model_structure in ['complete-mixing', 'advection-dispersion-power', 'time-variant_advection-dispersion-power']:
        # load parameter boundaries
        file_param_bounds = base_path / "param_bounds.yml"
        with open(file_param_bounds, "r") as file:
            bounds = yaml.safe_load(file)[transport_model_structure]

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
        df_params.columns = [
            [_UNITS[key] for key in bounds.keys()],
            [key for key in bounds.keys()],
        ]
        df_params.to_csv(base_path / f"parameters_for_{transport_model_structure}.csv", index=False, sep=";")
    return


if __name__ == "__main__":
    main()
