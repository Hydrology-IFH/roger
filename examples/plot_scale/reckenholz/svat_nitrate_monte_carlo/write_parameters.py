from pathlib import Path
import yaml
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
    "c1_transp": "-",
    "c2_transp": "-",
    "c1_q": "-",
    "c2_q": "-",
}
@click.option("-ns", "--nsamples", type=int, default=10000)
@click.command("main")
def main(nsamples):
    base_path = Path(__file__).parent

    lys_experiments = ["lys2", "lys3", "lys8"]
    for lys_experiment in lys_experiments:
        file = Path("/Volumes/LaCie/roger/examples/plot_scale/reckenholz") / "output" / "svat_monte_carlo" / f"params_eff_{lys_experiment}_weekly.txt"
        df_params_metrics = pd.read_csv(file, sep="\t")

        # calculate multi-objective efficiency
        df_params_metrics["E_multi"] = 0.7 * df_params_metrics["KGE_q_ss_perc_pet_2011-2017"] + 0.3 * (1 - ((1 - df_params_metrics["r_S_perc_pet_2011-2017"])**2 + (1 - df_params_metrics["KGE_alpha_S_perc_pet_2011-2017"])**2)**(0.5))

        # select best 100 model runs
        df_params_metrics = df_params_metrics.sort_values(by=["E_multi"], ascending=False)
        df_params_metrics.loc[:, "id"] = range(len(df_params_metrics.index))
        idx_best100 = df_params_metrics.loc[:df_params_metrics.index[99], "id"].values.tolist()
        df_params_metrics_best100 = df_params_metrics.loc[idx_best100, :]
        df_params_best100 = df_params_metrics_best100.loc[:, ["dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "kf", "clay"]]

        zsoil = 1350
        soil_fertility = 3
        clay = df_params_best100["clay"].values.astype(onp.float64)
        c_clay = clay / (0.4 - 0.02)
        phi_soil_temp = 91

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
            df_params.loc[:, "phi_soil_temp"] = phi_soil_temp
            df_params.loc[:, "clay"] = onp.repeat(clay, int(nsamples/100), axis=0).flatten()
            df_params.loc[:, "soil_fertility"] = soil_fertility
            df_params.loc[:, "z_soil"] = zsoil
            df_params.loc[:, "kfix"] = 40.
            df_params.loc[:, "kngl"] = 75. * (1 - onp.repeat(c_clay, int(nsamples/100), axis=0).flatten())

            if transport_model_structure == 'complete-mixing':
                param_names = ["alpha_transp", "alpha_q", "km_denit", "km_nit", "kmin", "kfix", "kngl", "dmax_denit", "dmax_nit", "phi_soil_temp", "clay", "soil_fertility", "z_soil"]
                df_params = df_params.loc[:, param_names]

                df_params.columns = [
                    ["[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[day of year]", "[-]", "", "[mm]"], 
                    param_names,
                ]

            elif transport_model_structure == 'advection-dispersion-power':
                param_names = ["alpha_transp", "alpha_q", "k_transp", "k_q", "km_denit", "km_nit", "kmin", "kfix", "kngl", "dmax_denit", "dmax_nit", "phi_soil_temp", "clay", "soil_fertility", "z_soil"]
                df_params = df_params.loc[:, param_names]

                df_params.columns = [
                    ["[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[day of year]", "[-]", "", "[mm]"], 
                    param_names,
                ]

            elif transport_model_structure == 'time-variant_advection-dispersion-power':
                param_names = ["alpha_transp", "alpha_q", "c1_transp", "c2_transp", "c1_q", "c2_q", "km_denit", "km_nit", "kmin", "kfix", "kngl", "dmax_denit", "dmax_nit", "phi_soil_temp", "clay", "soil_fertility", "z_soil"]
                df_params = df_params.loc[:, param_names]

                df_params.columns = [
                    ["[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[day of year]", "[-]", "", "[mm]"], 
                    param_names,
                ]

            df_params.columns = [
                [_UNITS[key] for key in param_names],
                [key for key in param_names],
            ]
            df_params.to_csv(base_path / f"parameters_for_{transport_model_structure}_{lys_experiment}.csv", index=False, sep=";")

            # write parameters to netcdf
            params_file = base_path / f"parameters_for_{transport_model_structure}_{lys_experiment}.nc"
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
