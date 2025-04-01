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
@click.option("-ns", "--nsamples", type=int, default=1)
@click.command("main")
def main(nsamples):
    base_path = Path(__file__).parent

    lys_experiments = ["lys2", "lys3", "lys8"]
    for lys_experiment in lys_experiments:
        zsoil = 1350
        soil_fertility = 3
        clay = 0.2822544050076512
        c_clay = clay / (0.4 - 0.02)
        phi_soil_temp = 91

        for transport_model_structure in ['complete-mixing', 'advection-dispersion-power', 'time-variant_advection-dispersion-power']:
            # load parameter boundaries
            file_param_bounds = base_path / "param_bounds.yml"
            with open(file_param_bounds, "r") as file:
                bounds = yaml.safe_load(file)[transport_model_structure]

            df_params = pd.DataFrame(index=range(nsamples))

            nrows = nsamples
            ncols = 1
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

            if transport_model_structure == 'complete-mixing':
                param_names = ["alpha_transp_sugarbeet", "alpha_q_sugarbeet", "km_denit_sugarbeet", "km_nit_sugarbeet", "dmax_denit_sugarbeet", "dmax_nit_sugarbeet",
                               "alpha_transp_winter-wheat", "alpha_q_winter-wheat", "km_denit_winter-wheat", "km_nit_winter-wheat", "dmax_denit_winter-wheat", "dmax_nit_winter-wheat",
                               "alpha_transp_winter-rape", "alpha_q_winter-rape", "km_denit_winter-rape", "km_nit_winter-rape", "dmax_denit_winter-rape", "dmax_nit_winter-rape",
                               "alpha_transp_winter-triticale", "alpha_q_winter-triticale", "km_denit_winter-triticale", "km_nit_winter-triticale", "dmax_denit_winter-triticale", "dmax_nit_winter-triticale",
                               "alpha_transp_silage-corn", "alpha_q_silage-corn", "km_denit_silage-corn", "km_nit_silage-corn", "dmax_denit_silage-corn", "dmax_nit_silage-corn",
                               "alpha_transp_winter-barley", "alpha_q_winter-barley", "km_denit_winter-barley", "km_nit_winter-barley", "dmax_denit_winter-barley", "dmax_nit_winter-barley",
                               "alpha_transp_green-manure", "alpha_q_green-manure", "km_denit_green-manure", "km_nit_green-manure", "dmax_denit_green-manure", "dmax_nit_green-manure",
                               "alpha_transp_bare", "alpha_q_bare", "km_denit_bare", "km_nit_bare", "dmax_denit_bare", "dmax_nit_bare",
                               "kmin", "kfix", "kngl", "phi_soil_temp", "clay", "soil_fertility", "z_soil"]
                df_params = df_params.loc[:, param_names]

                df_params.columns = [
                    ["[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]",
                     "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", 
                     "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", 
                     "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", 
                     "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", 
                     "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]",  
                     "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]",
                     "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]",    
                     "[kg N/ha/year]","[kg N/ha/year]", "[kg N/ha/year]", "[day of year]", "[-]", "", "[mm]"],                     
                     param_names,
                ]

            elif transport_model_structure == 'advection-dispersion-power':
                param_names = ["alpha_transp_sugarbeet", "alpha_q_sugarbeet", "k_transp_sugarbeet", "k_q_sugarbeet", "km_denit_sugarbeet", "km_nit_sugarbeet", "dmax_denit_sugarbeet", "dmax_nit_sugarbeet",
                               "alpha_transp_winter-wheat", "alpha_q_winter-wheat", "k_transp_winter-wheat", "k_q_winter-wheat", "km_denit_winter-wheat", "km_nit_winter-wheat", "dmax_denit_winter-wheat", "dmax_nit_winter-wheat",
                               "alpha_transp_winter-rape", "alpha_q_winter-rape", "k_transp_winter-rape", "k_q_winter-rape", "km_denit_winter-rape", "km_nit_winter-rape", "dmax_denit_winter-rape", "dmax_nit_winter-rape",
                               "alpha_transp_winter-triticale", "alpha_q_winter-triticale", "k_transp_winter-triticale", "k_q_winter-triticale", "km_denit_winter-triticale", "km_nit_winter-triticale", "dmax_denit_winter-triticale", "dmax_nit_winter-triticale",
                               "alpha_transp_silage-corn", "alpha_q_silage-corn", "k_transp_silage-corn", "k_q_silage-corn", "km_denit_silage-corn", "km_nit_silage-corn", "dmax_denit_silage-corn", "dmax_nit_silage-corn",
                               "alpha_transp_winter-barley", "alpha_q_winter-barley", "k_transp_winter-barley", "k_q_winter-barley", "km_denit_winter-barley", "km_nit_winter-barley", "dmax_denit_winter-barley", "dmax_nit_winter-barley",
                               "alpha_transp_green-manure", "alpha_q_green-manure", "k_transp_green-manure", "k_q_green-manure", "km_denit_green-manure", "km_nit_green-manure", "dmax_denit_green-manure", "dmax_nit_green-manure",
                               "alpha_transp_bare", "alpha_q_bare", "k_transp_bare", "k_q_bare", "km_denit_bare", "km_nit_bare", "dmax_denit_bare", "dmax_nit_bare",
                               "kmin", "kfix", "kngl", "phi_soil_temp", "clay", "soil_fertility", "z_soil"]
                df_params = df_params.loc[:, param_names]

                df_params.columns = [[
                     "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]",
                     "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", 
                     "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", 
                     "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", 
                     "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", 
                     "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]",  
                     "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]",
                     "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]",    
                     "[kg N/ha/year]","[kg N/ha/year]", "[kg N/ha/year]", "[day of year]", "[-]", "", "[mm]"], 
                    param_names,
                ]

            elif transport_model_structure == 'time-variant_advection-dispersion-power':
                param_names = ["alpha_transp_sugarbeet", "alpha_q_sugarbeet", "c1_transp_sugarbeet", "c2_transp_sugarbeet", "c1_q_sugarbeet", "c2_q_sugarbeet", "km_denit_sugarbeet", "km_nit_sugarbeet", "dmax_denit_sugarbeet", "dmax_nit_sugarbeet",
                               "alpha_transp_winter-wheat", "alpha_q_winter-wheat", "c1_transp_winter-wheat", "c2_transp_winter-wheat", "c1_q_winter-wheat", "c2_q_winter-wheat", "km_denit_winter-wheat", "km_nit_winter-wheat", "dmax_denit_winter-wheat", "dmax_nit_winter-wheat",
                               "alpha_transp_winter-rape", "alpha_q_winter-rape", "c1_transp_winter-rape", "c2_transp_winter-rape", "c1_q_winter-rape", "c2_q_winter-rape", "km_denit_winter-rape", "km_nit_winter-rape", "dmax_denit_winter-rape", "dmax_nit_winter-rape",
                               "alpha_transp_winter-triticale", "alpha_q_winter-triticale", "c1_transp_winter-triticale", "c2_transp_winter-triticale", "c1_q_winter-triticale", "c2_q_winter-triticale", "km_denit_winter-triticale", "km_nit_winter-triticale", "dmax_denit_winter-triticale", "dmax_nit_winter-triticale",
                               "alpha_transp_silage-corn", "alpha_q_silage-corn", "c1_transp_silage-corn", "c2_transp_silage-corn", "c1_q_silage-corn", "c2_q_silage-corn", "km_denit_silage-corn", "km_nit_silage-corn", "dmax_denit_silage-corn", "dmax_nit_silage-corn",
                               "alpha_transp_winter-barley", "alpha_q_winter-barley", "c1_transp_winter-barley", "c2_transp_winter-barley", "c1_q_winter-barley", "c2_q_winter-barley", "km_denit_winter-barley", "km_nit_winter-barley", "dmax_denit_winter-barley", "dmax_nit_winter-barley",
                               "alpha_transp_green-manure", "alpha_q_green-manure", "c1_transp_green-manure", "c2_transp_green-manure", "c1_q_green-manure", "c2_q_green-manure", "km_denit_green-manure", "km_nit_green-manure", "dmax_denit_green-manure", "dmax_nit_green-manure",
                               "alpha_transp_bare", "alpha_q_bare", "km_denit_bare", "c1_transp_bare", "c2_transp_bare", "c1_q_bare", "c2_q_bare", "km_nit_bare", "dmax_denit_bare", "dmax_nit_bare",
                               "kmin", "kfix", "kngl", "phi_soil_temp", "clay", "soil_fertility", "z_soil"]
                df_params = df_params.loc[:, param_names]

                df_params.columns = [[
                     "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]",
                     "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", 
                     "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", 
                     "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", 
                     "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", 
                     "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]",  
                     "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]",
                     "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]",    
                     "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[day of year]", "[-]", "", "[mm]"], 
                    param_names,
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
                for i, _key in enumerate(param_names):
                    if _key not in ["kmin", "kfix", "kngl", "phi_soil_temp", "clay", "soil_fertility", "z_soil"]:
                        ll = _key.split("_")[:-1]
                        key = "_".join(ll)
                    else:
                        key = _key
                    v = f.create_variable(
                        _key, ("x", "y"), onp.float32, compression="gzip", compression_opts=1
                    )
                    vals = df_params.iloc[:, i].values.flatten().reshape(nrows, 1)
                    v[:, :] = vals
                    v.attrs.update(long_name=_key, units=_UNITS[key])    

    return


if __name__ == "__main__":
    main()
