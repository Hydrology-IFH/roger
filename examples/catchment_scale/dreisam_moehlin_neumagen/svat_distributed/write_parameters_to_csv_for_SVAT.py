from pathlib import Path
import xarray as xr
import numpy as onp
import pandas as pd
import click


_UNITS = {
    "z_soil": "mm",
    "dmpv": "1/m2",
    "lmpv": "mm",
    "theta_ac": "-",
    "theta_ufc": "-",
    "theta_pwp": "-",
    "ks": "mm/hour",
    "kf": "mm/hour",
    "ta_offset": "degC",
    "pet_weight": "-",
    "prec_weight": "-",
    "station_id": "",
}


@click.command("main")
def main():
    base_path = Path(__file__).parent

    # load the netcdf file
    params_file = base_path / "parameters.nc"
    ds_params = xr.open_dataset(params_file)
    nrows = ds_params.sizes["y"]
    ncols = ds_params.sizes["x"]

    # write parameters to csv
    df_params = pd.DataFrame(index=range(nrows * ncols))
    df_params.loc[:, "lu_id"] = ds_params.lanu.values.flatten()
    df_params.loc[:, "sealing"] = ds_params.vers.values.flatten() / 100
    df_params.loc[:, "z_soil"] = ds_params.GRUND.values.flatten() * 10
    df_params.loc[:, "dmpv"] = ds_params.MPD_V.values.flatten()
    df_params.loc[:, "lmpv"] = ds_params.MPL_V.values.flatten()
    df_params.loc[:, "theta_ac"] = ds_params.LK.values.flatten()/100
    df_params.loc[:, "theta_ufc"] = ds_params.NFK.values.flatten()/100
    df_params.loc[:, "theta_pwp"] = ds_params.PWP.values.flatten()/100
    df_params.loc[:, "ks"] = ds_params.KS.values.flatten()
    df_params.loc[:, "kf"] = ds_params.TP.values.flatten()
    df_params.loc[:, "ta_offset"] = ds_params.F_t.values.flatten()
    df_params.loc[:, "pet_weight"] = ds_params.F_et.values.flatten() / 100
    df_params.loc[:, "prec_weight"] = ds_params.F_n_h_y.values.flatten() / 100
    df_params.loc[:, "ta_offset"] = df_params.loc[:, "ta_offset"].fillna(0)
    df_params.loc[:, "pet_weight"] = df_params.loc[:, "pet_weight"].fillna(0)
    df_params.loc[:, "prec_weight"] = df_params.loc[:, "prec_weight"].fillna(0)
    df_params.loc[:, "station_id"] = ds_params.STAT_ID.values.flatten().astype(int)

    df_params = df_params.loc[
        :,
        [   "lu_id",
            "sealing",
            "z_soil",
            "dmpv",
            "lmpv",
            "theta_ac",
            "theta_ufc",
            "theta_pwp",
            "ks",
            "kf",
            "ta_offset",
            "pet_weight",
            "prec_weight",
            "station_id",
        ],
    ]
    df_params.fillna(-9999, inplace=True)
    df_params["lu_id"] = df_params["lu_id"].astype(onp.int16)

    # write parameters to csv
    df_params.columns = [
        ["", "[-]", "[mm]", "[1/m2]", "[mm]", "[-]", "[-]", "[-]", "[mm/hour]", "[mm/hour]", "[degC]", "[-]", "[-]", ""],
        ["lu_id", "sealing", "z_soil", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "kf", "ta_offset", "pet_weight", "prec_weight", "station_id"],
    ]
    df_params.to_csv(base_path / "parameters.csv", index=False, sep=";")
    return


if __name__ == "__main__":
    main()
