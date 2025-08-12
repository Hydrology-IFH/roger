from pathlib import Path
import os
import xarray as xr
import numpy as onp
from cftime import num2date
import pandas as pd
import yaml
import click

_dict_locations = {
    "pfullendorf": "krauchenwies",
    "lahr": "orschweier",
    "bruchsal-heidelsheim": "kraichtal",
    "nagold": "tailfingen",
    "sachsenheim": "boennigheim",
    "heidelberg": "ladenburg",
    "mergentheim": "boxberg",
    "grosserlach-mannenweiler": "kupferzell",
    "wutoeschingen-ofteringen": "doeggingen",
    "ulm": "eiselau"
    }


@click.command("main")
def main():
    base_path = Path(__file__).parent.parent
    dir_name = os.path.basename(str(Path(__file__).parent))

    # load the parameters
    file = base_path / "parameters.csv"
    df_parameters = pd.read_csv(file, sep=";", skiprows=1, index_col=0)
    file = base_path / "crop_water_stress.csv"
    df_crop_water_stress = pd.read_csv(file, sep=";", skiprows=1, index_col=0)
    df_crop_water_stress["crop_type"] = df_crop_water_stress.index
    df_crop_water_stress.index = df_crop_water_stress.loc[:, "lu_id"]
    dict_crop_types = df_crop_water_stress.loc[:, "crop_type"].to_frame().to_dict()["crop_type"]
    df_crop_water_stress = pd.read_csv(file, sep=";", skiprows=1, index_col=0)
    df_crop_water_stress.index = df_crop_water_stress.loc[:, "lu_id"]
    dict_crop_water_stress = df_crop_water_stress.loc[:, "water_stress_frac"].to_frame().to_dict()["water_stress_frac"]
    file = base_path / "crop_heat_stress.csv"
    df_crop_heat_stress = pd.read_csv(file, sep=";", skiprows=1, index_col=0)
    df_crop_heat_stress.index = df_crop_heat_stress.loc[:, "lu_id"]
    dict_crop_heat_stress = df_crop_heat_stress.loc[:, "ta_heat_stress"].to_frame().to_dict()["ta_heat_stress"]

    # load the configuration file
    with open(base_path / "config.yml", "r") as file:
        config = yaml.safe_load(file)

    # identifiers of the simulations
    locations = _dict_locations.keys()
    irrigation_scenarios = config["irrigation_scenarios"]
    crop_rotation_scenarios = config["crop_rotation_scenarios"]
    soil_types = df_parameters.index.to_list()
    for location in locations:
        _location = _dict_locations[location]
        for irrigation_scenario in irrigation_scenarios:
            if irrigation_scenario == "20-ufc":
                c_irr = 0.2
            elif irrigation_scenario == "30-ufc":
                c_irr = 0.30
            elif irrigation_scenario == "50-ufc":
                c_irr = 0.50
            if os.path.exists(str(base_path / "output" / dir_name / irrigation_scenario)):
                for crop_rotation_scenario in crop_rotation_scenarios:
                    roger_file = (
                        base_path
                        / "output" 
                        / dir_name 
                        / irrigation_scenario
                        / f"SVATCROP_{location}_{crop_rotation_scenario}.nc"
                    )
                    print(roger_file)
                    if os.path.exists(roger_file):
                        ds = xr.open_dataset(roger_file, engine="h5netcdf")
                        # assign date
                        days = ds["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
                        date = num2date(
                            days,
                            units=f"days since {ds['Time'].attrs['time_origin']}",
                            calendar="standard",
                            only_use_cftime_datetimes=False,
                        )
                        ds = ds.assign_coords(Time=("Time", date))
                        dir_csv_files = base_path / "output" / dir_name / irrigation_scenario / crop_rotation_scenario / _location
                        if not os.path.exists(dir_csv_files):
                            os.makedirs(dir_csv_files)
                        for x, soil_type in enumerate(soil_types):
                            if soil_type.lower() == _location:
                                # write simulation to csv
                                df_simulation = pd.DataFrame(
                                    index=date, columns=["precip", "pet", "pt", "photosynthesis_index", "canopy_cover", "z_root", "theta_fc", "theta_irrig", "theta_rz", "irrig", "irrigation_demand", "root_ventilation", "ta_max", "heat_stress", "transp", "evap_soil", "perc", "lu_id", "crop_type"]
                                )
                                cond_bare = (ds["lu_id"].isel(x=x, y=0).values == 599)
                                df_simulation.loc[:, "precip"] = onp.where(ds["irrig"].isel(x=x, y=0).values > 0, 0, ds["prec"].isel(x=x, y=0).values)
                                df_simulation.loc[:, "pet"] = ds["pet"].isel(x=x, y=0).values
                                df_simulation.loc[:, "pt"] = ds["pt"].isel(x=x, y=0).values
                                df_simulation.loc[:, "photosynthesis_index"] = ds["transp"].isel(x=x, y=0).values / ds["pt"].isel(x=x, y=0).values
                                df_simulation.loc[cond_bare, "photosynthesis_index"] = onp.nan
                                df_simulation.loc[:, "irrig"] = ds["irrig"].isel(x=x, y=0).values
                                canopy_cover = ds["ground_cover"].isel(x=x, y=0).values
                                canopy_cover[canopy_cover <= 0.03] = 0
                                df_simulation.loc[:, "canopy_cover"] = canopy_cover
                                df_simulation.loc[:, "z_root"] = ds["z_root"].isel(x=x, y=0).values
                                df_simulation.loc[:, "irrigation_demand"] = ds["irr_demand"].isel(x=x, y=0).values
                                df_simulation.loc[:, "theta_rz"] = ds["theta_rz"].isel(x=x, y=0).values
                                df_simulation.loc[:, "theta_fc"] = df_parameters.loc[f"{soil_type}", "theta_pwp"] + df_parameters.loc[f"{soil_type}", "theta_ufc"]
                                df_simulation.loc[:, "transp"] = ds["transp"].isel(x=x, y=0).values
                                df_simulation.loc[:, "evap_soil"] = ds["evap_soil"].isel(x=x, y=0).values
                                df_simulation.loc[:, "perc"] = ds["q_ss"].isel(x=x, y=0).values
                                df_simulation.loc[:, "ta_max"] = ds["ta_max"].isel(x=x, y=0).values
                                # calculate heat stress of crops
                                df_simulation.loc[:, "heat_stress"] = 0
                                ta_heat_stress_list = [dict_crop_heat_stress[lu_id] for lu_id in ds["lu_id"].isel(x=x, y=0).values]
                                ta_heat_stress = onp.array(ta_heat_stress_list)
                                cond = (df_simulation.loc[:, "ta_max"].values >= ta_heat_stress)
                                df_simulation.loc[cond, "heat_stress"] = 1
                                df_simulation.loc[:, "lu_id"] = ds["lu_id"].isel(x=x, y=0).values
                                df_simulation.loc[:, "crop_type"] = [dict_crop_types[lu_id] for lu_id in ds["lu_id"].isel(x=x, y=0).values]
                                root_ventilation = (ds["theta_rz"].isel(x=x, y=0).values - (df_parameters.loc[f"{soil_type}", "theta_pwp"] + df_parameters.loc[f"{soil_type}", "theta_ufc"])) / df_parameters.loc[f"{soil_type}", "theta_ac"]
                                root_ventilation[root_ventilation < 0] = 0
                                root_ventilation[root_ventilation > 1] = 1
                                root_ventilation = (1 - root_ventilation) * 100
                                root_ventilation[cond_bare] = onp.nan
                                df_simulation.loc[:, "root_ventilation"] = root_ventilation
                                if irrigation_scenario in ["35-ufc", "45-ufc", "50-ufc", "80-ufc"]:
                                    theta_irr = df_parameters.loc[f"{soil_type}", "theta_pwp"] + (c_irr * df_parameters.loc[f"{soil_type}", "theta_ufc"])
                                    df_simulation.loc[:, "theta_irrig"] = theta_irr
                                    df_simulation.loc[cond_bare, "theta_irrig"] = onp.nan
                                else:
                                    c_irr_list = [dict_crop_water_stress[lu_id] for lu_id in ds["lu_id"].isel(x=x, y=0).values]
                                    c_irr = onp.array(c_irr_list)
                                    theta_irr = df_parameters.loc[f"{soil_type}", "theta_pwp"] + (c_irr * df_parameters.loc[f"{soil_type}", "theta_ufc"])
                                    df_simulation.loc[:, "theta_irrig"] = theta_irr
                                    df_simulation.loc[cond_bare, "theta_irrig"] = onp.nan
                                df_simulation.columns =[["[mm/day]", "[mm/day]", "[mm/day]", "[-]", "[-]", "[mm]", "[-]", "[-]", "[-]", "[mm/day]", "[mm]", "[%]", "[degC]", "[day]", "[mm/day]", "[mm/day]", "[mm/day]", "", ""],
                                                        ["precip", "pet", "pt", "photosynthesis_index", "canopy_cover", "z_root", "theta_fc", "theta_irrig", "theta_rz", "irrig", "irrigation_demand", "root_ventilation", "ta_max", "heat_stress", "transp", "evap_soil", "perc", "lu_id", "crop_type"]]
                                df_simulation = df_simulation.iloc[1:, :] # remove initial values
                                df_simulation.to_csv(
                                    dir_csv_files / "simulation.csv", sep=";"
                                )
                                print(str(dir_csv_files / "simulation.csv") + " written")

    return


if __name__ == "__main__":
    main()
