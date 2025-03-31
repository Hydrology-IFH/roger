from pathlib import Path
import os
import xarray as xr
import numpy as onp
from cftime import num2date
import pandas as pd
import yaml
import click


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

    # load the configuration file
    with open(base_path / "config.yml", "r") as file:
        config = yaml.safe_load(file)

    soil_compaction_scenarios = ["no_compaction", "compaction"]
    irrigation_scenarios = config["irrigation_scenarios"]
    irrigation_scenarios.append("no_irrigation")
    crop_rotation_scenarios = config["crop_rotation_scenarios"]

    soil_types = df_parameters.index.to_list()
    for soil_compaction_scenario in soil_compaction_scenarios:
        if soil_compaction_scenario == "no_compaction":
            _soil_compaction_scenario = ""
        else:
            _soil_compaction_scenario = "_soil-compaction"
        for irrigation_scenario in irrigation_scenarios:
            if os.path.exists(str(base_path / "output" / dir_name / irrigation_scenario)):
                for crop_rotation_scenario in crop_rotation_scenarios:
                    roger_file = (
                        base_path
                        / "output" 
                        / dir_name 
                        / f"{irrigation_scenario}{_soil_compaction_scenario}"
                        / f"SVATCROPNITRATE_{irrigation_scenario}_{crop_rotation_scenario}{_soil_compaction_scenario}.nc"
                    )
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
                        dir_csv_files = base_path / "output" / dir_name / irrigation_scenario / crop_rotation_scenario
                        if not os.path.exists(dir_csv_files):
                            os.makedirs(dir_csv_files)
                        for x, soil_type in enumerate(soil_types):
                            dir_csv_files = base_path / "output" / dir_name / irrigation_scenario / crop_rotation_scenario / soil_type
                            if not os.path.exists(dir_csv_files):
                                os.makedirs(dir_csv_files)
                            # write simulation to csv
                            df_simulation = pd.DataFrame(
                                index=date, columns=["N_fert", "N_uptake", "N_leach", "NO3_leach_conc", "NO3_soil_conc", "lu_id", "crop_type"]
                            )
                            df_simulation.loc[:, "N_fert"] = ds["Nfert"].isel(x=x, y=0).values * 0.01 # convert to mg/m2 to kg/ha
                            df_simulation.loc[:, "N_uptake"] = (ds["M_transp"].isel(x=x, y=0).values + ds["nh4_up"].isel(x=x, y=0).values) * 0.01 # convert to mg/m2 to kg/ha
                            df_simulation.loc[:, "N_leach"] = ds["M_q_ss"].isel(x=x, y=0).values * 0.01 # convert to mg/m2 to kg/ha
                            df_simulation.loc[:, "NO3_leach_conc"] = ds["C_q_ss"].isel(x=x, y=0).values * 4.43 # convert nitrate-nitrogen to nitrate
                            df_simulation.loc[:, "NO3_soil_conc"] = ds["C_s"].isel(x=x, y=0).values * 4.43 # convert nitrate-nitrogen to nitrate
                            df_simulation.loc[:, "lu_id"] = ds["lu_id"].isel(x=x, y=0).values
                            df_simulation.loc[:, "crop_type"] = [dict_crop_types[lu_id] for lu_id in ds["lu_id"].isel(x=x, y=0).values]
                            df_simulation.columns =[["[kg N/ha/day]", "[kg N/ha/day]", "[kg N/ha/day]", "[mg/l]", "[mg/l]", "", ""],
                                                    ["N_fert", "N_uptake", "N_leach", "NO3_leach_conc", "NO3_soil_conc", "lu_id", "crop_type"]]
                            df_simulation = df_simulation.iloc[1:, :] # remove initial values
                            df_simulation.to_csv(
                                dir_csv_files / "simulation.csv", sep=";"
                            )
    return


if __name__ == "__main__":
    main()
