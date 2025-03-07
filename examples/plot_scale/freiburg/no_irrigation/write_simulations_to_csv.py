from pathlib import Path
import os
import xarray as xr
import numpy as onp
from cftime import num2date
import pandas as pd
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent.parent
    dir_name = os.path.basename(str(Path(__file__).parent))

    # identifiers of simulations
    irrigation_scenarios = ["35-ufc",
                            "45-ufc",
                            "50-ufc",
                            "80-ufc",
                            "crop-specific",
                            ]
    crop_rotation_scenarios = ["winter-wheat_clover",
                               "winter-wheat_silage-corn",
                               "summer-wheat_winter-wheat",
                               "summer-wheat_clover_winter-wheat",
                               "winter-wheat_clover_silage-corn",
                               "winter-wheat_sugar-beet_silage-corn",
                               "summer-wheat_winter-wheat_silage-corn",
                               "summer-wheat_winter-wheat_winter-rape",
                               "winter-wheat_winter-rape",
                               "winter-wheat_soybean_winter-rape",
                               "sugar-beet_winter-wheat_winter-barley", 
                               "grain-corn_winter-wheat_winter-rape", 
                               "grain-corn_winter-wheat_winter-barley",
                               "grain-corn_winter-wheat_clover",
                               "winter-wheat_silage-corn_yellow-mustard",
                               "summer-wheat_winter-wheat_yellow-mustard",
                               "winter-wheat_sugar-beet_silage-corn_yellow-mustard",
                               "summer-wheat_winter-wheat_silage-corn_yellow-mustard",
                               "summer-wheat_winter-wheat_winter-rape_yellow-mustard",
                               "sugar-beet_winter-wheat_winter-barley_yellow-mustard", 
                               "grain-corn_winter-wheat_winter-rape_yellow-mustard", 
                               "grain-corn_winter-wheat_winter-barley_yellow-mustard",
                               "miscanthus",
                               "bare-grass"]
    soil_types = ["sandy_soil", "silty_soil", "clayey_soil"]
    for irrigation_scenario in irrigation_scenarios:
        if os.path.exists(str(base_path / "output" / dir_name / irrigation_scenario)):
            for crop_rotation_scenario in crop_rotation_scenarios:
                roger_file = (
                    base_path
                    / "output" 
                    / dir_name 
                    / irrigation_scenario
                    / f"SVATCROP_{crop_rotation_scenario}.nc"
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
                            index=date, columns=["precip", "canopy_cover", "irrigation_demand", "theta_rz", "transp", "perc"]
                        )
                        df_simulation.loc[:, "precip"] = ds["prec"].isel(x=x, y=0).values
                        df_simulation.loc[:, "canopy_cover"] = ds["ground_cover"].isel(x=x, y=0).values
                        df_simulation.loc[:, "irrigation_demand"] = ds["irr_demand"].isel(x=x, y=0).values
                        df_simulation.loc[:, "theta_rz"] = ds["theta_rz"].isel(x=x, y=0).values
                        df_simulation.loc[:, "transp"] = ds["transp"].isel(x=x, y=0).values
                        df_simulation.loc[:, "perc"] = ds["q_ss"].isel(x=x, y=0).values
                        df_simulation.columns =[["[mm/day]", "[-]", "[mm/day]", "[-]", "[mm/day]", "[mm/day]"],
                                                ["precip", "canopy_cover", "irrigation_demand", "theta_rz", "transp", "perc"]]
                        df_simulation.to_csv(
                            dir_csv_files / "simulation.csv", sep=";"
                        )
    return


if __name__ == "__main__":
    main()
