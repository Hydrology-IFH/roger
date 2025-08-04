from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd

base_path = Path(__file__).parent

file = base_path / "input" / "BK50_lsv_locations.gpkg"
gdf = gpd.read_file(file)

param_names = ["lsv_location", "z_soil", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "kf", "soil_fertility", "clay", "z_gw"]
df_params = pd.DataFrame(columns=param_names, index=range(len(gdf)))
df_params.loc[:, "lsv_location"] = gdf["Name"]
df_params.loc[:, "z_soil"] = gdf["GRUND"] * 10  # cm to mm
df_params.loc[:, "dmpv"] = gdf["MPD_V"]
df_params.loc[:, "lmpv"] = gdf["MPL_V"]
df_params.loc[:, "theta_ac"] = gdf["LK_OB"] / 100  # % to -
df_params.loc[:, "theta_ufc"] = gdf["NFK"] / 100  # % to -
df_params.loc[:, "theta_pwp"] = gdf["PWP"] / 100  # % to -
df_params.loc[:, "ks"] = gdf["KS_OB"]
df_params.loc[:, "kf"] = gdf["KS_GEO"]
df_params.loc[:, "soil_fertility"] = gdf["BOD_NAT"]
df_params.loc[:, "z_gw"] = gdf["GWFA"] / 100

theta_pwp = df_params["theta_pwp"].values.astype(np.float64)
theta_fc = df_params["theta_pwp"].values.astype(np.float64) + df_params["theta_ufc"].values.astype(np.float64)
theta_sat = df_params["theta_pwp"].values.astype(np.float64) + df_params["theta_ufc"].values.astype(np.float64) + df_params["theta_ac"].values.astype(np.float64)

# calculate pore-size distribution index
lambda_bc = (
            np.log(theta_fc / theta_sat)
            - np.log(theta_pwp/ theta_sat)
        ) / (np.log(15850) - np.log(63))

# calculate bubbling pressure
ha = ((theta_pwp / theta_sat) ** (1.0 / lambda_bc) * (-15850))

# calculate soil water content at pF = 6
theta_6 = ((ha / (-(10**6))) ** lambda_bc * theta_sat)

# calculate clay content
clay = (0.71 * (theta_6 - 0.01) / 0.3)
clay = np.where(clay < 0.01, 0.01, clay)

df_params.loc[:, "clay"] = clay

df_params.columns = [
    ["", "[mm]", "[1/m2]", "[mm]", "[-]", "[-]", "[-]", "[mm/hour]", "[mm/hour]", "", "[-]", "[m]"],
    ["lsv_location", "z_soil", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "kf", "soil_fertility", "clay", "z_gw"],
]
df_params.to_csv(base_path / "parameters_lsv.csv", index=False, sep=";")
