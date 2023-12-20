    
from pathlib import Path
import numpy as np
import pandas as pd

base_path = Path(__file__).parent
csv_file = base_path / "parameters.csv"
df_params = pd.read_csv(csv_file, sep=";", skiprows=1)

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
    ["", "", "", "[mm]", "[1/m2]", "[mm]", "[-]", "[-]", "[-]", "[mm/hour]", "[mm/hour]", "", "[-]"],
    ["CLUST_ID", "SHP_ID", "CLUST_flag", "z_soil", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "kf", "soil_fertility", "clay"],
]
df_params.to_csv(base_path / "parameters.csv", index=False, sep=";")