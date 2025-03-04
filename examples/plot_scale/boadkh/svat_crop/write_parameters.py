from pathlib import Path
import pandas as pd
import numpy as np
import h5netcdf
import datetime

_UNITS = {
   "z_soil": "mm",
   "dmpv": "1/m2",
   "lmpv": "mm",
   "theta_ac": "-",
   "theta_ufc": "-",
   "theta_pwp": "-",
   "ks": "mm/hour",
   "kf": "mm/hour",
   "clay": "-",
   "soil_fertility": "",
}


base_path = Path(__file__).parent

# load the base matrix
file = base_path / "test_matrix_acker_base.csv"
df = pd.read_csv(file, sep=",")

# create a dataframe for the parameters
df_parameters = pd.DataFrame(columns=["CLUST_ID", "SHP_ID", "CLUST_flag", "z_soil", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "kf", "soil_fertility"])
df_parameters["CLUST_ID"] = df["CLUST_ID"]
df_parameters["SHP_ID"] = df["SHP_ID"]
df_parameters["CLUST_flag"] = df["CLUST_flag"]
df_parameters["z_soil"] = df["GRUND"] * 10
df_parameters["dmpv"] = df["MPD_V"]
df_parameters["lmpv"] = df["MPL_V"] * 10
df_parameters["theta_ac"] = df["LK_OB"] / 100
df_parameters["theta_ufc"] = df["NFK"] / 100
df_parameters["theta_pwp"] = df["PWP"] / 100
df_parameters["ks"] = df["KS_OB"]
df_parameters["kf"] = df["KS_GEO"]
df_parameters["soil_fertility"] = df["BOD_NAT"]

# calculate the clay content
theta_pwp = df_parameters["theta_pwp"].values.astype(np.float64)
theta_fc = df_parameters["theta_pwp"].values.astype(np.float64) + df_parameters["theta_ufc"].values.astype(np.float64)
theta_sat = df_parameters["theta_pwp"].values.astype(np.float64) + df_parameters["theta_ufc"].values.astype(np.float64) + df_parameters["theta_ac"].values.astype(np.float64)

# calculate pore-size distribution index
lambda_bc = (
            np.log(theta_fc / theta_sat)
            - np.log(theta_pwp/ theta_sat)
        ) / (np.log(15850) - np.log(63))

# calculate bubbling pressure
ha = ((theta_pwp / theta_sat) ** (1.0 / lambda_bc) * (-15850))

# calculate soil water content at pF = 6
theta_6 = ((ha / (-(10**6))) ** lambda_bc * theta_sat)
clay = (0.71 * (theta_6 - 0.01) / 0.3)
clay = np.where(clay < 0.01, 0.01, clay)
df_parameters.loc[:, "clay"] = clay

# write parameters to netcdf
param_names = ["z_soil", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "kf", "soil_fertility", "clay"]
params_file = base_path / "parameters.nc"
with h5netcdf.File(params_file, "w", decode_vlen_strings=False) as f:
    f.attrs.update(
    date_created=datetime.datetime.today().isoformat(),
    title="RoGeR parameters for cropland in Baden-Wuerttemberg, Germany",
    institution="University of Freiburg, Chair of Hydrology",
    references="",
    comment="",
    )
    ncols = len(df_parameters.index)
    dict_dim = {"x": ncols, "y": 1, 'scalar': 1}
    f.dimensions = dict_dim
    v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
    v.attrs["long_name"] = "model run"
    v.attrs["units"] = "-"
    v[:] = np.arange(dict_dim["x"])
    v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
    v.attrs["long_name"] = "y"
    v.attrs["units"] = "-"
    v[:] = np.arange(dict_dim["y"])
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
            key, ("x", "y"), np.float32, compression="gzip", compression_opts=1
        )
        vals = df_parameters.loc[:, key].values.flatten().reshape(ncols, 1)
        v[:, :] = vals
        v.attrs.update(long_name=key, units=_UNITS[key]) 

df_zsoil = df_parameters[["CLUST_ID", "SHP_ID", "CLUST_flag", "z_soil"]]   
df_clay = df_parameters[["CLUST_ID", "SHP_ID", "CLUST_flag", "clay"]]   
df_soil_fertility = df_parameters[["CLUST_ID", "SHP_ID", "CLUST_flag", "soil_fertility"]]

# write parameters to csv
file = base_path / "parameters.csv"
df_parameters.columns = [
    ["", "", "", "[mm]", "[1/m2]", "[mm]", "[-]", "[-]", "[-]", "[mm/hour]", "[mm/hour]", "", "[-]",],
    ["CLUST_ID", "SHP_ID", "CLUST_flag", "z_soil", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "kf", "soil_fertility", "clay"],
]
df_parameters.to_csv(file, index=False, sep=";")

file = base_path / "clay.csv"
df_clay.columns = [
    ["", "", "", "[-]",],
    ["CLUST_ID", "SHP_ID", "CLUST_flag", "clay"],
]
df_clay.to_csv(file, index=False, sep=";")

file = base_path / "soil_fertility.csv"
df_soil_fertility.columns = [
    ["", "", "", "",],
    ["CLUST_ID", "SHP_ID", "CLUST_flag", "soil_fertility"],
]
df_soil_fertility.to_csv(file, index=False, sep=";")

file = base_path / "z_soil.csv"
df_zsoil.columns = [
    ["", "", "", "[mm]",],
    ["CLUST_ID", "SHP_ID", "CLUST_flag", "z_soil"],
]
df_zsoil.to_csv(file, index=False, sep=";")