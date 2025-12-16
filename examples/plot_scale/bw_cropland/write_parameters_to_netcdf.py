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
   "z_gw": "m",
   "soil_fertility": "",
   "alpha_transp": "-",
   "alpha_q": "-",
   "c2_transp": "-",
   "c2_q_rz": "-",
   "c2_q_ss": "-",
   "km_denit": "kg N/ha/year",
   "km_nit": "kg N/ha/year",
   "kmin": "kg N/ha/year",
   "kfix": "kg N/ha/year",
   "kngl": "kg N/ha/year",
   "dmax_denit": "kg N/ha/year",
   "dmax_nit": "kg N/ha/year",
   "phi_soil_temp": "day of year",
   "clay": "-",
   "soil_fertility": "",
   "z_soil": "mm",
}

_VALS = {
    "alpha_transp": 1.5,
    "alpha_q": 0.5,
    "c2_transp": 0.6,
    "c2_q_rz": 1.5,
    "c2_q_ss": 1.5,
    "km_denit": [2.5, 18.7],
    "km_nit": [2.5, 12],
    "kmin": [10, 30],
    "kfix": 40,
    "kngl": 75,
    "dmax_denit": [10, 40],
    "dmax_nit": [10, 60],
    "phi_soil_temp": 91,
}

base_path = Path(__file__).parent

# load the parameters
file = base_path / "parameters.csv"
df_parameters = pd.read_csv(file, sep=";", skiprows=1, index_col=0)

# write parameters to netcdf
param_names = ["z_soil", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "kf", "soil_fertility", "clay", "z_gw"]
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


    c_clay = df_parameters.loc[:, "clay"].values.astype(float) / (0.4 - 0.02)
    NN = len(df_parameters.index)

    c_soil_fertility = np.zeros(NN)
    cond1 = (df_parameters.loc[:, "soil_fertility"].values.astype(float) > 1) & (df_parameters.loc[:, "soil_fertility"].values.astype(float) <= 2)
    cond2 = (df_parameters.loc[:, "soil_fertility"].values.astype(float) > 2) & (df_parameters.loc[:, "soil_fertility"].values.astype(float) <= 3)
    cond3 = (df_parameters.loc[:, "soil_fertility"].values.astype(float) > 3) & (df_parameters.loc[:, "soil_fertility"].values.astype(float) <= 4)
    c_soil_fertility[cond1] = 1/3
    c_soil_fertility[cond2] = 2/3
    c_soil_fertility[cond3] = 1

    # write parameters to csv
    df_parameters_sas_nitrate = pd.DataFrame(index=df_parameters.index)
    df_parameters_sas_nitrate.loc[:, "alpha_transp"] = _VALS["alpha_transp"]
    df_parameters_sas_nitrate.loc[:, "alpha_q"] = _VALS["alpha_q"]
    df_parameters_sas_nitrate.loc[:, "c2_transp"] = _VALS["c2_transp"]
    df_parameters_sas_nitrate.loc[:, "c2_q_rz"] = _VALS["c2_q_rz"]
    df_parameters_sas_nitrate.loc[:, "c2_q_ss"] = _VALS["c2_q_ss"]
    df_parameters_sas_nitrate.loc[:, "phi_soil_temp"] = _VALS["phi_soil_temp"]
    df_parameters_sas_nitrate.loc[:, "km_denit"] = _VALS["km_denit"][0] + (_VALS["km_denit"][1] - _VALS["km_denit"][0]) * (1 - c_clay) 
    df_parameters_sas_nitrate.loc[:, "dmax_denit"] = _VALS["dmax_denit"][0] + (_VALS["dmax_denit"][1] - _VALS["dmax_denit"][0]) * c_clay
    df_parameters_sas_nitrate.loc[:, "km_nit"] = _VALS["km_nit"][0] + (_VALS["km_nit"][1] - _VALS["km_nit"][0]) * (1 - c_clay) * c_soil_fertility
    df_parameters_sas_nitrate.loc[:, "dmax_nit"] = _VALS["dmax_nit"][0] + (_VALS["dmax_nit"][1] - _VALS["dmax_nit"][0]) * (1 - c_clay) * c_soil_fertility
    df_parameters_sas_nitrate.loc[:, "kmin"] = _VALS["kmin"][0] + (_VALS["kmin"][1] - _VALS["kmin"][0]) * c_soil_fertility
    df_parameters_sas_nitrate.loc[:, "kfix"] = _VALS["kfix"] * c_soil_fertility
    df_parameters_sas_nitrate.loc[:, "kngl"] = _VALS["kngl"] * (1 - c_clay)

    df_parameters_sas_nitrate.loc[:, "clay"] = df_parameters.loc[:, "clay"].values.astype(float)
    df_parameters_sas_nitrate.loc[:, "soil_fertility"] = df_parameters.loc[:, "soil_fertility"].values.astype(float)
    df_parameters_sas_nitrate.loc[:, "z_soil"] = df_parameters.loc[:, "z_soil"].values.astype(float)

    param_names = ["alpha_transp", "alpha_q", "c2_transp", "c2_q_rz", "c2_q_ss", "km_denit", "km_nit", "kmin", "kfix", "kngl", "dmax_denit", "dmax_nit", "phi_soil_temp", "clay", "soil_fertility", "z_soil"]
    df_parameters_sas_nitrate = df_parameters_sas_nitrate.loc[:, param_names]

    # write parameters to csv
    df_parameters_sas_nitrate.columns = [
        ["[-]", "[-]", "[-]", "[-]", "[-]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", "[day of year]", "[-]", "", "[mm]"], 
        param_names,
    ]
    df_parameters_sas_nitrate.to_csv(base_path / "nitrate" / "parameters_sas-nitrate.csv", index=False, sep=";")

    # write parameters to netcdf
    params_file = base_path / "nitrate" / "parameters_sas-nitrate.nc"
    with h5netcdf.File(params_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title="RoGeR SAS and nitrate parameters",
        institution="University of Freiburg, Chair of Hydrology",
        references="",
        comment="",
        )
        dict_dim = {"x": len(df_parameters_sas_nitrate.index), "y": 1, 'scalar': 1}
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
            vals = df_parameters_sas_nitrate.iloc[:, i].values.flatten().reshape(NN, 1)
            v[:, :] = vals
            v.attrs.update(long_name=key, units=_UNITS[key])

