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

# load the parameters
file = base_path / "parameters.csv"
df_parameters = pd.read_csv(file, sep=";", skiprows=1, index_col=0)

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
