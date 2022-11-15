import os
import glob
import h5netcdf
import datetime
from pathlib import Path
import xarray as xr
from cftime import num2date
import numpy as onp
import roger
import matplotlib as mpl
import seaborn as sns
mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
sns.set_style("ticks")

base_path = Path(__file__).parent
# directory of results
base_path_results = base_path / "results"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# merge model output into single file
path = str(base_path / "SVAT.*.nc")
diag_files = glob.glob(path)
states_hm_file = base_path / "states_hm.nc"
if not os.path.exists(states_hm_file):
    with h5netcdf.File(states_hm_file, 'w', decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title='RoGeR simulations (Tutorial)',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment='First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).',
            model_structure='SVAT model with free drainage',
            roger_version=f'{roger.__version__}'
        )
        # collect dimensions
        for dfs in diag_files:
            with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                # set dimensions with a dictionary
                if not dfs.split('/')[-1].split('.')[1] == 'constant':
                    dict_dim = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'Time': len(df.variables['Time'])}
                    time = onp.array(df.variables.get('Time'))
        for dfs in diag_files:
            with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                if not f.dimensions:
                    f.dimensions = dict_dim
                    v = f.create_variable('x', ('x',), float, compression="gzip", compression_opts=1)
                    v.attrs['long_name'] = 'x'
                    v.attrs['units'] = 'm'
                    v[:] = onp.arange(dict_dim["x"]) * 5
                    v = f.create_variable('y', ('y',), float, compression="gzip", compression_opts=1)
                    v.attrs['long_name'] = 'y'
                    v.attrs['units'] = 'm'
                    v[:] = onp.arange(dict_dim["y"]) * 5
                    v = f.create_variable('Time', ('Time',), float, compression="gzip", compression_opts=1)
                    var_obj = df.variables.get('Time')
                    v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                   units=var_obj.attrs["units"])
                    v[:] = time
                for key in list(df.variables.keys()):
                    var_obj = df.variables.get(key)
                    if key not in list(f.dimensions.keys()) and var_obj.ndim == 3:
                        v = f.create_variable(key, ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        v[:, :, :] = vals.swapaxes(0, 2)
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                       units=var_obj.attrs["units"])

# load simulation
states_hm_file = base_path / "states_hm.nc"
ds = xr.open_dataset(states_hm_file, engine="h5netcdf")
# assign date
days = (ds['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
date = num2date(days, units=f"days since {ds['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
ds = ds.assign_coords(Time=("Time", date))

# plot spatially distributed soil moisture at different dates
fig, axes = plt.subplots(1, 2, figsize=(4, 2))
axes[0].imshow(ds['theta'].isel(Time=120).values.T, origin="lower", cmap='Blues', vmin=0.2, vmax=0.4)
axes[0].set_xticks(onp.arange(-.5, 11, 5))
axes[0].set_yticks(onp.arange(-.5, 23, 5))
axes[0].set_xticklabels(onp.arange(0, 12, 5) * 5)
axes[0].set_yticklabels(onp.arange(0, 24, 5) * 5)
axes[0].set_xlabel('[m]')
axes[0].set_ylabel('[m]')
axes[0].set_title(str(ds['Time'].values[120]).split('T')[0])
axes[1].imshow(ds['theta'].isel(Time=300).values.T, origin="lower", cmap='Blues', vmin=0.2, vmax=0.4)
axes[1].set_xticks(onp.arange(-.5, 11, 5))
axes[1].set_yticks(onp.arange(-.5, 23, 5))
axes[1].set_xticklabels(onp.arange(0, 12, 5) * 5)
axes[1].set_yticklabels(onp.arange(0, 24, 5) * 5)
axes[1].set_xlabel('[m]')
axes[1].set_title(str(ds['Time'].values[300]).split('T')[0])
fig.tight_layout()
file = base_path_figs / "theta.png"
fig.savefig(file, dpi=250)


# plot fluxes of a single grid cell


# plot spatially averaged water balance
