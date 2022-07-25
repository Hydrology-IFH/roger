import os
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as onp
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import glob
import h5netcdf
import roger
import roger.tools.labels as labs

sns.set_context("talk", font_scale=1)

base_path = Path(__file__).parent
# directory of results
base_path_results = base_path / "results"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

meteo_stations = ["breitnau", "ihringen"]
# merge model output into single file
for meteo_station in meteo_stations:
    path = str(base_path / f"ONED_nosnow_noint_{meteo_station}.*.nc")
    diag_files = glob.glob(path)
    states_hm_file = base_path / "states_hm_nosnow_noint.nc"
    with h5netcdf.File(states_hm_file, 'a', decode_vlen_strings=False) as f:
        if meteo_station not in list(f.groups.keys()):
            f.create_group(meteo_station)
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title='RoGeR model results for realistic parameter set and input from DWD stations Breitnau and Ihringen',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment='',
            model_structure='1D model with free drainage',
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
                if not f.groups[meteo_station].dimensions:
                    f.groups[meteo_station].dimensions = dict_dim
                    v = f.groups[meteo_station].create_variable('x', ('x',), float)
                    v.attrs['long_name'] = 'Model run'
                    v.attrs['units'] = ''
                    v[:] = onp.arange(dict_dim["x"])
                    v = f.groups[meteo_station].create_variable('y', ('y',), float)
                    v.attrs['long_name'] = ''
                    v.attrs['units'] = ''
                    v[:] = onp.arange(dict_dim["y"])
                    v = f.groups[meteo_station].create_variable('Time', ('Time',), float)
                    var_obj = df.variables.get('Time')
                    v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                   units=var_obj.attrs["units"])
                    v[:] = time
                for key in list(df.variables.keys()):
                    var_obj = df.variables.get(key)
                    if key not in list(dict_dim.keys()) and var_obj.ndim == 3:
                        v = f.groups[meteo_station].create_variable(key, ('x', 'y', 'Time'), float)
                        vals = onp.array(var_obj)
                        v[:, :, :] = vals.swapaxes(0, 2)
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                       units=var_obj.attrs["units"])

with h5netcdf.File(states_hm_file, 'a', decode_vlen_strings=False) as f:
    for meteo_station in meteo_stations:
        # water for infiltration
        try:
            v = f.groups[meteo_station].create_variable('inf_in', ('x', 'y', 'Time'), float)
        except ValueError:
            v = f.groups[meteo_station].variables.get('inf_in')
        vals = onp.array(f.groups[meteo_station].variables.get('prec')) - onp.array(f.groups[meteo_station].variables.get('int_rain_top')) - onp.array(f.groups[meteo_station].variables.get('int_rain_ground')) - onp.array(f.groups[meteo_station].variables.get('int_snow_top')) - onp.array(f.groups[meteo_station].variables.get('int_snow_ground')) - onp.array(f.groups[meteo_station].variables.get('snow_ground')) + onp.array(f.groups[meteo_station].variables.get('q_snow'))
        v[:, :, :] = vals
        v.attrs.update(long_name='infiltration input',
                       units='mm/day')

        # initial soil water content
        try:
            v = f.groups[meteo_station].create_variable('theta_init', ('x', 'y'), float)
        except ValueError:
            v = f.groups[meteo_station].variables.get('theta_init')
        vals = onp.array(f.groups[meteo_station].variables.get('theta'))
        v[:, :] = vals[:, :, 0]
        v.attrs.update(long_name='initial soil water content',
                       units='-')
        try:
            v = f.groups[meteo_station].create_variable('S_s_init', ('x', 'y'), float)
        except ValueError:
            v = f.groups[meteo_station].variables.get('S_s_init')
        vals = onp.array(f.groups[meteo_station].variables.get('S_s'))
        v[:, :] = vals[:, :, 0]
        v.attrs.update(long_name='initial soil water content',
                       units='mm')
        # end soil water content
        try:
            v = f.groups[meteo_station].create_variable('theta_end', ('x', 'y'), float)
        except ValueError:
            v = f.groups[meteo_station].variables.get('theta_end')
        vals = onp.array(f.groups[meteo_station].variables.get('theta'))
        v[:, :] = vals[:, :, -1]
        v.attrs.update(long_name='end soil water content',
                       units='-')
        try:
            v = f.groups[meteo_station].create_variable('S_s_end', ('x', 'y'), float)
        except ValueError:
            v = f.groups[meteo_station].variables.get('S_s_end')
        vals = onp.array(f.groups[meteo_station].variables.get('S_s'))
        v[:, :] = vals[:, :, -1]
        v.attrs.update(long_name='end soil water content',
                       units='mm')


ds_sim = xr.open_dataset(states_hm_file, engine="h5netcdf", group="ihringen")
days_sim = ds_sim.Time.values + 1
grid0 = pd.DataFrame(index=range(len(days_sim[1:])))
for var_sim in ['inf_in', 'inf_mat', 'inf_mp', 'inf_sc', 'q_sub_mat', 'q_sub_mp']:
    grid0.loc[:, var_sim] = ds_sim[var_sim].isel(x=0, y=0).values[1:]
grid0.loc[:, 'perc'] = ds_sim['q_ss'].isel(x=0, y=0).values[1:]
file = base_path_results / "grid0.csv"
grid0.to_csv(file, header=True, index=True, sep=";")
