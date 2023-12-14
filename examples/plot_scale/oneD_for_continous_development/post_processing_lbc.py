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

base_path = Path(__file__).parent
# directory of results
base_path_output = base_path / "output"
if not os.path.exists(base_path_output):
    os.mkdir(base_path_output)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

meteo_stations = ["breitnau", "ihringen"]
# merge model output into single file
for meteo_station in meteo_stations:
    path = str(base_path / f"ONED_lbc_{meteo_station}.*.nc")
    diag_files = glob.glob(path)
    ONED_file = base_path / "ONED.nc"
    with h5netcdf.File(ONED_file, 'a', decode_vlen_strings=False) as f:
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
                    v = f.groups[meteo_station].create_variable('x', ('x',), float, compression="gzip", compression_opts=1)
                    v.attrs['long_name'] = 'Model run'
                    v.attrs['units'] = ''
                    v[:] = onp.arange(dict_dim["x"])
                    v = f.groups[meteo_station].create_variable('y', ('y',), float, compression="gzip", compression_opts=1)
                    v.attrs['long_name'] = ''
                    v.attrs['units'] = ''
                    v[:] = onp.arange(dict_dim["y"])
                    v = f.groups[meteo_station].create_variable('Time', ('Time',), float, compression="gzip", compression_opts=1)
                    var_obj = df.variables.get('Time')
                    v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                   units=var_obj.attrs["units"])
                    v[:] = time
                for key in list(df.variables.keys()):
                    var_obj = df.variables.get(key)
                    if key not in list(dict_dim.keys()) and var_obj.ndim == 3:
                        v = f.groups[meteo_station].create_variable(key, ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        v[:, :, :] = vals.swapaxes(0, 2)
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                       units=var_obj.attrs["units"])

with h5netcdf.File(ONED_file, 'a', decode_vlen_strings=False) as f:
    for meteo_station in meteo_stations:
        # water for infiltration
        try:
            v = f.groups[meteo_station].create_variable('inf_in', ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
        except ValueError:
            v = f.groups[meteo_station].variables.get('inf_in')
        vals = onp.array(f.groups[meteo_station].variables.get('prec')) - onp.array(f.groups[meteo_station].variables.get('int_rain_top')) - onp.array(f.groups[meteo_station].variables.get('int_rain_ground')) - onp.array(f.groups[meteo_station].variables.get('int_snow_top')) - onp.array(f.groups[meteo_station].variables.get('int_snow_ground')) + onp.array(f.groups[meteo_station].variables.get('q_snow'))
        v[:, :, :] = vals
        v.attrs.update(long_name='infiltration input',
                       units='mm/day')

        # initial soil water content
        try:
            v = f.groups[meteo_station].create_variable('theta_init', ('x', 'y'), float, compression="gzip", compression_opts=1)
        except ValueError:
            v = f.groups[meteo_station].variables.get('theta_init')
        vals = onp.array(f.groups[meteo_station].variables.get('theta'))
        v[:, :] = vals[:, :, 0]
        v.attrs.update(long_name='initial soil water content',
                       units='-')
        try:
            v = f.groups[meteo_station].create_variable('S_s_init', ('x', 'y'), float, compression="gzip", compression_opts=1)
        except ValueError:
            v = f.groups[meteo_station].variables.get('S_s_init')
        vals = onp.array(f.groups[meteo_station].variables.get('S_s'))
        v[:, :] = vals[:, :, 0]
        v.attrs.update(long_name='initial soil water content',
                       units='mm')
        # end soil water content
        try:
            v = f.groups[meteo_station].create_variable('theta_end', ('x', 'y'), float, compression="gzip", compression_opts=1)
        except ValueError:
            v = f.groups[meteo_station].variables.get('theta_end')
        vals = onp.array(f.groups[meteo_station].variables.get('theta'))
        v[:, :] = vals[:, :, -1]
        v.attrs.update(long_name='end soil water content',
                       units='-')
        try:
            v = f.groups[meteo_station].create_variable('S_s_end', ('x', 'y'), float, compression="gzip", compression_opts=1)
        except ValueError:
            v = f.groups[meteo_station].variables.get('S_s_end')
        vals = onp.array(f.groups[meteo_station].variables.get('S_s'))
        v[:, :] = vals[:, :, -1]
        v.attrs.update(long_name='end soil water content',
                       units='mm')


meteo_stations = ["breitnau", "ihringen"]
vars_sim = ['prec', 'int_prec', 'q_snow', 'inf_in', 'inf_mat', 'inf_mp', 'inf_sc',
            'q_hof', 'q_sof', 'q_sub', 'q_sub_mat', 'q_sub_mp', 'q_ss',
            'pet', 'aet', 'evap_sue', 'evap_soil', 'transp']
idx_percentiles = ['min', 'q25', 'median', 'mean', 'q75', 'max']
ll_df_sim_sum = []
ll_df_sim_sum_tot = []
for i, meteo_station in enumerate(meteo_stations):
    # load simulation
    ONED_file = base_path / "ONED.nc"
    ds_sim = xr.open_dataset(ONED_file, engine="h5netcdf", group=meteo_station)

    # assign date
    days_sim = ds_sim.Time.values + 1
    ds_sim = ds_sim.assign_coords(date=("Time", days_sim))

    # sums per grid
    ds_sim_sum = ds_sim.sum(dim="Time")
    nx = ds_sim_sum.dims['x']  # number of rows
    df = pd.DataFrame(index=range(nx))
    for var_sim in vars_sim:
        df.loc[:, var_sim] = ds_sim_sum[var_sim].values.flatten()

    df_percentiles = pd.DataFrame(index=idx_percentiles, columns=vars_sim)
    for var_sim in vars_sim:
        df_percentiles.loc["min", var_sim] = df.loc[:, var_sim].min()
        df_percentiles.loc["q25", var_sim] = df.loc[:, var_sim].quantile(0.25)
        df_percentiles.loc["median", var_sim] = df.loc[:, var_sim].median()
        df_percentiles.loc["mean", var_sim] = df.loc[:, var_sim].mean()
        df_percentiles.loc["q75", var_sim] = df.loc[:, var_sim].quantile(0.75)
        df_percentiles.loc["max", var_sim] = df.loc[:, var_sim].max()
    file = base_path_output / f"percentiles_{meteo_station}.csv"
    df_percentiles.to_csv(file, header=True, index=True, sep=";")

    file = base_path_output / f"summary_{meteo_station}.txt"
    df.to_csv(file, header=True, index=False, sep="\t")
    df.loc[:, 'meteo_station'] = meteo_station
    df.loc[:, 'idx'] = df.index

    ll_df_sim_sum.append(df)

    # total sums
    ds_sim_sum_tot = ds_sim.sum()
    df = pd.DataFrame(index=["sum"])
    for j, var_sim in enumerate(vars_sim):
        df.loc[:, var_sim] = ds_sim_sum_tot[var_sim].values
    df.loc[:, 'meteo_station'] = meteo_station

    ll_df_sim_sum_tot.append(df)

    # plot time series
    vars_sim_trace = ["S_s", "theta"]
    nx = ds_sim.dims['x']
    days_sim = ds_sim.Time.values + 1
    for j, var_sim in enumerate(vars_sim_trace):
        fig, ax = plt.subplots()
        for x in range(nx):
            vals = ds_sim[var_sim].isel(x=x, y=0).values
            ax.plot(days_sim[1:], vals[1:], color='black')
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel(labs._LABS[var_sim])
        fig.tight_layout()
        file = base_path_figs / f"trace_{var_sim}_{meteo_station}.png"
        fig.savefig(file, dpi=250)
        plt.close(fig)

# concatenate dataframes
df_sim_sum = pd.concat(ll_df_sim_sum, sort=False)
df_sim_sum_tot = pd.concat(ll_df_sim_sum_tot, sort=False)

# convert from wide to long
df_sim_sum = pd.melt(df_sim_sum, id_vars=['meteo_station', 'idx'])
df_sim_sum_tot = pd.melt(df_sim_sum_tot, id_vars=['meteo_station'])
for i, meteo_station in enumerate(meteo_stations):
    df_sim_sum_tot.loc[df_sim_sum_tot['meteo_station'] == meteo_station, 'idx'] = range(len(vars_sim))


# compare total sums
ax = sns.catplot(x="variable", y="value", hue="meteo_station",
                data=df_sim_sum_tot, height=7, aspect=2, palette="RdPu", kind="bar")
xticklabels = [labs._TICKLABS[var_sim] for var_sim in vars_sim]
ax.set_xticklabels(xticklabels)
ax.set(xlabel='', ylabel='[mm]')
ax._legend.set_title("Meteo station")
file = base_path_figs / "total_sums.png"
ax.savefig(file, dpi=250)

# compare sums per grid
ax = sns.catplot(x="variable", y="value", hue="meteo_station",
                data=df_sim_sum, kind="box", height=7, aspect=2, palette="RdPu", whis=[0, 100])
xticklabels = [labs._TICKLABS[var_sim] for var_sim in vars_sim]
ax.set_xticklabels(xticklabels)
ax.set(xlabel='', ylabel='[mm]')
ax._legend.set_title("Meteo station")
file = base_path_figs / "sums_per_grid_box.png"
ax.savefig(file, dpi=250)

#TODO: compare differences
# for i, meteo_station in enumerate(meteo_stations):
#     fig, ax = plt.subplots(3, 4, sharey=False, figsize=(16, 8))
#     data = df_sim_sum.loc[df_sim_sum['meteo_station'] == meteo_station, :]
#     for j, var_sim in enumerate(vars_sim):
#         data1 = data.loc[data['variable'] == var_sim, :]
#         ax.flatten()[j].bar(data1['idx'], data1['value'], color='black', edgecolor='black', width=1, align="edge")
#         ax.flatten()[j].set_xlabel('')
#         ax.flatten()[j].set_ylabel(labs._Y_LABS_CUM[var_sim])
#     fig.tight_layout()
#     file = base_path_figs / f"sums_per_grid_{meteo_station}.png"
#     fig.savefig(file, dpi=250)
plt.close('all')
