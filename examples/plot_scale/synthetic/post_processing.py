import os
from pathlib import Path
import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
vars_sim = ['prec', 'int_prec', 'q_snow', 'inf_in', 'inf_mat', 'inf_mp', 'inf_sc',
            'q_hof', 'q_sof', 'q_sub', 'q_sub_mat', 'q_sub_mp', 'q_ss',
            'pet', 'aet', 'evap_int', 'evap_soil', 'transp']
idx_percentiles = ['min', 'q25', 'median', 'mean', 'q75', 'max']
ll_df_sim_sum = []
ll_df_sim_sum_tot = []
for i, meteo_station in enumerate(meteo_stations):
    # load simulation
    states_hm_file = base_path / "states_hm.nc"
    ds_sim = xr.open_dataset(states_hm_file, engine="h5netcdf", group=meteo_station)

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
    file = base_path_results / f"percentiles_{meteo_station}.csv"
    df_percentiles.to_csv(file, header=True, index=True, sep=";")

    file = base_path_results / f"summary_{meteo_station}.txt"
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
