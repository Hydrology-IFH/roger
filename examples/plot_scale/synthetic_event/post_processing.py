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

sns.set_context("talk", font_scale=1.2)

base_path = Path(__file__).parent
# directory of results
base_path_results = base_path / "results"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

initial_conditions = ["dry", "normal", "wet", "75"]
rainfall_scenarios = ["rain", "block-rain", "rain-with-break", "heavyrain",
                      "heavyrain-normal", "heavyrain-gamma",
                      "heavyrain-gamma-reverse", "block-heavyrain"]
scenarios = []
for initial_condition in initial_conditions:
    for rainfall_scenario in rainfall_scenarios:
        scenarios.append(f"{rainfall_scenario}_{initial_condition}")
# merge model output into single file
for scenario in scenarios:
    path = str(base_path / f"ONEDEVENT_{scenario}.*.nc")
    diag_files = glob.glob(path)
    states_hm_file = base_path / "states_hm.nc"
    with h5netcdf.File(states_hm_file, 'a', decode_vlen_strings=False) as f:
        if scenario not in list(f.groups.keys()):
            f.create_group(scenario)
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title='RoGeR model results for realistic parameter set and rainfall scenarios as input',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment='',
            model_structure='1D model with free drainage for single event',
            roger_version=f'{roger.__version__}'
        )
        if scenario not in list(f.groups.keys()):
            f.create_group(scenario)
        # collect dimensions
        for dfs in diag_files:
            with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                # set dimensions with a dictionary
                if not dfs.split('/')[-1].split('.')[1] == 'constant':
                    dict_dim = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'Time': len(df.variables['Time'])}
                    time = onp.array(df.variables.get('Time'))
        for dfs in diag_files:
            with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                if not f.groups[scenario].dimensions:
                    f.groups[scenario].dimensions = dict_dim
                    v = f.groups[scenario].create_variable('x', ('x',), float)
                    v.attrs['long_name'] = 'Model run'
                    v.attrs['units'] = ''
                    v[:] = onp.arange(dict_dim["x"])
                    v = f.groups[scenario].create_variable('y', ('y',), float)
                    v.attrs['long_name'] = ''
                    v.attrs['units'] = ''
                    v[:] = onp.arange(dict_dim["y"])
                    v = f.groups[scenario].create_variable('Time', ('Time',), float)
                    var_obj = df.variables.get('Time')
                    v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                   units=var_obj.attrs["units"])
                    v[:] = time
                for key in list(df.variables.keys()):
                    var_obj = df.variables.get(key)
                    if key not in list(dict_dim.keys()) and var_obj.ndim == 3:
                        v = f.groups[scenario].create_variable(key, ('x', 'y', 'Time'), float)
                        vals = onp.array(var_obj)
                        v[:, :, :] = vals.swapaxes(0, 2)
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                       units=var_obj.attrs["units"])

with h5netcdf.File(states_hm_file, 'a', decode_vlen_strings=False) as f:
    for scenario in scenarios:
        # water for infiltration
        try:
            v = f.groups[scenario].create_variable('inf_in', ('x', 'y', 'Time'), float)
        except ValueError:
            v = f.groups[scenario].variables.get('inf_in')
        vals = onp.array(f.groups[scenario].variables.get('prec')) - onp.array(f.groups[scenario].variables.get('int_rain_top')) - onp.array(f.groups[scenario].variables.get('int_rain_ground'))
        v[:, :, :] = vals
        v.attrs.update(long_name='infiltration input',
                       units='mm/dt')
        # initial soil water content
        try:
            v = f.groups[scenario].create_variable('theta_init', ('x', 'y'), float)
        except ValueError:
            v = f.groups[scenario].variables.get('theta_init')
        vals = onp.array(f.groups[scenario].variables.get('theta'))
        v[:, :] = vals[:, :, 0]
        v.attrs.update(long_name='initial soil water content',
                       units='-')
        try:
            v = f.groups[scenario].create_variable('S_s_init', ('x', 'y'), float)
        except ValueError:
            v = f.groups[scenario].variables.get('S_s_init')
        vals = onp.array(f.groups[scenario].variables.get('S_s'))
        v[:, :] = vals[:, :, 0]
        v.attrs.update(long_name='initial soil water content',
                       units='mm')
        # end soil water content
        try:
            v = f.groups[scenario].create_variable('theta_end', ('x', 'y'), float)
        except ValueError:
            v = f.groups[scenario].variables.get('theta_end')
        vals = onp.array(f.groups[scenario].variables.get('theta'))
        v[:, :] = vals[:, :, -1]
        v.attrs.update(long_name='end soil water content',
                       units='-')
        try:
            v = f.groups[scenario].create_variable('S_s_end', ('x', 'y'), float)
        except ValueError:
            v = f.groups[scenario].variables.get('S_s_end')
        vals = onp.array(f.groups[scenario].variables.get('S_s'))
        v[:, :] = vals[:, :, -1]
        v.attrs.update(long_name='end soil water content',
                       units='mm')

for initial_condition in initial_conditions:
    scenarios1 = [scenario for scenario in scenarios if scenario.split('_')[-1] == initial_condition]
    vars_sim = ['prec', 'int_prec', 'inf_in', 'inf_mat', 'inf_mp', 'inf_sc', 'q_hof',
                'q_sof', 'q_sub', 'q_sub_mat', 'q_sub_mp', 'q_ss']
    vars_S = ['S_s_init', 'S_s_end']
    idx_percentiles = ['min', 'q25', 'median', 'mean', 'q75', 'max']
    ll_df_sim_sum = []
    ll_df_sim_sum_tot = []
    ll_df_sim_S = []
    ll_df_sim_S_tot = []
    ll_df_sim_max = []
    ll_df_sim_max_tot = []
    for i, scenario in enumerate(scenarios1):
        # load simulation
        states_hm_file = base_path / "states_hm.nc"
        ds_sim = xr.open_dataset(states_hm_file, engine="h5netcdf", group=scenario)

        # assign date
        days_sim = ds_sim.Time.values + 1
        ds_sim = ds_sim.assign_coords(date=("Time", days_sim))

        # maximums per grid
        ds_sim_max = ds_sim.max(dim="Time")
        nx = ds_sim_max.dims['x']  # number of rows
        df_max = pd.DataFrame(index=range(nx))
        for var_sim in vars_sim:
            df_max.loc[:, var_sim] = ds_sim_max[var_sim].values.flatten()

        df_percentiles_max = pd.DataFrame(index=idx_percentiles, columns=vars_sim)
        for var_sim in vars_sim:
            df_percentiles_max.loc["min", var_sim] = df_max.loc[:, var_sim].min()
            df_percentiles_max.loc["q25", var_sim] = df_max.loc[:, var_sim].quantile(0.25)
            df_percentiles_max.loc["median", var_sim] = df_max.loc[:, var_sim].median()
            df_percentiles_max.loc["mean", var_sim] = df_max.loc[:, var_sim].mean()
            df_percentiles_max.loc["q75", var_sim] = df_max.loc[:, var_sim].quantile(0.75)
            df_percentiles_max.loc["max", var_sim] = df_max.loc[:, var_sim].max()
        file = base_path_results / f"percentiles_max_{scenario}.csv"
        df_percentiles_max.to_csv(file, header=True, index=True, sep=";")

        file = base_path_results / f"maximum_{scenario}.txt"
        df_max.to_csv(file, header=True, index=False, sep="\t")
        df_max.loc[:, 'scenario'] = scenario
        df_max.loc[:, 'idx'] = df_max.index
        ll_df_sim_max.append(df_max)

        # sums per grid
        ds_sim_sum = ds_sim.sum(dim="Time")
        nx = ds_sim_sum.dims['x']  # number of rows
        df_sum = pd.DataFrame(index=range(nx))
        for var_sim in vars_sim:
            df_sum.loc[:, var_sim] = ds_sim_sum[var_sim].values.flatten()

        # states per grid
        df_S = pd.DataFrame(index=range(nx))
        for var_sim in vars_S:
            df_S.loc[:, var_sim] = ds_sim[var_sim].values.flatten()

        df_percentiles = pd.DataFrame(index=idx_percentiles, columns=vars_sim)
        for var_sim in vars_sim:
            df_percentiles.loc["min", var_sim] = df_sum.loc[:, var_sim].min()
            df_percentiles.loc["q25", var_sim] = df_sum.loc[:, var_sim].quantile(0.25)
            df_percentiles.loc["median", var_sim] = df_sum.loc[:, var_sim].median()
            df_percentiles.loc["mean", var_sim] = df_sum.loc[:, var_sim].mean()
            df_percentiles.loc["q75", var_sim] = df_sum.loc[:, var_sim].quantile(0.75)
            df_percentiles.loc["max", var_sim] = df_sum.loc[:, var_sim].max()
        for var_sim in vars_S:
            df_percentiles.loc["min", var_sim] = df_S.loc[:, var_sim].min()
            df_percentiles.loc["q25", var_sim] = df_S.loc[:, var_sim].quantile(0.25)
            df_percentiles.loc["median", var_sim] = df_S.loc[:, var_sim].median()
            df_percentiles.loc["mean", var_sim] = df_S.loc[:, var_sim].mean()
            df_percentiles.loc["q75", var_sim] = df_S.loc[:, var_sim].quantile(0.75)
            df_percentiles.loc["max", var_sim] = df_S.loc[:, var_sim].max()
        file = base_path_results / f"percentiles_{scenario}.csv"
        df_percentiles.to_csv(file, header=True, index=True, sep=";")

        file = base_path_results / f"summary_{scenario}.txt"
        df_sum.to_csv(file, header=True, index=False, sep="\t")
        df_sum.loc[:, 'scenario'] = scenario
        df_sum.loc[:, 'idx'] = df_sum.index
        ll_df_sim_sum.append(df_sum)

        file = base_path_results / f"summary_S_{scenario}.txt"
        df_S.to_csv(file, header=True, index=False, sep="\t")
        df_S.loc[:, 'scenario'] = scenario
        df_S.loc[:, 'idx'] = df_S.index
        ll_df_sim_S.append(df_S)

        # total sums
        ds_sim_sum_tot = ds_sim.sum()
        df = pd.DataFrame(index=["sum"])
        for j, var_sim in enumerate(vars_sim):
            df.loc[:, var_sim] = ds_sim_sum_tot[var_sim].values
        df.loc[:, 'scenario'] = scenario

        ll_df_sim_sum_tot.append(df)

        # plot time series
        vars_sim_trace = ["S_s", "theta", "z_sat"]
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
            file = base_path_figs / f"trace_{var_sim}_{scenario}.png"
            fig.savefig(file, dpi=250)
            plt.close(fig)


    # concatenate dataframes
    df_sim_max = pd.concat(ll_df_sim_max, sort=False)
    df_sim_S = pd.concat(ll_df_sim_S, sort=False)
    df_sim_sum = pd.concat(ll_df_sim_sum, sort=False)
    df_sim_sum_tot = pd.concat(ll_df_sim_sum_tot, sort=False)

    # convert from wide to long
    df_sim_max = pd.melt(df_sim_max, id_vars=['scenario', 'idx'])
    df_sim_S = pd.melt(df_sim_S, id_vars=['scenario', 'idx'])

    df_sim_sum = pd.melt(df_sim_sum, id_vars=['scenario', 'idx'])
    df_sim_sum_tot = pd.melt(df_sim_sum_tot, id_vars=['scenario'])
    for i, scenario in enumerate(scenarios1):
        df_sim_sum_tot.loc[df_sim_sum_tot['scenario'] == scenario, 'idx'] = range(len(vars_sim))


    # # compare total sums
    # ax = sns.catplot(x="variable", y="value", hue="scenario",
    #                  data=df_sim_sum_tot, height=7, aspect=2, palette="RdPu", kind="bar")
    # xticklabels = [labs._TICKLABS[var_sim] for var_sim in vars_sim]
    # ax.set_xticklabels(xticklabels)
    # ax.set(xlabel='', ylabel='[mm]')
    # ax._legend.set_title("Rainfall scenario")
    # file = base_path_figs / "total_sums.png"
    # ax.savefig(file, dpi=250)

    # compare sums per grid
    ax = sns.catplot(x="variable", y="value", hue="scenario",
                      data=df_sim_sum, kind="box", height=7, aspect=2, palette="RdPu", whis=[0, 100])
    xticklabels = [labs._TICKLABS[var_sim] for var_sim in vars_sim]
    ax.set_xticklabels(xticklabels)
    ax.set(xlabel='', ylabel='[mm]')
    ax._legend.set_title("Rainfall scenario")
    file = base_path_figs / f"sums_per_grid_box_{initial_condition}.png"
    ax.savefig(file, dpi=250)

    # #TODO: compare differences
    # for i, scenario in enumerate(scenarios):
    #     fig, ax = plt.subplots(3, 4, sharey=False, figsize=(16, 8))
    #     data = df_sim_sum.loc[df_sim_sum['scenario'] == scenario, :]
    #     for j, var_sim in enumerate(vars_sim):
    #         data1 = data.loc[data['variable'] == var_sim, :]
    #         ax.flatten()[j].bar(data1['idx'], data1['value'], color='black', edgecolor='black', width=1, align="edge")
    #         ax.flatten()[j].set_xlabel('')
    #         ax.flatten()[j].set_ylabel(labs._Y_LABS_CUM[var_sim])

    #     ax[2,1].remove()
    #     ax[2,2].remove()
    #     ax[2,3].remove()
    #     fig.tight_layout()
    #     file = base_path_figs / f"sums_per_grid_{scenario}.png"
    #     fig.savefig(file, dpi=250)

    # # plot rainfall scenarios
    # fig, ax = plt.subplots(2, 4, sharey=True, figsize=(16, 8))
    # for j, scenario in enumerate(scenarios):
    # 	prec_file = base_path / "input" / scenario / "PREC.txt"
    # 	df_prec = pd.read_csv(prec_file, sep=r"\s+", header=0)
    # 	ax.flatten()[j].bar(df_prec['hh'], df_prec['PREC'], color='black', edgecolor='black', width=10/60, align="edge")
    # 	ax.flatten()[j].set_xlabel('Time [hours]')
    # 	ax.flatten()[j].set_ylabel('')
    # 	ax.flatten()[j].set_title(scenario)

    # ax[0, 0].set_ylabel('[mm/10 minutes]')
    # ax[1, 0].set_ylabel('[mm/10 minutes]')
    # fig.tight_layout()
    # file = base_path_figs / "scenarios.png"
    # fig.savefig(file, dpi=250)
plt.close('all')
