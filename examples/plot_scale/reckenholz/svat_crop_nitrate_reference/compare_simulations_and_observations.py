import os
from pathlib import Path
from cftime import num2date
import xarray as xr
import pandas as pd
import numpy as onp
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt  # noqa: E402
import roger.tools.evaluation as eval_utils
mpl.use("agg")

# paper style
# mpl.rcParams["font.size"] = 8
# mpl.rcParams["axes.titlesize"] = 8
# mpl.rcParams["axes.labelsize"] = 9
# mpl.rcParams["xtick.labelsize"] = 8
# mpl.rcParams["ytick.labelsize"] = 8
# mpl.rcParams["legend.fontsize"] = 8
# mpl.rcParams["legend.title_fontsize"] = 9
# sns.set_style("ticks")
# sns.plotting_context(
#     "paper",
#     font_scale=1,
#     rc={
#         "font.size": 8.0,
#         "axes.labelsize": 9.0,
#         "axes.titlesize": 8.0,
#         "xtick.labelsize": 8.0,
#         "ytick.labelsize": 8.0,
#         "legend.fontsize": 8.0,
#         "legend.title_fontsize": 9.0,
#     },
# )

# presentation style
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10
mpl.rcParams["legend.title_fontsize"] = 11
sns.set_style("ticks")
sns.plotting_context(
    "paper",
    font_scale=1,
    rc={
        "font.size": 10.0,
        "axes.labelsize": 11.0,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        "legend.title_fontsize": 11.0,
    },
)

base_path = Path(__file__).parent

colors = ['#fee8c8', '#fc8d59', '#b30000']
lys_experiments = ["lys8", "lys3", "lys2"]
tm_structures = ['complete-mixing', 'advection-dispersion-power',
                    'time-variant_advection-dispersion-power']

lys_experiments = ["lys3"]
tm_structures = ['advection-dispersion-power']
for lys_experiment in lys_experiments:
    for tm_structure in tm_structures:
        # load simulations
        sim_file = base_path.parent / "output" / "svat_crop_reference" / f"SVATCROP_{lys_experiment}.nc"
        ds_water_sim = xr.open_dataset(sim_file, engine="h5netcdf")

        sim_file = base_path.parent / "output" / "svat_crop_nitrate_reference" / f"SVATCROPNITRATE_{tm_structure}_{lys_experiment}.nc"
        ds_nitrate_sim = xr.open_dataset(sim_file, engine="h5netcdf")

        # load observations (measured data)
        path_obs = Path("/Users/robinschwemmle/Desktop/PhD/data/plot/reckenholz/reckenholz_lysimeter.nc")
        ds_obs = xr.open_dataset(path_obs, engine="h5netcdf", group=lys_experiment)

        # plot observed and simulated time series
        base_path_figs = base_path.parent / "figures" / "svat_crop_nitrate_reference"
        if not os.path.exists(base_path_figs):
            os.mkdir(base_path_figs)

        time_origin = ds_water_sim['Time'].attrs['time_origin']
        days_sim = (ds_water_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date_water_sim = num2date(days_sim, units=f"days since {ds_water_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_water_sim = ds_water_sim.assign_coords(date=("Time", date_water_sim))

        time_origin = ds_nitrate_sim['Time'].attrs['time_origin']
        days_sim = (ds_nitrate_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date_nitrate_sim = num2date(days_sim, units=f"days since {ds_nitrate_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_nitrate_sim = ds_nitrate_sim.assign_coords(date=("Time", date_nitrate_sim))

        days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

        # vars_obs = ['NO3_PERC']
        # vars_sim = ['C_q_ss']
        # for var_obs, var_sim in zip(vars_obs, vars_sim):
        #     obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
        #     df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
        #     df_obs.loc[:, 'obs'] = obs_vals
        #     for x in range(0, 3):
        #         sim_vals = ds_nitrate_sim[var_sim].isel(x=x, y=0).values
        #         # join observations on simulations
        #         df_eval = eval_utils.join_obs_on_sim(date_nitrate_sim, sim_vals, df_obs)
        #         # plot observed and simulated time series
        #         fig = eval_utils.plot_obs_sim(df_eval, y_lab="[mg/l]", x_lab='Time [year]')
        #         file_str = '%s_%s_%s_%s.pdf' % (var_sim, lys_experiment, tm_structure, x)
        #         path_fig = base_path_figs / file_str
        #         fig.savefig(path_fig, dpi=300)

        # vars_obs = ['NO3_PERC']
        # vars_sim = ['M_q_ss']
        # for var_obs, var_sim in zip(vars_obs, vars_sim):
        #     obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
        #     df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
        #     df_obs.loc[:, 'obs'] = obs_vals
        #     for x in range(0, 3):
        #         sim_vals = ds_nitrate_sim[var_sim].isel(x=x, y=0).values
        #         # join observations on simulations
        #         df_eval = eval_utils.join_obs_on_sim(date_nitrate_sim, sim_vals, df_obs)
        #         df_eval = df_eval.dropna()
        #         # plot cumulated observed and simulated time series
        #         fig = eval_utils.plot_obs_sim_cum(df_eval, y_lab="[mg]", x_lab='Time [year]')
        #         file_str = '%s_cum_%s_%s_%s.pdf' % (var_sim, lys_experiment, tm_structure, x)
        #         path_fig = base_path_figs / file_str
        #         fig.savefig(path_fig, dpi=300)
        #         fig = eval_utils.plot_obs_sim_cum_year_facet(df_eval, y_lab="[mg]", x_lab='Time\n[day-month-hydyear]')
        #         file_str = '%s_cum_year_facet_%s_%s_%s.pdf' % (var_sim, lys_experiment, tm_structure, x)
        #         path_fig = base_path_figs / file_str
        #         fig.savefig(path_fig, dpi=300)


        fig, axes = plt.subplots(2, 1, figsize=(6, 4))
        obs_vals = ds_obs['PERC'].isel(x=0, y=0).values
        sim_vals = ds_water_sim['q_ss'].isel(x=0, y=0).values
        axes[0].plot(date_obs, obs_vals, color="blue")
        axes[0].plot(date_water_sim, sim_vals, color="red")
        axes[0].set_ylabel("PERC \n [mm/day]")
        axes[0].set_xlim(date_obs[0], date_obs[-1])
        axes[0].set_ylim(0, )

        obs_vals1 = ds_obs['NO3_PERC'].isel(x=0, y=0).values
        axes[1].scatter(date_obs, obs_vals1, color="blue", s=1)
        for x in range(0, 3):
            sim_vals1 = ds_nitrate_sim['C_q_ss'].isel(x=x, y=0).values
            sim_vals1 = onp.where(sim_vals < 0.01, onp.nan, sim_vals1)
            axes[1].plot(date_nitrate_sim, sim_vals1, color=colors[x])
        axes[1].set_ylim(0, )
        axes[1].set_ylabel("$NO_3$-N in PERC \n[mg/l]")
        axes[1].set_xlabel("Time [year]")
        axes[1].set_xlim(date_obs[0], date_obs[-1])
        fig.tight_layout()
        file = base_path_figs / f'PERC_NO3_conc_{lys_experiment}_{tm_structure}.png'
        fig.savefig(file, dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(6, 6))
        sim_vals = ds_nitrate_sim['M_in'].isel(x=1, y=0).values
        axes[0].plot(date_nitrate_sim, sim_vals, color="red")
        axes[0].set_ylabel("N-Fertilisation \n [mg]")
        axes[0].set_xlim(date_obs[0], date_obs[-1])
        axes[0].set_ylim(0, )

        obs_vals = ds_obs['PERC'].isel(x=0, y=0).values
        sim_vals = ds_water_sim['q_ss'].isel(x=0, y=0).values
        axes[1].plot(date_obs, obs_vals, color="blue")
        axes[1].plot(date_water_sim, sim_vals, color="red")
        axes[1].set_ylabel("PERC \n [mm/day]")
        axes[1].set_xlim(date_obs[0], date_obs[-1])
        axes[1].set_ylim(0, )

        obs_vals1 = ds_obs['NO3_PERC'].isel(x=0, y=0).values
        axes[2].scatter(date_obs, obs_vals1, color="blue", s=1)
        for x in range(0, 3):
            sim_vals1 = ds_nitrate_sim['C_q_ss'].isel(x=x, y=0).values
            sim_vals1 = onp.where(sim_vals < 0.01, onp.nan, sim_vals1)
            axes[2].plot(date_nitrate_sim, sim_vals1, color=colors[x])
        axes[2].set_ylim(0, )
        axes[2].set_ylabel("$NO_3$-N in PERC \n[mg/l]")
        axes[2].set_xlabel("Time [year]")
        axes[2].set_xlim(date_obs[0], date_obs[-1])
        fig.tight_layout()
        file = base_path_figs / f'NFERT_PERC_NO3_conc_{lys_experiment}_{tm_structure}.png'
        fig.savefig(file, dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(6, 6))
        sim_vals = ds_nitrate_sim['tt50_transp'].isel(x=1, y=0).values.astype(onp.float32) / 8.64e+13
        axes[0].plot(date_nitrate_sim, sim_vals, color="red")
        axes[0].set_ylabel("$TT_{50}$\n [days]")
        axes[0].set_xlim(date_nitrate_sim[0], date_nitrate_sim[-1])
        axes[0].set_ylim(0, 1000)

        sim_vals = ds_nitrate_sim['rt50_s'].isel(x=1, y=0).values.astype(onp.float32) / 8.64e+13
        axes[1].plot(date_nitrate_sim, sim_vals, color="red")
        axes[1].set_ylabel("$RT_{50}$\n [days]")
        axes[1].set_xlim(date_nitrate_sim[0], date_nitrate_sim[-1])
        axes[1].set_ylim(0, )

        sim_vals = ds_nitrate_sim['tt50_q_ss'].isel(x=1, y=0).values.astype(onp.float32) / 8.64e+13
        axes[2].plot(date_nitrate_sim, sim_vals, color="red")
        axes[2].set_ylabel("$TT_{50}$\n [days]")
        axes[2].set_xlim(date_nitrate_sim[0], date_nitrate_sim[-1])
        axes[2].set_ylim(0, 1000)
        axes[2].set_xlabel("Time [year]")
        axes[2].set_xlim(date_obs[0], date_obs[-1])
        fig.tight_layout()
        file = base_path_figs / f'tt_rt_{lys_experiment}_{tm_structure}.png'
        fig.savefig(file, dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(6, 6))
        sim_vals = ds_nitrate_sim['temp_soil'].isel(x=1, y=0).values
        axes[0].plot(date_nitrate_sim, sim_vals, color="red")
        axes[0].set_ylabel("Soil temperature \n [degC]")
        axes[0].set_xlim(date_obs[0], date_obs[-1])

        ssat1 = ds_nitrate_sim['S_sat_rz'].isel(x=1, y=0).values * 0.7
        ssat2 = ds_nitrate_sim['S_sat_rz'].isel(x=1, y=0).values
        axes[1].plot(date_nitrate_sim, ssat2, color="blue") 
        axes[1].plot(date_nitrate_sim, ssat1, color="blue")
        sim_vals = onp.sum(ds_nitrate_sim['sa_rz'].isel(x=1, y=0).values, axis=-1)
        axes[1].plot(date_nitrate_sim, sim_vals, color="red", ls="--")
        sim_vals = ds_water_sim['S_rz'].isel(x=0, y=0).values
        axes[1].plot(date_water_sim, sim_vals, color="red")
        axes[1].set_ylabel("Soil water content \n [mm]")
        axes[1].set_xlim(date_obs[0], date_obs[-1])
        axes[1].set_ylim(0, )

        sim_vals = ds_nitrate_sim['denit_s'].isel(x=1, y=0).values
        axes[2].plot(date_nitrate_sim, sim_vals1, color=colors[x])
        axes[2].set_ylim(0, )
        axes[2].set_ylabel("denitrification \n[mg/day]")
        axes[2].set_xlabel("Time [year]")
        axes[2].set_xlim(date_obs[0], date_obs[-1])
        fig.tight_layout()
        file = base_path_figs / f'soiltemp_swc_denit_{lys_experiment}_{tm_structure}.png'
        fig.savefig(file, dpi=300)
        plt.close(fig)


        fig, axes = plt.subplots(2, 1, figsize=(6, 4))
        sim_vals = ds_nitrate_sim['Nfert'].isel(x=1, y=0).values
        axes[0].plot(date_nitrate_sim, sim_vals, color="red")
        axes[0].set_ylabel("Fertilisation\n [mg]")
        axes[0].set_xlim(date_obs[0], date_obs[-1])
        axes[0].set_ylim(0, )


        sim_vals1 = onp.sum(ds_nitrate_sim['msa_rz'].isel(x=1, y=0).values, axis=-1)
        axes[1].plot(date_nitrate_sim, sim_vals1, color=colors[x])
        axes[1].set_ylim(0, )
        axes[1].set_ylabel("$NO_3$-N in RZ \n[mg]")
        axes[1].set_xlabel("Time [year]")
        axes[1].set_xlim(date_obs[0], date_obs[-1])
        fig.tight_layout()
        file = base_path_figs / f'Fert_Nsoil_{lys_experiment}_{tm_structure}.png'
        fig.savefig(file, dpi=300)
        plt.close(fig)


_ls = ['--', '-.', '-']
_lw = [1, 1.5, 2]
# for var_sim in ["M_transp", "M_q_ss", "denit_s", "min_s", "nit_s", "ndep_s", "nh4_up", "nfix_s", "ngas_s"]:
#     for lys_experiment in lys_experiments:
#         fig, axes = plt.subplots(1, 1, figsize=(6, 2))
#         for i, tm_structure in enumerate(tm_structures):
#             # load simulations
#             sim_file = base_path.parent / "output" / "svat_crop_nitrate_reference" / f"SVATCROPNITRATE_{tm_structure}_{lys_experiment}.nc"
#             ds_nitrate_sim = xr.open_dataset(sim_file, engine="h5netcdf")

#             time_origin = ds_nitrate_sim['Time'].attrs['time_origin']
#             days_sim = (ds_nitrate_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
#             date_nitrate_sim = num2date(days_sim, units=f"days since {ds_nitrate_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
#             ds_nitrate_sim = ds_nitrate_sim.assign_coords(date=("Time", date_nitrate_sim))

#             sim_vals1 = onp.cumsum(ds_nitrate_sim[var_sim].isel(x=2, y=0).values)
#             axes.plot(date_nitrate_sim, sim_vals1, color='black', ls=_ls[i], lw=_lw[i], label=tm_structure)
#         axes.set_ylim(0, )
#         axes.set_ylabel("[mg]")
#         axes.set_xlabel("Time [year]")
#         axes.set_xlim(date_obs[0], date_obs[-1])
#         fig.tight_layout()
#         file = base_path_figs / f'{var_sim}_{lys_experiment}.png'
#         fig.savefig(file, dpi=300)
#         plt.close(fig)

for var_sim in ["C_s"]:
    for lys_experiment in lys_experiments:
        fig, axes = plt.subplots(1, 1, figsize=(6, 2))
        for i, tm_structure in enumerate(tm_structures):
            # load simulations
            sim_file = base_path.parent / "output" / "svat_crop_nitrate_reference" / f"SVATCROPNITRATE_{tm_structure}_{lys_experiment}.nc"
            ds_nitrate_sim = xr.open_dataset(sim_file, engine="h5netcdf")

            time_origin = ds_nitrate_sim['Time'].attrs['time_origin']
            days_sim = (ds_nitrate_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
            date_nitrate_sim = num2date(days_sim, units=f"days since {ds_nitrate_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
            ds_nitrate_sim = ds_nitrate_sim.assign_coords(date=("Time", date_nitrate_sim))

            sim_vals1 = ds_nitrate_sim[var_sim].isel(x=2, y=0).values
            axes.plot(date_nitrate_sim, sim_vals1, color='black', ls=_ls[i], lw=_lw[i], label=tm_structure)
        axes.set_ylim(0, )
        axes.set_ylabel("[mg/l]")
        axes.set_xlabel("Time [year]")
        axes.set_xlim(date_obs[0], date_obs[-1])
        fig.tight_layout()
        file = base_path_figs / f'{var_sim}_{lys_experiment}.png'
        fig.savefig(file, dpi=300)
        plt.close(fig)

for var_sim in ["tt50_transp", "rt50_s", "tt50_q_ss"]:
    for lys_experiment in lys_experiments:
        fig, axes = plt.subplots(1, 1, figsize=(6, 2))
        for i, tm_structure in enumerate(tm_structures):
            # load simulations
            sim_file = base_path.parent / "output" / "svat_crop_nitrate_reference" / f"SVATCROPNITRATE_{tm_structure}_{lys_experiment}.average.nc"
            ds_nitrate_sim = xr.open_dataset(sim_file, engine="h5netcdf")

            time_origin = ds_nitrate_sim['Time'].attrs['time_origin']
            days_sim = (ds_nitrate_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
            date_nitrate_sim = num2date(days_sim, units=f"days since {ds_nitrate_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
            ds_nitrate_sim = ds_nitrate_sim.assign_coords(date=("Time", date_nitrate_sim))

            sim_vals1 = ds_nitrate_sim[var_sim].isel(x=2, y=0).values.astype(onp.float32) / 8.64e+13
            axes.plot(date_nitrate_sim, sim_vals1, color='black', ls=_ls[i], lw=_lw[i], label=tm_structure)
        axes.set_ylim(0, )
        axes.set_ylabel("[days]")
        axes.set_xlabel("Time [year]")
        axes.set_xlim(date_nitrate_sim[0], date_nitrate_sim[-1])
        fig.tight_layout()
        file = base_path_figs  / f'{var_sim}_{lys_experiment}.png'
        fig.savefig(file, dpi=300)
        plt.close(fig)

        
