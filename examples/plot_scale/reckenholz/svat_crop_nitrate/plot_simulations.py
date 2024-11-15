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

_LABS = {'NO3_perc_bs': '$NO_3$ [mg/l]', 
         'NO3_perc_mass_bs': '$NO_3$-N [mg]'}

base_path = Path(__file__).parent

# plot observed and simulated time series
base_path_figs = base_path.parent / "figures" / "svat_crop_nitrate"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

colors = ['#fee8c8', '#fc8d59', '#b30000']
lys_experiments = ["lys8", "lys3", "lys2"]
tm_structures = ['complete-mixing', 'advection-dispersion-power',
                    'time-variant_advection-dispersion-power']

lys_experiments = ["lys3"]
tm_structures = ['complete-mixing', 'advection-dispersion-power',
                 'time-variant_advection-dispersion-power']
for lys_experiment in lys_experiments:
    for tm_structure in tm_structures:
        # load simulations
        sim_file = base_path.parent / "output" / "svat_crop" / f"SVATCROP_{lys_experiment}.nc"
        ds_water_sim = xr.open_dataset(sim_file, engine="h5netcdf")

        time_origin = ds_water_sim['Time'].attrs['time_origin']
        days_sim = (ds_water_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date_water_sim = num2date(days_sim, units=f"days since {ds_water_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_water_sim = ds_water_sim.assign_coords(date=("Time", date_water_sim))

        sim_file = base_path.parent / "output" / "svat_crop_nitrate" / f"SVATCROPNITRATE_{tm_structure}_{lys_experiment}.nc"
        ds_nitrate_sim = xr.open_dataset(sim_file, engine="h5netcdf")

        time_origin = ds_nitrate_sim['Time'].attrs['time_origin']
        days_sim = (ds_nitrate_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date_nitrate_sim = num2date(days_sim, units=f"days since {ds_nitrate_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_nitrate_sim = ds_nitrate_sim.assign_coords(date=("Time", date_nitrate_sim))


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
        axes[2].set_xlim(date_water_sim[0], date_water_sim[-1])
        fig.tight_layout()
        file = base_path_figs / f'tt_rt_{lys_experiment}_{tm_structure}.png'
        fig.savefig(file, dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(6, 6))
        sim_vals = ds_nitrate_sim['temp_soil'].isel(x=1, y=0).values
        axes[0].plot(date_nitrate_sim, sim_vals, color="red")
        axes[0].set_ylabel("Soil temperature \n [degC]")
        axes[0].set_xlim(date_water_sim[0], date_water_sim[-1])

        ssat1 = ds_nitrate_sim['S_sat_rz'].isel(x=1, y=0).values * 0.7
        ssat2 = ds_nitrate_sim['S_sat_rz'].isel(x=1, y=0).values
        axes[1].plot(date_nitrate_sim, ssat2, color="blue") 
        axes[1].plot(date_nitrate_sim, ssat1, color="blue")
        sim_vals = onp.sum(ds_nitrate_sim['sa_rz'].isel(x=1, y=0).values, axis=-1)
        axes[1].plot(date_nitrate_sim, sim_vals, color="red", ls="--")
        sim_vals = ds_water_sim['S_rz'].isel(x=0, y=0).values
        axes[1].plot(date_water_sim, sim_vals, color="red")
        axes[1].set_ylabel("Soil water content \n [mm]")
        axes[1].set_xlim(date_water_sim[0], date_water_sim[-1])
        axes[1].set_ylim(0, )

        sim_vals = ds_nitrate_sim['denit_s'].isel(x=1, y=0).values
        axes[2].plot(date_nitrate_sim, sim_vals1, color=colors[x])
        axes[2].set_ylim(0, )
        axes[2].set_ylabel("denitrification \n[mg/day]")
        axes[2].set_xlabel("Time [year]")
        axes[2].set_xlim(date_water_sim[0], date_water_sim[-1])
        fig.tight_layout()
        file = base_path_figs / f'soiltemp_swc_denit_{lys_experiment}_{tm_structure}.png'
        fig.savefig(file, dpi=300)
        plt.close(fig)


        fig, axes = plt.subplots(2, 1, figsize=(6, 4))
        sim_vals = ds_nitrate_sim['Nfert'].isel(x=1, y=0).values
        axes[0].plot(date_nitrate_sim, sim_vals, color="red")
        axes[0].set_ylabel("Fertilisation\n [mg]")
        axes[0].set_xlim(date_water_sim[0], date_water_sim[-1])
        axes[0].set_ylim(0, )


        sim_vals1 = onp.sum(ds_nitrate_sim['msa_rz'].isel(x=1, y=0).values, axis=-1)
        axes[1].plot(date_nitrate_sim, sim_vals1, color=colors[x])
        axes[1].set_ylim(0, )
        axes[1].set_ylabel("$NO_3$-N in RZ \n[mg]")
        axes[1].set_xlabel("Time [year]")
        axes[1].set_xlim(date_water_sim[0], date_water_sim[-1])
        fig.tight_layout()
        file = base_path_figs / f'Fert_Nsoil_{lys_experiment}_{tm_structure}.png'
        fig.savefig(file, dpi=300)
        plt.close(fig)


_ls = ['--', '-.', '-']
_lw = [1, 1.5, 2]
for var_sim in ["M_transp", "M_q_ss", "denit_s", "min_s", "nit_s", "ndep_s", "nh4_up", "nfix_s", "ngas_s"]:
    for lys_experiment in lys_experiments:
        fig, axes = plt.subplots(1, 1, figsize=(6, 2))
        for i, tm_structure in enumerate(tm_structures):
            # load simulations
            sim_file = base_path.parent / "output" / "svat_crop_nitrate" / f"SVATCROPNITRATE_{tm_structure}_{lys_experiment}.nc"
            ds_nitrate_sim = xr.open_dataset(sim_file, engine="h5netcdf")

            time_origin = ds_nitrate_sim['Time'].attrs['time_origin']
            days_sim = (ds_nitrate_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
            date_nitrate_sim = num2date(days_sim, units=f"days since {ds_nitrate_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
            ds_nitrate_sim = ds_nitrate_sim.assign_coords(date=("Time", date_nitrate_sim))

            sim_vals1 = onp.cumsum(ds_nitrate_sim[var_sim].isel(x=2, y=0).values)
            axes.plot(date_nitrate_sim, sim_vals1, color='black', ls=_ls[i], lw=_lw[i], label=tm_structure)
        axes.set_ylim(0, )
        axes.set_ylabel("[mg]")
        axes.set_xlabel("Time [year]")
        axes.set_xlim(date_water_sim[0], date_water_sim[-1])
        fig.tight_layout()
        file = base_path_figs / f'{var_sim}_{lys_experiment}.png'
        fig.savefig(file, dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(1, 1, figsize=(6, 2))
        for i, tm_structure in enumerate(tm_structures):
            # load simulations
            sim_file = base_path.parent / "output" / "svat_crop_nitrate" / f"SVATCROPNITRATE_{tm_structure}_{lys_experiment}.nc"
            ds_nitrate_sim = xr.open_dataset(sim_file, engine="h5netcdf")

            time_origin = ds_nitrate_sim['Time'].attrs['time_origin']
            days_sim = (ds_nitrate_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
            date_nitrate_sim = num2date(days_sim, units=f"days since {ds_nitrate_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
            ds_nitrate_sim = ds_nitrate_sim.assign_coords(date=("Time", date_nitrate_sim))

            sim_vals1 = ds_nitrate_sim[var_sim].isel(x=2, y=0).values
            df = pd.DataFrame(index=date_nitrate_sim[1:], columns=['sim'])
            df.loc[:, 'sim'] = sim_vals1[1:] * 0.01 # convert from mg/m2 to kg/ha
            df_annual = df.resample('YE').sum()
            axes.plot(df_annual.index, df_annual.values, color='black', ls=_ls[i], lw=_lw[i], label=tm_structure)
        axes.set_ylim(0, )
        axes.set_ylabel("[kg/ha]")
        axes.set_xlabel("Time [year]")
        fig.tight_layout()
        file = base_path_figs / f'{var_sim}_{lys_experiment}_annual.png'
        fig.savefig(file, dpi=300)
        plt.close(fig)

for var_sim in ["C_s"]:
    for lys_experiment in lys_experiments:
        fig, axes = plt.subplots(1, 1, figsize=(6, 2))
        for i, tm_structure in enumerate(tm_structures):
            # load simulations
            sim_file = base_path.parent / "output" / "svat_crop_nitrate" / f"SVATCROPNITRATE_{tm_structure}_{lys_experiment}.nc"
            ds_nitrate_sim = xr.open_dataset(sim_file, engine="h5netcdf")

            time_origin = ds_nitrate_sim['Time'].attrs['time_origin']
            days_sim = (ds_nitrate_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
            date_nitrate_sim = num2date(days_sim, units=f"days since {ds_nitrate_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
            ds_nitrate_sim = ds_nitrate_sim.assign_coords(date=("Time", date_nitrate_sim))

            sim_vals1 = ds_nitrate_sim[var_sim].isel(x=2, y=0).values * 4.427
            axes.plot(date_nitrate_sim, sim_vals1, color='black', ls=_ls[i], lw=_lw[i], label=tm_structure)
        axes.set_ylim(0, )
        axes.set_ylabel("[mg/l]")
        axes.set_xlabel("Time [year]")
        axes.set_xlim(date_water_sim[0], date_water_sim[-1])
        fig.tight_layout()
        file = base_path_figs / f'{var_sim}_{lys_experiment}.png'
        fig.savefig(file, dpi=300)
        plt.close(fig)

for var_sim in ["tt50_transp", "rt50_s", "tt50_q_ss"]:
    for lys_experiment in lys_experiments:
        fig, axes = plt.subplots(1, 1, figsize=(6, 2))
        for i, tm_structure in enumerate(tm_structures):
            # load simulations
            sim_file = base_path.parent / "output" / "svat_crop_nitrate" / f"SVATCROPNITRATE_{tm_structure}_{lys_experiment}.nc"
            ds_nitrate_sim = xr.open_dataset(sim_file, engine="h5netcdf")

            time_origin = ds_nitrate_sim['Time'].attrs['time_origin']
            days_sim = (ds_nitrate_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
            date_nitrate_sim = num2date(days_sim, units=f"days since {ds_nitrate_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
            ds_nitrate_sim = ds_nitrate_sim.assign_coords(date=("Time", date_nitrate_sim))

            sim_vals1 = ds_nitrate_sim[var_sim].isel(x=2, y=0).values.astype(onp.float32) / 8.64e+13
            sim_vals1 = onp.where(sim_vals1 >= 1000., onp.nan, sim_vals1)
            axes.plot(date_nitrate_sim, sim_vals1, color='black', ls=_ls[i], lw=_lw[i], label=tm_structure)
        axes.set_ylim(0, )
        axes.set_ylabel("[days]")
        axes.set_xlabel("Time [year]")
        axes.set_xlim(date_nitrate_sim[0], date_nitrate_sim[-1])
        fig.tight_layout()
        file = base_path_figs / f'{var_sim}_{lys_experiment}.png'
        fig.savefig(file, dpi=300)
        plt.close(fig)




        
