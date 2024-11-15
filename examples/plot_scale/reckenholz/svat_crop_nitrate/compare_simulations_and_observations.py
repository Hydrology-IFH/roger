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

        sim_file = base_path.parent / "output" / "svat_crop_nitrate" / f"SVATCROPNITRATE_{tm_structure}_{lys_experiment}.nc"
        ds_nitrate_sim = xr.open_dataset(sim_file, engine="h5netcdf")

        # load observations (measured data)
        path_obs = Path("/Users/robinschwemmle/Desktop/PhD/data/plot/reckenholz/reckenholz_lysimeter.nc")
        ds_obs = xr.open_dataset(path_obs, engine="h5netcdf", group=lys_experiment)

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


for lys_experiment in lys_experiments:
    for tm_structure in tm_structures:
        # load simulations
        sim_file = base_path.parent / "output" / "svat_crop" / f"SVATCROP_{lys_experiment}.nc"
        ds_water_sim = xr.open_dataset(sim_file, engine="h5netcdf")

        sim_file = base_path.parent / "output" / "svat_crop_nitrate" / f"SVATCROPNITRATE_{tm_structure}_{lys_experiment}.nc"
        ds_nitrate_sim = xr.open_dataset(sim_file, engine="h5netcdf")

        # load observations (measured data)
        path_obs = Path("/Users/robinschwemmle/Desktop/PhD/data/plot/reckenholz/reckenholz_lysimeter.nc")
        ds_obs = xr.open_dataset(path_obs, engine="h5netcdf", group=lys_experiment)

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

        # compare observations and simulations
        idx = ds_nitrate_sim.date.values  # time index
        df_idx_bs = pd.DataFrame(index=date_obs, columns=['sol'])
        df_idx_bs.loc[:, 'sol'] = ds_obs['NO3_PERC'].isel(x=0, y=0).values
        idx_bs = df_idx_bs['sol'].dropna().index

        # calculate simulated nitrate bulk samples
        sample_no = pd.DataFrame(index=idx_bs, columns=['sample_no'])
        sample_no['sample_no'] = range(len(sample_no.index))
        df_perc_NO3_sim = pd.DataFrame(index=idx, columns=['perc', 'NO3_mass'])
        df_perc_NO3_sim['perc'] = ds_water_sim['q_ss'].isel(x=0, y=0).values
        df_perc_NO3_sim['NO3_mass'] = ds_nitrate_sim['M_q_ss'].isel(x=2, y=0).values
        df_perc_NO3_sim = df_perc_NO3_sim.join(sample_no)
        df_perc_NO3_sim.loc[:, 'sample_no'] = df_perc_NO3_sim.loc[:, 'sample_no'].bfill(limit=14)
        perc_sim_sum = df_perc_NO3_sim.groupby(['sample_no']).sum().loc[:, 'perc']
        NO3_sim_sum = df_perc_NO3_sim.groupby(['sample_no']).sum().loc[:, 'NO3_mass']
        sample_no['perc_sum'] = perc_sim_sum.values
        sample_no['NO3_mass_sum'] = NO3_sim_sum.values
        sample_no['NO3_conc'] = sample_no['NO3_mass_sum'] / sample_no['perc_sum']
        df_perc_NO3_sim = df_perc_NO3_sim.join(sample_no['NO3_conc'])
        df_perc_NO3_sim = df_perc_NO3_sim.join(sample_no['NO3_mass_sum'])
        # concentration of simulated bulk samples
        NO3_perc_bs = df_perc_NO3_sim.loc[:, 'NO3_conc'].values * 4.43
        # mass of simulated bulk samples
        NO3_perc_mass_bs = df_perc_NO3_sim.loc[:, 'NO3_mass_sum'].values

        # calculate metrics
        vars_sim = ['NO3_perc_bs', 'NO3_perc_mass_bs']
        vars_obs = ['NO3_PERC', 'NO3_PERC_MASS']
        for var_sim, var_obs in zip(vars_sim, vars_obs):
            fig, axes = plt.subplots(2, 1, figsize=(6, 4))
            obs_vals = ds_obs['PERC'].isel(x=0, y=0).values
            sim_vals = ds_water_sim['q_ss'].isel(x=0, y=0).values

            df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
            df_obs.loc[:, 'obs'] = obs_vals
            df_eval = eval_utils.join_obs_on_sim(date_water_sim, sim_vals, df_obs)
            df_eval = df_eval.dropna()
            obs_vals = df_eval.loc[:, 'obs'].values.astype(onp.float32)
            sim_vals = df_eval.loc[:, 'sim'].values.astype(onp.float32)

            kge_val = onp.round(eval_utils.calc_kge(obs_vals, sim_vals), 2)
            kge_alpha_val = onp.round(eval_utils.calc_kge_alpha(obs_vals, sim_vals), 2)
            kge_alpha_val = onp.round(eval_utils.calc_kge_beta(obs_vals, sim_vals), 2)
            r_val = onp.round(eval_utils.calc_temp_cor(obs_vals, sim_vals), 2)

            axes[0].plot(df_eval.index, obs_vals, color="blue")
            axes[0].plot(df_eval.index, sim_vals, color="red")
            axes[0].set_ylabel("PERC [mm/day]")
            axes[0].set_xlim(date_obs[0], date_obs[-1])
            axes[0].set_ylim(0, )
            axes[0].text(0.02, 0.9, f"KGE: {kge_val}", transform=axes[0].transAxes)

            # join observations on simulations
            obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
            if var_sim == 'NO3_perc_bs':
                sim_vals = NO3_perc_bs
                obs_vals = obs_vals * 4.427
            elif var_sim == 'NO3_perc_mass_bs':
                sim_vals = NO3_perc_mass_bs
            df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
            df_obs.loc[:, 'obs'] = obs_vals
            df_eval = eval_utils.join_obs_on_sim(date_water_sim, sim_vals, df_obs)
            df_eval = df_eval.dropna().iloc[1:-1, :]
            obs_vals = df_eval.loc[:, 'obs'].values.astype(onp.float32)
            sim_vals = df_eval.loc[:, 'sim'].values.astype(onp.float32)

            kge_val = onp.round(eval_utils.calc_kge(obs_vals, sim_vals), 2)
            kge_alpha_val = onp.round(eval_utils.calc_kge_alpha(obs_vals, sim_vals), 2)
            kge_alpha_val = onp.round(eval_utils.calc_kge_beta(obs_vals, sim_vals), 2)
            r_val = onp.round(eval_utils.calc_temp_cor(obs_vals, sim_vals), 2)

            axes[1].scatter(df_eval.index, obs_vals, color="blue", s=1)
            axes[1].scatter(df_eval.index, sim_vals, color="red", s=1)
            axes[1].set_ylim(0, )
            axes[1].set_ylabel(_LABS[var_sim])
            axes[1].set_xlabel("Time [year]")
            axes[1].set_xlim(date_obs[0], date_obs[-1])
            axes[1].text(0.02, 0.9, f"KGE: {kge_val}", transform=axes[1].transAxes)
            fig.tight_layout()
            file = base_path_figs / f'PERC_{var_sim}_{lys_experiment}_{tm_structure}.png'
            fig.savefig(file, dpi=300)
            plt.close(fig)


            fig, axes = plt.subplots(2, 1, figsize=(6, 4))
            obs_vals = ds_obs['PERC'].isel(x=0, y=0).values
            sim_vals = ds_water_sim['q_ss'].isel(x=0, y=0).values

            df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
            df_obs.loc[:, 'obs'] = obs_vals
            df_eval = eval_utils.join_obs_on_sim(date_water_sim, sim_vals, df_obs)
            df_eval = df_eval.dropna()
            obs_vals = df_eval.loc[:, 'obs'].values.astype(onp.float32)
            sim_vals = df_eval.loc[:, 'sim'].values.astype(onp.float32)

            kge_val = onp.round(eval_utils.calc_kge(obs_vals, sim_vals), 2)
            kge_alpha_val = onp.round(eval_utils.calc_kge_alpha(obs_vals, sim_vals), 2)
            kge_alpha_val = onp.round(eval_utils.calc_kge_beta(obs_vals, sim_vals), 2)
            r_val = onp.round(eval_utils.calc_temp_cor(obs_vals, sim_vals), 2)

            axes[0].plot(df_eval.index, obs_vals, color="blue")
            axes[0].plot(df_eval.index, sim_vals, color="red")
            axes[0].set_ylabel("PERC [mm/day]")
            axes[0].set_xlim(date_obs[0], date_obs[-1])
            axes[0].set_ylim(0, )
            axes[0].text(0.02, 0.9, f"KGE: {kge_val}", transform=axes[0].transAxes)

            # join observations on simulations
            obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
            if var_sim == 'NO3_perc_bs':
                sim_vals = NO3_perc_bs
                obs_vals = obs_vals * 4.427
            elif var_sim == 'NO3_perc_mass_bs':
                sim_vals = NO3_perc_mass_bs
            df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
            df_obs.loc[:, 'obs'] = obs_vals
            df_eval = eval_utils.join_obs_on_sim(date_water_sim, sim_vals, df_obs)
            df_eval = df_eval.dropna().iloc[1:-1, :]
            obs_vals = df_eval.loc[:, 'obs'].values.astype(onp.float32)
            sim_vals = df_eval.loc[:, 'sim'].values.astype(onp.float32)

            kge_val = onp.round(eval_utils.calc_kge(obs_vals, sim_vals), 2)
            kge_alpha_val = onp.round(eval_utils.calc_kge_alpha(obs_vals, sim_vals), 2)
            kge_alpha_val = onp.round(eval_utils.calc_kge_beta(obs_vals, sim_vals), 2)
            r_val = onp.round(eval_utils.calc_temp_cor(obs_vals, sim_vals), 2)
            obs_avg = onp.nanmean(obs_vals)
            sim_avg = onp.nanmean(sim_vals)

            axes[1].scatter(df_eval.index, obs_vals, color="blue", s=1)
            axes[1].scatter(df_eval.index, sim_vals, color="red", s=1)
            axes[1].axhline(obs_avg, color="blue", ls="--")
            axes[1].axhline(sim_avg, color="red", ls="--")
            axes[1].set_ylim(0, )
            axes[1].set_ylabel(_LABS[var_sim])
            axes[1].set_xlabel("Time [year]")
            axes[1].set_xlim(date_obs[0], date_obs[-1])
            axes[1].text(0.02, 0.9, f"KGE: {kge_val}", transform=axes[1].transAxes)
            fig.tight_layout()
            file = base_path_figs / f'PERC_{var_sim}_{lys_experiment}_{tm_structure}_with_avg.png'
            fig.savefig(file, dpi=300)
            plt.close(fig)

        # calculate metrics
        vars_sim = ['NO3_perc_mass_bs']
        vars_obs = ['NO3_PERC_MASS']
        for var_sim, var_obs in zip(vars_sim, vars_obs):
            fig, axes = plt.subplots(2, 1, figsize=(6, 4))
            obs_vals = ds_obs['PERC'].isel(x=0, y=0).values
            sim_vals = ds_water_sim['q_ss'].isel(x=0, y=0).values

            df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
            df_obs.loc[:, 'obs'] = obs_vals
            df_eval = eval_utils.join_obs_on_sim(date_water_sim, sim_vals, df_obs)
            df_eval = df_eval.dropna()
            obs_vals = df_eval.loc[:, 'obs'].values.astype(onp.float32)
            sim_vals = df_eval.loc[:, 'sim'].values.astype(onp.float32)

            kge_val = onp.round(eval_utils.calc_kge(obs_vals, sim_vals), 2)
            kge_alpha_val = onp.round(eval_utils.calc_kge_alpha(obs_vals, sim_vals), 2)
            kge_alpha_val = onp.round(eval_utils.calc_kge_beta(obs_vals, sim_vals), 2)
            r_val = onp.round(eval_utils.calc_temp_cor(obs_vals, sim_vals), 2)

            axes[0].plot(df_eval.index, onp.cumsum(obs_vals), color="blue")
            axes[0].plot(df_eval.index, onp.cumsum(sim_vals), color="red")
            axes[0].set_ylabel("PERC [mm]")
            axes[0].set_xlim(date_obs[0], date_obs[-1])
            axes[0].set_ylim(0, )
            axes[0].text(0.02, 0.9, f"KGE: {kge_val}", transform=axes[0].transAxes)

            # join observations on simulations
            obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
            if var_sim == 'NO3_perc_bs':
                sim_vals = NO3_perc_bs
                obs_vals = obs_vals * 4.427
            elif var_sim == 'NO3_perc_mass_bs':
                sim_vals = NO3_perc_mass_bs
            df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
            df_obs.loc[:, 'obs'] = obs_vals
            df_eval = eval_utils.join_obs_on_sim(date_water_sim, sim_vals, df_obs)
            df_eval = df_eval.dropna().iloc[1:-1, :]
            obs_vals = df_eval.loc[:, 'obs'].values.astype(onp.float32)
            sim_vals = df_eval.loc[:, 'sim'].values.astype(onp.float32)

            kge_val = onp.round(eval_utils.calc_kge(obs_vals, sim_vals), 2)
            kge_alpha_val = onp.round(eval_utils.calc_kge_alpha(obs_vals, sim_vals), 2)
            kge_alpha_val = onp.round(eval_utils.calc_kge_beta(obs_vals, sim_vals), 2)
            r_val = onp.round(eval_utils.calc_temp_cor(obs_vals, sim_vals), 2)

            axes[1].scatter(df_eval.index, onp.cumsum(obs_vals), color="blue", s=1)
            axes[1].scatter(df_eval.index, onp.cumsum(sim_vals), color="red", s=1)
            axes[1].set_ylim(0, )
            axes[1].set_ylabel("$NO_3$-N [mg]")
            axes[1].set_xlabel("Time [year]")
            axes[1].set_xlim(date_obs[0], date_obs[-1])
            axes[1].text(0.02, 0.9, f"KGE: {kge_val}", transform=axes[1].transAxes)
            fig.tight_layout()
            file = base_path_figs / f'PERC_{var_sim}_{lys_experiment}_{tm_structure}_cumulated.png'
            fig.savefig(file, dpi=300)
            plt.close(fig)

        fig, axes = plt.subplots(1, 1, figsize=(6, 2))
        for i, tm_structure in enumerate(tm_structures):
            obs_vals = ds_obs['N_UP'].isel(x=0, y=0).values[:, 0]
            # load simulations
            sim_file = base_path.parent / "output" / "svat_crop_nitrate" / f"SVATCROPNITRATE_{tm_structure}_{lys_experiment}.nc"
            ds_nitrate_sim = xr.open_dataset(sim_file, engine="h5netcdf")

            time_origin = ds_nitrate_sim['Time'].attrs['time_origin']
            days_sim = (ds_nitrate_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
            date_nitrate_sim = num2date(days_sim, units=f"days since {ds_nitrate_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
            ds_nitrate_sim = ds_nitrate_sim.assign_coords(date=("Time", date_nitrate_sim))

            sim_vals1 = ds_nitrate_sim["M_transp"].isel(x=0, y=0).values + ds_nitrate_sim["nh4_up"].isel(x=0, y=0).values
            sim_vals2 = ds_nitrate_sim["M_transp"].isel(x=0, y=0).values
            sim_vals3 = ds_nitrate_sim["nh4_up"].isel(x=0, y=0).values
            df = pd.DataFrame(index=date_nitrate_sim[1:], columns=['sim'])
            df.loc[:, 'sim'] = sim_vals1[1:] * 0.01 # convert from mg/m2 to kg/ha
            df.loc[:, 'sim1'] = sim_vals2[1:] * 0.01 # convert from mg/m2 to kg/ha
            df.loc[:, 'sim2'] = sim_vals3[1:] * 0.01 # convert from mg/m2 to kg/ha
            df_annual = df.resample('YE').sum()
            df_annual['obs'] = obs_vals

            mae_val = onp.round(eval_utils.calc_mae(df_annual['obs'].values, df_annual['sim'].values), 2)
            fig, axes = plt.subplots(1, 1, figsize=(6, 2))
            axes.plot(df_annual.index, df_annual['sim'].values, color='red', lw=1.5, label=tm_structure)
            axes.plot(df_annual.index, df_annual['sim1'].values, color='red', ls='--', label=tm_structure)
            axes.plot(df_annual.index, df_annual['sim2'].values, color='red', ls=':', label=tm_structure)
            axes.plot(df_annual.index, df_annual['obs'].values, color='blue', lw=1.5, label=tm_structure)
            axes.set_ylim(0, )
            axes.set_ylabel("Plant uptake [kg N/ha]")
            axes.set_xlabel("Time [year]")
            axes.text(0.02, 0.9, f"MAE: {mae_val}", transform=axes.transAxes)
            fig.tight_layout()
            file = base_path_figs / f'N_uptake_{lys_experiment}_{tm_structure}.png'
            fig.savefig(file, dpi=300)
            plt.close(fig)

            rbs_val = onp.round((onp.sum(df_annual['sim'].values) - onp.sum(df_annual['obs'].values)) / onp.sum(df_annual['obs'].values), 2)
            fig, axes = plt.subplots(1, 1, figsize=(6, 2))
            axes.plot(df_annual.index, df_annual['sim'].cumsum().values, color='red', label=tm_structure)
            axes.plot(df_annual.index, df_annual['obs'].cumsum().values, color='blue', label=tm_structure)
            axes.set_ylim(0, )
            axes.set_ylabel("Plant uptake [kg N/ha]")
            axes.set_xlabel("Time [year]")
            axes.text(0.02, 0.9, f"RBS: {rbs_val}", transform=axes.transAxes)
            fig.tight_layout()
            file = base_path_figs / f'N_uptake_{lys_experiment}_{tm_structure}_cumulated.png'
            fig.savefig(file, dpi=300)
            plt.close(fig)




        
