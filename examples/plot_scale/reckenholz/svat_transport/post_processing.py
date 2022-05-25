from pathlib import Path
import os
import pandas as pd
import numpy as onp
import xarray as xr
import h5netcdf
from cftime import num2date
import matplotlib.pyplot as plt

base_path = Path(__file__).parent

lys_experiments = ["lys3", "lys4", "lys8", "lys9"]
for lys_experiment in lys_experiments:
    # directory of results
    base_path_results = base_path / "results" / lys_experiment
    if not os.path.exists(base_path_results):
        os.mkdir(base_path_results)
    # directory of figures
    base_path_figs = base_path / "figures" / lys_experiment
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    # load simulation
    states_tm_file = base_path / "states_tm.nc"
    states_hm_file = base_path / "states_hm.nc"
    ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf", group=lys_experiment)
    ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf", group=lys_experiment)

    # load observations (measured data)
    path_obs = Path("/Users/robinschwemmle/Desktop/PhD/data/plot/rietholzbach/rietholzbach_lysimeter.nc")
    ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")

    time_origin = ds_sim_tm['Time'].attrs['time_origin']
    days = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim = num2date(days, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    date_obs = num2date(days, units=f"days since {ds_obs['time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_tm = ds_sim_tm.assign_coords(date=("Time", date_sim))
    ds_sim_hm = ds_sim_hm.assign_coords(date=("Time", date_sim))
    ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

    # compare observations and simulations
    nrow = 0
    ncol = 0
    idx = ds_sim_tm.date.values  # time index
    NO3_perc_cs = onp.zeros((1, 1, len(idx)))
    NO3_mass_perc_cs = onp.zeros((1, 1, len(idx)))
    # calculate simulated nitrate composite samples
    sample_no = pd.DataFrame(index=idx, columns=['sample_no'])
    sample_no['sample_no'] = range(len(sample_no.index))
    df_perc_NO3_sim_obs = pd.DataFrame(index=idx, columns=['perc_sim', 'NO3_mass_sim', 'perc_obs', 'NO3_conc_obs'])
    df_perc_NO3_sim_obs['perc_sim'] = ds_sim_hm['q_ss'].isel(x=0, y=0).values
    df_perc_NO3_sim_obs['NO3_mass_sim'] = ds_sim_tm['M_q_ss'].isel(x=0, y=0).values
    df_perc_NO3_sim_obs['perc_obs'] = ds_obs['PERC'].isel(x=0, y=0).values
    df_perc_NO3_sim_obs['NO3_conc_obs'] = ds_sim_tm['C_q_ss'].isel(x=0, y=0).values
    df_perc_NO3_sim_obs = df_perc_NO3_sim_obs.join(sample_no)
    df_perc_NO3_sim_obs.loc[:, 'sample_no'] = df_perc_NO3_sim_obs.loc[:, 'sample_no'].fillna(method='bfill', limit=14)
    perc_sim_sum = df_perc_NO3_sim_obs.groupby(['sample_no']).sum().loc[:, 'perc_sim']
    NO3_sim_sum = df_perc_NO3_sim_obs.groupby(['sample_no']).sum().loc[:, 'NO3_mass_sim']
    perc_obs_sum = df_perc_NO3_sim_obs.groupby(['sample_no']).sum().loc[:, 'perc_obs']
    NO3_obs_sum = df_perc_NO3_sim_obs.groupby(['sample_no']).mean().loc[:, 'NO3_conc_obs']
    sample_no['perc_sim_sum'] = perc_sim_sum.values
    sample_no['NO3_mass_sim_sum'] = NO3_sim_sum.values
    sample_no['perc_obs_sum'] = perc_obs_sum.values
    sample_no['NO3_conc_obs'] = NO3_obs_sum.values
    sample_no['NO3_conc_sim'] = sample_no['NO3_mass_sim_sum'] / sample_no['perc_sim_sum']
    sample_no['NO3_mass_obs_sum'] = sample_no['NO3_conc_obs'] * sample_no['perc_obs_sum']
    df_perc_NO3_sim_obs = df_perc_NO3_sim_obs.join(sample_no['NO3_conc_sim'])
    df_perc_NO3_sim_obs = df_perc_NO3_sim_obs.join(sample_no['NO3_mass_obs_sum'])
    df_perc_NO3_sim_obs = df_perc_NO3_sim_obs.join(sample_no['NO3_mass_sim_sum'])
    # concentration of simulated composite samples
    NO3_perc_cs[nrow, ncol, :] = df_perc_NO3_sim_obs.loc[:, 'NO3_conc_sim'].values
    # mass of simulated composite samples
    NO3_mass_perc_cs[nrow, ncol, :] = df_perc_NO3_sim_obs.loc[:, 'NO3_mass_sim_sum'].values
    # write composite sample to output file
    states_tm_file = base_path / "states_tm.nc"
    with h5netcdf.File(states_tm_file, 'a', decode_vlen_strings=False) as f:
        v = f.groups[tm_structure].create_variable('d18O_perc_cs', ('x', 'y', 'Time'), float)
        v[:, :, :] = d18O_perc_cs
        v.attrs.update(long_name="composite sample of d18O in percolation",
                       units="permil")

    vars_tt_sim = ['tt_q_ss']
    vars_TT_sim = ['TT_q_ss']
    vars_sim = ['q_ss']
    for var_tt_sim, var_TT_sim, var_sim in zip(vars_tt_sim, vars_TT_sim, vars_sim):
        # plot cumulative travel time distributions
        TT = ds_sim_tm[var_TT_sim].isel(x=nrow, y=ncol).values
        fig, axs = plt.subplots()
        for i in range(len(ds_sim_tm[var_sim].Time)):
            axs.plot(TT[i, :], lw=1, color='grey')
        axs.set_xlim((0, 1200))
        axs.set_ylim((0, 1))
        axs.set_ylabel('$P_Q(T)$')
        axs.set_xlabel('T [days]')
        fig.tight_layout()
        file_str = '%s.pdf' % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=250)

        tt = ds_sim_tm[var_sim].isel(x=nrow, y=ncol).values
        # calculate mean travel time for each time step
        mtt = onp.sum(tt * ds_sim_tm['ages'].values[onp.newaxis, :], axis=1)
        mtt[mtt == 0] = onp.NaN
        # calculate median travel time for each time step
        mediantt = onp.zeros((len(ds_sim_tm['ages'].values)))
        for i in range(len(ds_sim_tm['Time'].values)):
            mediant = onp.where(TT[i, :] >= 0.5)[0]
            if len(mediant) == 0:
                mediantt[i] = onp.NaN
            else:
                mediantt[i] = mediant[0]
        # calculate lower interquartile travel time for each time step
        tt25 = onp.zeros((len(ds_sim_tm['Time'].values)))
        for i in range(len(ds_sim_tm['Time'].values)):
            t25 = onp.where(TT[i, :] >= 0.25)[0]
            if len(t25) == 0:
                tt25[i] = onp.NaN
            else:
                tt25[i] = t25[0]
        # calculate lower interquartile travel time for each time step
        tt75 = onp.zeros((len(ds_sim_tm['Time'].values)))
        for i in range(len(ds_sim_tm['Time'].values)):
            t75 = onp.where(TT[i, :] >= 0.75)[0]
            if len(t75) == 0:
                tt75[i] = onp.NaN
            else:
                tt75[i] = t75[0]
        # calculate upper interquartile travel time for each time step
        df_tt = pd.DataFrame(index=idx[1:], columns=['MTT', 'MEDIANTT', 'TT25', 'TT75'])
        df_tt.loc[:, 'MTT'] = mtt
        df_tt.loc[:, 'MEDIANTT'] = mediantt
        df_tt.loc[:, 'TT25'] = tt25
        df_tt.loc[:, 'TT75'] = tt75
        df_tt.loc[:, var_sim] = ds_sim_hm[var_sim].isel(x=nrow, y=ncol).values

        # mean and median travel time over entire simulation period
        df_tt_mean_median = pd.DataFrame(index=['mean', 'median'], columns=['MTT', 'MEDIANTT'])
        df_tt_mean_median.loc['mean', 'MTT'] = onp.nanmean(df_tt['MTT'].values)
        df_tt_mean_median.loc['mean', 'MEDIANTT'] = onp.nanmean(df_tt['MEDIANTT'].values)
        df_tt_mean_median.loc['median', 'MTT'] = onp.nanmedian(df_tt['MTT'].values)
        df_tt_mean_median.loc['median', 'MEDIANTT'] = onp.nanmedian(df_tt['MEDIANTT'].values)
        file_str = 'tt_mean_median_%s.pdf' % (var_sim)
        path_csv = base_path_figs / file_str
        df_tt_mean_median.to_csv(path_csv, header=True, index=True, sep="\t")

        # plot mean and median travel time
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 7))
        axes[0].plot(df_tt.index, df_tt['MTT'], ls='--', lw=2, color='magenta')
        axes[0].plot(df_tt.index, df_tt['MEDIANTT'], ls=':', lw=2, color='purple')
        axes[0].fill_between(df_tt.index, df_tt['TT25'], df_tt['TT75'], color='purple',
                             edgecolor=None, alpha=0.2)
        tt_50 = str(int(df_tt_mean_median.loc['mean', 'MEDIANTT']))
        tt_mean = str(int(df_tt_mean_median.loc['mean', 'MTT']))
        axes[0].text(0.75, 0.93, r'$\overline{TT}_{50}$: %s days' % (tt_50), size=12, horizontalalignment='left',
                     verticalalignment='center', transform=axes[1].transAxes)
        axes[0].text(0.75, 0.83, r'$\overline{TT}$: %s days' % (tt_mean), size=12, horizontalalignment='left',
                     verticalalignment='center', transform=axes[1].transAxes)
        axes[0].set_ylabel('age\n[days]')
        axes[0].set_ylim((0, 500))
        axes[0].set_xlim((df_tt.index[0], df_tt.index[-1]))
        axes[0].text(0.985, 0.05, '(b)', size=15, horizontalalignment='center',
                     verticalalignment='center', transform=axes[1].transAxes)
        axes[1].bar(df_tt.index, df_tt['PERC'], width=-1, align='edge', edgecolor='grey')
        axes[1].set_ylim(0,)
        axes[1].invert_yaxis()
        axes[1].set_xlim((df_tt.index[0], df_tt.index[-1]))
        axes[1].set_ylabel('Percolation\n[mm $day^{-1}$]')
        axes[1].set_xlabel(r'Time [year]')
        axes[1].text(0.985, 0.05, '(c)', size=15, horizontalalignment='center',
                     verticalalignment='center', transform=axes[2].transAxes)
        fig.tight_layout()
        file_str = 'mean_median_tt_%s.pdf' % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=250)
