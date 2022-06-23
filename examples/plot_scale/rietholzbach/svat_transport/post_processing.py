from pathlib import Path
import os
import glob
import datetime
import h5netcdf
import matplotlib.pyplot as plt
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp

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
path = str(base_path / "SVATTRANSPORT.*.nc")
diag_files = glob.glob(path)
states_tm_file = base_path / "states_tm.nc"
with h5netcdf.File(states_tm_file, 'w', decode_vlen_strings=False) as f:
    f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title='RoGeR transport model results at Rietholzbach Lysimeter site',
        institution='University of Freiburg, Chair of Hydrology',
        references='',
        comment='SVAT transport model with free drainage'
    )
    # collect dimensions
    for dfs in diag_files:
        with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
            # set dimensions with a dictionary
            dict_dim = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'ages': len(df.variables['ages']), 'nages': len(df.variables['nages']), 'n_sas_params': len(df.variables['n_sas_params'])}
            if not dfs.split('/')[-1].split('.')[1] == 'constant' and 'Time' not in list(dict_dim.keys()):
                dict_dim['Time'] = len(df.variables['Time'])
                time = onp.array(df.variables.get('Time'))
    for dfs in diag_files:
        with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
            if not f.dimensions:
                f.dimensions = dict_dim
                v = f.create_variable('x', ('x',), float)
                v.attrs['long_name'] = 'Number of model run'
                v.attrs['units'] = ''
                v[:] = onp.arange(dict_dim["x"])
                v = f.create_variable('y', ('y',), float)
                v.attrs['long_name'] = ''
                v.attrs['units'] = ''
                v[:] = onp.arange(dict_dim["y"])
                v = f.create_variable('ages', ('ages',), float)
                v.attrs['long_name'] = 'Water ages'
                v.attrs['units'] = 'days'
                v[:] = onp.arange(1, dict_dim["ages"]+1)
                v = f.create_variable('nages', ('nages',), float)
                v.attrs['long_name'] = 'Water ages (cumulated)'
                v.attrs['units'] = 'days'
                v[:] = onp.arange(0, dict_dim["nages"])
                v = f.create_variable('n_sas_params', ('n_sas_params',), float)
                v.attrs['long_name'] = 'Number of SAS parameters'
                v.attrs['units'] = ''
                v[:] = onp.arange(0, dict_dim["n_sas_params"])
                v = f.create_variable('Time', ('Time',), float)
                var_obj = df.variables.get('Time')
                v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                               units=var_obj.attrs["units"])
                v[:] = time
            for var_sim in list(df.variables.keys()):
                var_obj = df.variables.get(var_sim)
                if var_sim not in list(dict_dim.keys()) and ('Time', 'y', 'x') == var_obj.dimensions:
                    v = f.create_variable(var_sim, ('x', 'y', 'Time'), float)
                    vals = onp.array(var_obj)
                    v[:, :, :] = vals.swapaxes(0, 2)
                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                   units=var_obj.attrs["units"])
                elif var_sim not in list(dict_dim.keys()) and ('Time', 'n_sas_params', 'y', 'x') == var_obj.dimensions:
                    v = f.create_variable(var_sim, ('x', 'y', 'n_sas_params'), float)
                    vals = onp.array(var_obj)
                    vals = vals.swapaxes(0, 3)
                    vals = vals.swapaxes(1, 2)
                    v[:, :, :] = vals[:, :, :, 0]
                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                   units=var_obj.attrs["units"])
                elif var_sim not in list(dict_dim.keys()) and ('Time', 'ages', 'y', 'x') == var_obj.dimensions:
                    v = f.create_variable(var_sim, ('x', 'y', 'Time', 'ages'), float)
                    vals = onp.array(var_obj)
                    vals = vals.swapaxes(0, 3)
                    vals = vals.swapaxes(1, 2)
                    vals = vals.swapaxes(2, 3)
                    v[:, :, :, :] = vals
                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                   units=var_obj.attrs["units"])
                elif var_sim not in list(dict_dim.keys()) and ('Time', 'nages', 'y', 'x') == var_obj.dimensions:
                    v = f.create_variable(var_sim, ('x', 'y', 'Time', 'nages'), float)
                    vals = onp.array(var_obj)
                    vals = vals.swapaxes(0, 3)
                    vals = vals.swapaxes(1, 2)
                    vals = vals.swapaxes(2, 3)
                    v[:, :, :, :] = vals
                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                   units=var_obj.attrs["units"])


# load simulation
ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
states_hm_file = base_path / "states_hm.nc"
ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")

# load observations (measured data)
path_obs = base_path.parent / "observations" / "rietholzbach_lysimeter.nc"
ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")

# plot observed and simulated time series
base_path_figs = base_path / "figures"

# assign date
days_sim_hm = (ds_sim_hm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
date_sim_hm = num2date(days_sim_hm, units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
ds_sim_hm = ds_sim_hm.assign_coords(date=("Time", date_sim_hm))
ds_sim_tm = ds_sim_tm.assign_coords(date=("Time", date_sim_tm))
ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

# # compare observations and simulations
# nrow = 0
# ncol = 0
# idx = ds_sim_tm.date.values  # time index
# d18O_perc_cs = onp.zeros((1, 1, len(idx)))
# # calculate simulated oxygen-18 composite sample
# df_idx_cs = pd.DataFrame(index=date_obs, columns=['sol'])
# df_idx_cs.loc[:, 'sol'] = ds_obs['d18O_PERC'].isel(x=0, y=0).values
# idx_cs = df_idx_cs['sol'].dropna().index
# df_perc_18O_obs = pd.DataFrame(index=idx_cs, columns=['perc_obs', 'd18O_perc_obs'])
# df_perc_18O_obs.loc[:, 'perc_obs'] = ds_obs['PERC'].isel(x=nrow, y=ncol).values
# df_perc_18O_obs.loc[:, 'd18O_perc_obs'] = ds_obs['d18O_PERC'].isel(x=nrow, y=ncol).values
# sample_no = pd.DataFrame(index=df_perc_18O_obs.dropna().index, columns=['sample_no'])
# sample_no = sample_no.loc['1997':'2007']
# sample_no['sample_no'] = range(len(sample_no.index))
# df_perc_18O_sim = pd.DataFrame(index=idx, columns=['perc_sim', 'd18O_perc_sim'])
# df_perc_18O_sim['perc_sim'] = ds_sim_hm['q_ss'].isel(x=nrow, y=ncol).values
# df_perc_18O_sim['d18O_perc_sim'] = ds_sim_tm['C_q_ss'].isel(x=nrow, y=ncol).values
# df_perc_18O_sim = df_perc_18O_sim.join(sample_no)
# df_perc_18O_sim.loc[:, 'sample_no'] = df_perc_18O_sim.loc[:, 'sample_no'].fillna(method='bfill', limit=14)
# perc_sum = df_perc_18O_sim.groupby(['sample_no']).sum().loc[:, 'perc_sim']
# sample_no['perc_sum'] = perc_sum.values
# df_perc_18O_sim = df_perc_18O_sim.join(sample_no['perc_sum'])
# df_perc_18O_sim.loc[:, 'perc_sum'] = df_perc_18O_sim.loc[:, 'perc_sum'].fillna(method='bfill', limit=14)
# df_perc_18O_sim['weight'] = df_perc_18O_sim['perc_sim'] / df_perc_18O_sim['perc_sum']
# df_perc_18O_sim['d18O_weight'] = df_perc_18O_sim['d18O_perc_sim'] * df_perc_18O_sim['weight']
# d18O_sample = df_perc_18O_sim.groupby(['sample_no']).sum().loc[:, 'd18O_weight']
# sample_no['d18O_sample'] = d18O_sample.values
# df_perc_18O_sim = df_perc_18O_sim.join(sample_no['d18O_sample'])
# df_perc_18O_sim.loc[:, 'd18O_sample'] = df_perc_18O_sim.loc[:, 'd18O_sample'].fillna(method='bfill', limit=14)
# cond = (df_perc_18O_sim['d18O_sample'] == 0)
# df_perc_18O_sim.loc[cond, 'd18O_sample'] = onp.NaN
# d18O_perc_cs[nrow, ncol, :] = df_perc_18O_sim.loc[:, 'd18O_sample'].values
# # calculate observed oxygen-18 composite sample
# df_perc_18O_obs.loc[:, 'd18O_perc_cs'] = df_perc_18O_obs['d18O_perc_obs'].fillna(method='bfill', limit=14)

# perc_sample_sum_obs = df_perc_18O_sim.join(df_perc_18O_obs).groupby(['sample_no']).sum().loc[:, 'perc_obs']
# sample_no['perc_obs_sum'] = perc_sample_sum_obs.values
# df_perc_18O_sim = df_perc_18O_sim.join(sample_no['perc_obs_sum'])
# df_perc_18O_sim.loc[:, 'perc_obs_sum'] = df_perc_18O_sim.loc[:, 'perc_obs_sum'].fillna(method='bfill', limit=14)

# vars_tt_sim = ['tt_q_ss']
# vars_TT_sim = ['TT_q_ss']
# vars_sim = ['q_ss']
# for var_tt_sim, var_TT_sim, var_sim in zip(vars_tt_sim, vars_TT_sim, vars_sim):
#     # plot cumulative travel time distributions
#     TT = ds_sim_tm[var_TT_sim].isel(x=nrow, y=ncol).values
#     fig, axs = plt.subplots()
#     for i in range(len(ds_sim_tm[var_sim].Time)):
#         axs.plot(TT[i, :], lw=1, color='grey')
#     axs.set_xlim((0, 1200))
#     axs.set_ylim((0, 1))
#     axs.set_ylabel('$P_Q(T)$')
#     axs.set_xlabel('T [days]')
#     fig.tight_layout()
#     file_str = '%s.pdf' % (var_sim)
#     path_fig = base_path_figs / file_str
#     fig.savefig(path_fig, dpi=250)

#     tt = ds_sim_tm[var_sim].isel(x=nrow, y=ncol).values
#     # calculate mean travel time for each time step
#     mtt = onp.sum(tt * ds_sim_tm['ages'].values[onp.newaxis, :], axis=1)
#     mtt[mtt == 0] = onp.NaN
#     # calculate median travel time for each time step
#     mediantt = onp.zeros((len(ds_sim_tm['ages'].values)))
#     for i in range(len(ds_sim_tm['Time'].values)):
#         mediant = onp.where(TT[i, :] >= 0.5)[0]
#         if len(mediant) == 0:
#             mediantt[i] = onp.NaN
#         else:
#             mediantt[i] = mediant[0]
#     # calculate lower interquartile travel time for each time step
#     tt25 = onp.zeros((len(ds_sim_tm['Time'].values)))
#     for i in range(len(ds_sim_tm['Time'].values)):
#         t25 = onp.where(TT[i, :] >= 0.25)[0]
#         if len(t25) == 0:
#             tt25[i] = onp.NaN
#         else:
#             tt25[i] = t25[0]
#     # calculate lower interquartile travel time for each time step
#     tt75 = onp.zeros((len(ds_sim_tm['Time'].values)))
#     for i in range(len(ds_sim_tm['Time'].values)):
#         t75 = onp.where(TT[i, :] >= 0.75)[0]
#         if len(t75) == 0:
#             tt75[i] = onp.NaN
#         else:
#             tt75[i] = t75[0]
#     # calculate upper interquartile travel time for each time step
#     df_tt = pd.DataFrame(index=idx[1:], columns=['MTT', 'MEDIANTT', 'TT25', 'TT75'])
#     df_tt.loc[:, 'MTT'] = mtt
#     df_tt.loc[:, 'MEDIANTT'] = mediantt
#     df_tt.loc[:, 'TT25'] = tt25
#     df_tt.loc[:, 'TT75'] = tt75
#     df_tt.loc[:, var_sim] = ds_sim_hm[var_sim].isel(x=nrow, y=ncol).values

#     # mean and median travel time over entire simulation period
#     df_tt_mean_median = pd.DataFrame(index=['mean', 'median'], columns=['MTT', 'MEDIANTT'])
#     df_tt_mean_median.loc['mean', 'MTT'] = onp.nanmean(df_tt['MTT'].values)
#     df_tt_mean_median.loc['mean', 'MEDIANTT'] = onp.nanmean(df_tt['MEDIANTT'].values)
#     df_tt_mean_median.loc['median', 'MTT'] = onp.nanmedian(df_tt['MTT'].values)
#     df_tt_mean_median.loc['median', 'MEDIANTT'] = onp.nanmedian(df_tt['MEDIANTT'].values)
#     file_str = 'tt_mean_median_%s.pdf' % (var_sim)
#     path_csv = base_path_figs / file_str
#     df_tt_mean_median.to_csv(path_csv, header=True, index=True, sep="\t")

#     # plot mean and median travel time
#     fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 7))
#     axes[0].plot(df_tt.index, df_tt['MTT'], ls='--', lw=2, color='magenta')
#     axes[0].plot(df_tt.index, df_tt['MEDIANTT'], ls=':', lw=2, color='purple')
#     axes[0].fill_between(df_tt.index, df_tt['TT25'], df_tt['TT75'], color='purple',
#                          edgecolor=None, alpha=0.2)
#     tt_50 = str(int(df_tt_mean_median.loc['mean', 'MEDIANTT']))
#     tt_mean = str(int(df_tt_mean_median.loc['mean', 'MTT']))
#     axes[0].text(0.75, 0.93, r'$\overline{TT}_{50}$: %s days' % (tt_50), size=12, horizontalalignment='left',
#                  verticalalignment='center', transform=axes[1].transAxes)
#     axes[0].text(0.75, 0.83, r'$\overline{TT}$: %s days' % (tt_mean), size=12, horizontalalignment='left',
#                  verticalalignment='center', transform=axes[1].transAxes)
#     axes[0].set_ylabel('age\n[days]')
#     axes[0].set_ylim((0, 500))
#     axes[0].set_xlim((df_tt.index[0], df_tt.index[-1]))
#     axes[0].text(0.985, 0.05, '(b)', size=15, horizontalalignment='center',
#                  verticalalignment='center', transform=axes[1].transAxes)
#     axes[1].bar(df_tt.index, df_tt['PERC'], width=-1, align='edge', edgecolor='grey')
#     axes[1].set_ylim(0,)
#     axes[1].invert_yaxis()
#     axes[1].set_xlim((df_tt.index[0], df_tt.index[-1]))
#     axes[1].set_ylabel('Percolation\n[mm $day^{-1}$]')
#     axes[1].set_xlabel(r'Time [year]')
#     axes[1].text(0.985, 0.05, '(c)', size=15, horizontalalignment='center',
#                  verticalalignment='center', transform=axes[2].transAxes)
#     fig.tight_layout()
#     file_str = 'mean_median_tt_%s.pdf' % (var_sim)
#     path_fig = base_path_figs / file_str
#     fig.savefig(path_fig, dpi=250)


# write HYDRUS-1D input
# monthly variable soil cover fraction of grass
df_scf = pd.DataFrame(index=date_obs, columns=['scf'])
df_scf.loc[df_scf.index.month==1] = 1. - onp.exp(-0.463*1)
df_scf.loc[df_scf.index.month==2] = 1. - onp.exp(-0.463*1)
df_scf.loc[df_scf.index.month==3] = 1. - onp.exp(-0.463*1.33)
df_scf.loc[df_scf.index.month==4] = 1. - onp.exp(-0.463*1.66)
df_scf.loc[df_scf.index.month==5] = 1. - onp.exp(-0.463*2)
df_scf.loc[df_scf.index.month==6] = 1. - onp.exp(-0.463*2)
df_scf.loc[df_scf.index.month==7] = 1. - onp.exp(-0.463*2)
df_scf.loc[df_scf.index.month==8] = 1. - onp.exp(-0.463*2)
df_scf.loc[df_scf.index.month==9] = 1. - onp.exp(-0.463*2)
df_scf.loc[df_scf.index.month==10] = 1. - onp.exp(-0.463*1.33)
df_scf.loc[df_scf.index.month==11] = 1. - onp.exp(-0.463*1.16)
df_scf.loc[df_scf.index.month==12] = 1. - onp.exp(-0.463*1)

df_rsoil = pd.DataFrame(index=date_obs, columns=['rSoil'])
df_rsoil.loc[:, 'rSoil'] = ds_obs['PET'].isel(x=0, y=0).values * (1 - df_scf['scf'].values)
df_rsoil.iloc[0, 0] = 0.05
df_rroot = pd.DataFrame(index=date_obs, columns=['rRoot'])
df_rroot.loc[:, 'rRoot'] = ds_obs['PET'].isel(x=0, y=0).values * df_scf['scf'].values
df_rroot.iloc[0, 0] = 0.03
df_prec = pd.DataFrame(index=date_sim_hm, columns=['Prec'])
df_prec.loc[:, 'Prec'] = ds_sim_hm['rain_top'].isel(x=0, y=0).values + ds_sim_hm['snow_melt'].isel(x=0, y=0).values
df_ttop = pd.DataFrame(index=date_obs, columns=['tTop'])
df_ttop.loc[:, 'tTop'] = ds_obs['TA'].isel(x=0, y=0).values
df_hcrita = pd.DataFrame(index=date_obs, columns=['hCritA'])
df_hcrita.loc[:, 'hCritA'] = 100000
df_ampl = pd.DataFrame(index=date_obs, columns=['Ampl'])
df_ampl.loc[:, 'Ampl'] = 0
df_tatm = pd.DataFrame(index=date_obs, columns=['tAtm'])
df_tatm.loc[:, 'tAtm'] = range(1, len(df_tatm.index)+1)
df_ctop = pd.DataFrame(index=date_sim_tm, columns=['cTop'])
df_ctop.loc[:, 'cTop'] = ds_sim_tm['C_in'].isel(x=0, y=0).values + 20

df = pd.DataFrame(index=date_obs)
df = df.join([df_tatm, df_prec, df_rsoil, df_rroot, df_hcrita, df_ttop, df_ampl, df_ctop])

file = base_path / "results" / "atmosphere_daily_18O.csv"
df.to_csv(file, header=True, index=False, sep=";")

onp.nansum(ds_obs['PREC'].isel(x=0, y=0).values)
onp.sum(ds_sim_hm['prec'].isel(x=0, y=0).values)
onp.sum(ds_sim_hm['rain_top'].isel(x=0, y=0).values)
onp.sum(ds_sim_hm['snow_melt'].isel(x=0, y=0).values)
df_prec.sum()

# compare original and snow-corrected input signal
df_d18O_prec = pd.DataFrame(index=date_obs)
df_d18O_prec.loc[:, 'd18O_prec'] = ds_obs['d18O_PREC'].isel(x=0, y=0).values
df_d18O_in = pd.DataFrame(index=date_obs)
df_d18O_in = df_d18O_in.join(df_ctop)
df_d18O_in = df_d18O_in.join(df_d18O_prec)

fig, ax = plt.subplots()
ax.scatter(df_d18O_in.index, df_d18O_in['d18O_prec'], s=2, color='black')
ax.scatter(df_d18O_in.index, df_d18O_in['cTop'], s=2, color='red')
# ax.set_ylim(0,)
ax.set_xlim((df_d18O_in.index[0], df_d18O_in.index[-1]))
ax.set_ylabel(r'$d_{18}O$ [permil]')
ax.set_xlabel(r'Time [year]')
