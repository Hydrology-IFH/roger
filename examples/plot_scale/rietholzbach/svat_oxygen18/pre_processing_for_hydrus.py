from pathlib import Path
import os
import matplotlib.pyplot as plt
import xarray as xr
from cftime import num2date
import pandas as pd
from datetime import timedelta
import numpy as onp

base_path = Path(__file__).parent
# directory of results
base_path_results = base_path / "results"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)
base_path_results = base_path / "results" / "hydrus_input"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# load hydrologic simulation
states_hm_file = base_path / "states_hm.nc"
ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")
days_sim_hm = (ds_sim_hm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
date_sim_hm = num2date(days_sim_hm, units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))

# load observations (measured data)
path_obs = base_path.parent / "observations" / "rietholzbach_lysimeter.nc"
ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")
days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
ds_obs = ds_obs.assign_coords(Time=("Time", date_obs))

# load transport simulation which contains modified isotope input signal
# (could be any transport model) since each transport model uses the same
# approach for mixing isotopes while snowfall/snow melt
states_tm_file = base_path / "d18O_in_for_hydrus.nc"
ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))

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
df_prec.loc[:, 'Prec'] = onp.where(ds_sim_hm['ta'].isel(x=0, y=0).values > 0, ds_sim_hm['prec'].isel(x=0, y=0).values - ds_sim_hm['int_ground'].isel(x=0, y=0).values, 0) + ds_sim_hm['q_snow'].isel(x=0, y=0).values
df_ttop = pd.DataFrame(index=date_obs, columns=['tTop'])
df_ttop.loc[:, 'tTop'] = ds_obs['TA'].isel(x=0, y=0).values
df_hcrita = pd.DataFrame(index=date_obs, columns=['hCritA'])
df_hcrita.loc[:, 'hCritA'] = 100000
df_ampl = pd.DataFrame(index=date_obs, columns=['Ampl'])
df_ampl.loc[:, 'Ampl'] = 0
df_tatm = pd.DataFrame(index=date_obs, columns=['tAtm'])
df_tatm.loc[:, 'tAtm'] = range(1, len(df_tatm.index)+1)
df_ctop = pd.DataFrame(index=date_sim_tm, columns=['cTop'])
df_ctop.loc[:, 'cTop'] = ds_sim_tm['C_iso_in'].isel(x=0, y=0).values

df = pd.DataFrame(index=date_obs)
df = df.join([df_tatm, df_prec, df_rsoil, df_rroot, df_hcrita, df_ttop, df_ampl, df_ctop])
df.columns = [['', '[mm/day]', '[-]', '[-]', '[cm]', '[degC]', '[K]', '[per mil]'],
              ['tAtm', 'Prec', 'rSoil', 'rRoot', 'hCritA', 'tTop', 'Ampl', 'cTop']]

file = base_path / "results" / "atmosphere_daily_18O.csv"
df.to_csv(file, header=True, index=False, sep=";")

onp.nansum(ds_obs['PREC'].isel(x=0, y=0).values)
onp.sum(ds_sim_hm['prec'].isel(x=0, y=0).values)
onp.sum(onp.where(ds_sim_hm['ta'].isel(x=0, y=0).values > 0, ds_sim_hm['prec'].isel(x=0, y=0).values, 0))
onp.sum(ds_sim_hm['q_snow'].isel(x=0, y=0).values)
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
fig.tight_layout()

# write HYDRUS-1D input for virtual bromide experiments
years = onp.arange(1997, 2007).tolist()
for year in years:
    ds_obs_year = ds_obs.sel(Time=slice(f'{year}-01-01', f'{year + 1}-12-31'))
    date_obs_year = ds_obs_year['Time'].values
    ds_sim_hm_year = ds_sim_hm.sel(Time=slice(f'{year}-01-01', f'{year + 1}-12-31'))
    date_sim_hm_year = ds_sim_hm_year['Time'].values

    # monthly variable soil cover fraction of grass
    df_scf_year = pd.DataFrame(index=date_obs_year, columns=['scf'])
    df_scf_year.loc[df_scf_year.index.month==1] = 1. - onp.exp(-0.463*1)
    df_scf_year.loc[df_scf_year.index.month==2] = 1. - onp.exp(-0.463*1)
    df_scf_year.loc[df_scf_year.index.month==3] = 1. - onp.exp(-0.463*1.33)
    df_scf_year.loc[df_scf_year.index.month==4] = 1. - onp.exp(-0.463*1.66)
    df_scf_year.loc[df_scf_year.index.month==5] = 1. - onp.exp(-0.463*2)
    df_scf_year.loc[df_scf_year.index.month==6] = 1. - onp.exp(-0.463*2)
    df_scf_year.loc[df_scf_year.index.month==7] = 1. - onp.exp(-0.463*2)
    df_scf_year.loc[df_scf_year.index.month==8] = 1. - onp.exp(-0.463*2)
    df_scf_year.loc[df_scf_year.index.month==9] = 1. - onp.exp(-0.463*2)
    df_scf_year.loc[df_scf_year.index.month==10] = 1. - onp.exp(-0.463*1.33)
    df_scf_year.loc[df_scf_year.index.month==11] = 1. - onp.exp(-0.463*1.16)
    df_scf_year.loc[df_scf_year.index.month==12] = 1. - onp.exp(-0.463*1)

    df_rsoil = pd.DataFrame(index=date_obs_year, columns=['rSoil'])
    df_rsoil.loc[:, 'rSoil'] = ds_obs_year['PET'].isel(x=0, y=0).values * (1 - df_scf_year['scf'].values)
    df_rsoil.iloc[0, 0] = 0.05
    df_rroot = pd.DataFrame(index=date_obs_year, columns=['rRoot'])
    df_rroot.loc[:, 'rRoot'] = ds_obs_year['PET'].isel(x=0, y=0).values * df_scf_year['scf'].values
    df_rroot.iloc[0, 0] = 0.03
    df_prec = pd.DataFrame(index=date_sim_hm_year, columns=['Prec'])
    df_prec.loc[:, 'Prec'] = onp.where(ds_sim_hm_year['ta'].isel(x=0, y=0).values > 0, ds_sim_hm_year['prec'].isel(x=0, y=0).values - ds_sim_hm['int_ground'].isel(x=0, y=0).values, 0) + ds_sim_hm_year['q_snow'].isel(x=0, y=0).values
    df_ttop = pd.DataFrame(index=date_obs_year, columns=['tTop'])
    df_ttop.loc[:, 'tTop'] = ds_obs_year['TA'].isel(x=0, y=0).values
    df_hcrita = pd.DataFrame(index=date_obs_year, columns=['hCritA'])
    df_hcrita.loc[:, 'hCritA'] = 100000
    df_ampl = pd.DataFrame(index=date_obs_year, columns=['Ampl'])
    df_ampl.loc[:, 'Ampl'] = 0
    df_tatm = pd.DataFrame(index=date_obs_year, columns=['tAtm'])
    df_tatm.loc[:, 'tAtm'] = range(1, len(df_tatm.index)+1)
    df_ctop = pd.DataFrame(index=df_scf_year.index, columns=['cTop'])
    injection_date = f'{year}-11-12'
    # set new injection dates within 20 mm of cumulated rainfall
    cond = (df_prec.loc[injection_date:, 'Prec'].values.cumsum() <= 20)
    injection_dates_new = df_prec.loc[injection_date:, ].index[cond]
    if df_prec.loc[injection_dates_new, 'Prec'].sum() > 0:
        df_ctop.loc[injection_dates_new, 'cTop'] = (79.9/3.14)/df_prec.loc[injection_dates_new, 'Prec']  # bromide mass in g per m2
    else:
        injection_dates_new = injection_dates_new[-1] + timedelta(days=1)
        df_ctop.loc[injection_dates_new, 'cTop'] = (79.9/3.14)/df_prec.loc[injection_dates_new, 'Prec']
    df_ctop.replace([onp.inf, -onp.inf, onp.nan], 0, inplace=True)

    df_year = pd.DataFrame(index=date_obs_year)
    df_year = df_year.join([df_tatm, df_prec, df_rsoil, df_rroot, df_hcrita, df_ttop, df_ampl, df_ctop])
    df_year.columns = [['', '[mm/day]', '[-]', '[-]', '[cm]', '[degC]', '[K]', '[g/l]'],
                       ['tAtm', 'Prec', 'rSoil', 'rRoot', 'hCritA', 'tTop', 'Ampl', 'cTop']]

    file = base_path / "results" / f"atmosphere_daily_bromide_{year}.csv"
    df_year.to_csv(file, header=True, index=False, sep=";")
