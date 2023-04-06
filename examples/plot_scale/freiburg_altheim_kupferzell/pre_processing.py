import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pathlib import Path
import os
from cftime import date2num
import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.titlesize'] = 7
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['legend.title_fontsize'] = 7
sns.set_style("ticks")
sns.plotting_context("paper", font_scale=1, rc={'font.size': 6.0,
                                                'axes.labelsize': 7.0,
                                                'axes.titlesize': 8.0,
                                                'xtick.labelsize': 6.0,
                                                'ytick.labelsize': 6.0,
                                                'legend.fontsize': 6.0,
                                                'legend.title_fontsize': 7.0})

base_path = Path(__file__).parent
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

periods = ['hist', 'future']
station_ids = [1443, 2787, 4189]
cms = ['CCCma-CanESM2_CCLM4-8-17', 'ICHEC-EC-EARTH_CCLM4-8-17', 
       'ICHEC-EC-EARTH_RCA4', 'IPSL-IPSL-CM5A-MR_RCA4',
       'MIROC-MIROC5_CCLM4-8-17', 'MPI-M-MPI-ESM-LR_CCLM4-8-17', 
       'MPI-M-MPI-ESM-LR_RCA4']
station_label = {1443: 'Freiburg',
                2787: 'Kupferzell',
                4189: 'Altheim',

}
station_label1 = {1443: 'freiburg',
                  2787: 'kupferzell',
                  4189: 'altheim',
}

# --- time index to join data --------------------------------------------------
idx_daily_1985_2100 = pd.date_range(start='1985-01-01',
                                    end='2100-12-31', freq='d')
idx_daily_2003_2005 = pd.date_range(start='2003-01-01',
                                    end='2005-12-31', freq='d')
idx_daily_2016_2021 = pd.date_range(start='2016-01-01',
                                    end='2021-12-31', freq='d')
idx_daily_1994_2005 = pd.date_range(start='1994-01-01',
                                    end='2005-12-31', freq='d')
idx_daily_1985_2005 = pd.date_range(start='1985-01-01',
                                    end='2005-12-31', freq='d')
idx_daily_2040_2060 = pd.date_range(start='2040-01-01',
                                    end='2060-12-31', freq='d')
idx_daily_2080_2100 = pd.date_range(start='2080-01-01',
                                    end='2100-12-31', freq='d')

idx_hourly_1985_2100 = pd.date_range(start='1985-01-01 00:00:00',
                                    end='2100-12-31 23:00:00', freq='h')
idx_hourly_2003_2005 = pd.date_range(start='2003-01-01 00:00:00',
                                    end='2005-12-31 23:00:00', freq='h')
idx_hourly_2016_2021 = pd.date_range(start='2016-01-01 00:00:00',
                                    end='2021-12-31 23:00:00', freq='h')
idx_hourly_1994_2005 = pd.date_range(start='1994-01-01 00:00:00',
                                    end='2005-12-31 23:00:00', freq='h')
idx_hourly_1985_2005 = pd.date_range(start='1985-01-01 00:00:00',
                                    end='2005-12-31 23:00:00', freq='h')
idx_hourly_2040_2060 = pd.date_range(start='2040-01-01 00:00:00',
                                    end='2060-12-31 23:00:00', freq='h')
idx_hourly_2080_2100 = pd.date_range(start='2080-01-01 00:00:00',
                                    end='2100-12-31 23:00:00', freq='h')

idx_3hourly_1985_2100c = pd.date_range(start='1985-01-01 01:30:00',
                                      end='2100-12-31 22:30:00', freq='3h')
idx_3hourly_1985_2100 = pd.date_range(start='1985-01-01 00:00:00',
                                      end='2100-12-31 23:00:00', freq='3h')

idx_10mins_1985_2100 = pd.date_range(start='1985-01-01 00:00:00',
                                     end='2100-12-31 23:50:00', freq='10T')
idx_10mins_2003_2005 = pd.date_range(start='2003-01-01 00:00:00',
                                    end='2005-12-31 23:50:00', freq='10T')
idx_10mins_2016_2021 = pd.date_range(start='2016-01-01 00:00:00',
                                    end='2021-12-31 23:50:00', freq='10T')
idx_10mins_1994_2005 = pd.date_range(start='1994-01-01 00:00:00',
                                    end='2005-12-31 23:50:00', freq='10T')
idx_10mins_1985_2005 = pd.date_range(start='1985-01-01 00:00:00',
                                    end='2005-12-31 23:50:00', freq='10T')
idx_10mins_2040_2060 = pd.date_range(start='2040-01-01 00:00:00',
                                    end='2060-12-31 23:50:00', freq='10T')
idx_10mins_2080_2100 = pd.date_range(start='2080-01-01 00:00:00',
                                    end='2100-12-31 23:50:00', freq='10T')

# --- load climate projections ---------------------------------------------------
dict_meteo_daily = {}
for station_id in station_ids:
    dict_meteo_daily[station_id] = {}
    for cm in cms:
        dict_meteo_daily[station_id][cm] = {}
        for period in periods:
            file = base_path / 'climate_projections' / 'data' / 'daily' / f'BC-Neu-pcmgthr-MBCn_{cm}_DWD_{station_id}_{period}.csv'
            data = pd.read_csv(file, sep=',', index_col=0)
            data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
            data.columns = ['PREC', 'TA', 'TAD', 'RS']
            if period == 'hist':
                data_1985_2005 = pd.DataFrame(index=idx_daily_1985_2005)
                data_1985_2005 = data_1985_2005.join(data)
                # fill NaNs at 29th February  
                data_1985_2005.loc[:, 'TA'] = data_1985_2005['TA'].interpolate()
                data_1985_2005.loc[:, 'RS'] = data_1985_2005['RS'].interpolate()
                dict_meteo_daily[station_id][cm]['1985-2005'] = data_1985_2005
            elif period == 'future':
                data_2016_2021 = pd.DataFrame(index=idx_daily_2016_2021)
                data_2016_2021 = data_2016_2021.join(data)
                # fill NaNs at 29th February  
                data_2016_2021.loc[:, 'TA'] = data_2016_2021['TA'].interpolate()
                data_2016_2021.loc[:, 'RS'] = data_2016_2021['RS'].interpolate()
                data_2040_2060 = pd.DataFrame(index=idx_daily_2040_2060)
                data_2040_2060 = data_2040_2060.join(data)
                # fill NaNs at 29th February 
                data_2040_2060.loc[:, 'TA'] = data_2040_2060['TA'].interpolate()
                data_2040_2060.loc[:, 'RS'] = data_2040_2060['RS'].interpolate()
                data_2080_2100 = pd.DataFrame(index=idx_daily_2080_2100)
                data_2080_2100 = data_2080_2100.join(data)
                # fill NaNs at 29th February 
                data_2080_2100.loc[:, 'TA'] = data_2080_2100['TA'].interpolate()
                data_2080_2100.loc[:, 'RS'] = data_2080_2100['RS'].interpolate()
                dict_meteo_daily[station_id][cm]['2016-2021'] = data_2016_2021
                dict_meteo_daily[station_id][cm]['2040-2060'] = data_2040_2060
                dict_meteo_daily[station_id][cm]['2080-2100'] = data_2080_2100

    dict_meteo_daily[station_id]['observed'] = {}
    file = base_path / 'input' / f'{station_label1[station_id]}' / 'observed' / '2016-2021' / 'PREC.txt'
    data_prec = pd.read_csv(file, sep='\t')
    data_prec.index = idx_10mins_2016_2021
    file = base_path / 'input' / f'{station_label1[station_id]}' / 'observed' / '2016-2021' / 'TA.txt'
    data_ta = pd.read_csv(file, sep='\t')
    file = base_path / 'input' / f'{station_label1[station_id]}' / 'observed' / '2016-2021' / 'PET.txt'
    data_pet = pd.read_csv(file, sep='\t')
    data_2016_2021 = pd.DataFrame(index=idx_daily_2016_2021, columns=['PREC', 'TA', 'PET'])
    data_2016_2021.loc[:, 'PREC'] = data_prec.loc[:, 'PREC'].resample('1D').sum().values
    data_2016_2021.loc[:, 'TA'] = data_ta.loc[:, 'TA'].values
    data_2016_2021.loc[:, 'PET'] = data_pet.loc[:, 'PET'].values
    dict_meteo_daily[station_id]['observed']['2016-2021'] = data_2016_2021

# --- projected annual air temperature and precipitation --------------------------------
dict_meteo_ann = {}
for station_id in station_ids:
    dict_meteo_ann[station_id] = {}
    for cm in cms:
        dict_meteo_ann[station_id][cm] = {}
        for period in periods:
            file = file = base_path / 'climate_projections' / 'data' / 'daily' / f'BC-Neu-pcmgthr-MBCn_{cm}_DWD_{station_id}_{period}.csv'
            data = pd.read_csv(file, sep=',', index_col=0)
            data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
            data.columns = ['PREC', 'TA', 'TAD', 'RS']
            data_ann = data.loc[:, 'PREC'].resample('1Y').sum().to_frame().join(data.loc[:, 'TA'].resample('1Y').mean().to_frame())
            dict_meteo_ann[station_id][cm][period] = data_ann
            
# --- observed annual air temperature and precipitation --------------------------------
for station_id in station_ids:
    for cm in cms:
        file = file = base_path / 'climate_projections' / 'data' / 'daily' / f'BC-Neu-pcmgthr-MBCn_{cm}_DWD_{station_id}_future.csv'
        data = pd.read_csv(file, sep=',', index_col=0)
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
        data.columns = ['PREC', 'TA', 'TAD', 'RS']
        data_2016_2021 = pd.DataFrame(index=idx_daily_2016_2021)
        data_2016_2021 = data_2016_2021.join(data)
        data_ann = data_2016_2021.loc[:, 'PREC'].resample('1Y').sum().to_frame().join(data_2016_2021.loc[:, 'TA'].resample('1Y').mean().to_frame())
        dict_meteo_ann[station_id][cm]['2016-2021'] = data_ann
    data_2016_2021 = dict_meteo_daily[station_id]['observed']['2016-2021']
    data_ann = data_2016_2021.loc[:, 'PREC'].resample('1Y').sum().to_frame().join(data_2016_2021.loc[:, 'TA'].resample('1Y').mean().to_frame())
    dict_meteo_ann[station_id]['observed'] = {}
    dict_meteo_ann[station_id]['observed']['2016-2021'] = data_ann

# --- compare projected annual air temperature and precipitation ------------------------
# color = {'CCCma-CanESM2_CCLM4-8-17_hist': '#eff3ff',
#          'ICHEC-EC-EARTH_CCLM4-8-17_hist': '#c6dbef', 
#          'ICHEC-EC-EARTH_RCA4_hist': '#9ecae1', 
#          'IPSL-IPSL-CM5A-MR_RCA4_hist': '#6baed6',
#          'MIROC-MIROC5_CCLM4-8-17_hist': '#4292c6', 
#          'MPI-M-MPI-ESM-LR_CCLM4-8-17_hist': '#2171b5', 
#          'MPI-M-MPI-ESM-LR_RCA4_hist': '#084594',
#          'CCCma-CanESM2_CCLM4-8-17_future': '#fee5d9',
#          'ICHEC-EC-EARTH_CCLM4-8-17_future': '#fcbba1', 
#          'ICHEC-EC-EARTH_RCA4_future': '#fc9272', 
#          'IPSL-IPSL-CM5A-MR_RCA4_future': '#fb6a4a',
#          'MIROC-MIROC5_CCLM4-8-17_future': '#ef3b2c', 
#          'MPI-M-MPI-ESM-LR_CCLM4-8-17_future': '#cb181d', 
#          'MPI-M-MPI-ESM-LR_RCA4_future': '#99000d'}

# label = {'CCCma-CanESM2_CCLM4-8-17': 'CCCma-CanESM2 CCLM4.8.17',
#          'ICHEC-EC-EARTH_CCLM4-8-17': 'ICHEC-EC-EARTH CCLM4.8.17', 
#          'ICHEC-EC-EARTH_RCA4': 'ICHEC-EC-EARTH RCA4', 
#          'IPSL-IPSL-CM5A-MR_RCA4': 'IPSL-IPSL-CM5A-MR RCA4',
#          'MIROC-MIROC5_CCLM4-8-17': 'MIROC-MIROC5 CCLM4.8.17', 
#          'MPI-M-MPI-ESM-LR_CCLM4-8-17': 'MPI-M-MPI-ESM-LR CCLM4.8.17', 
#          'MPI-M-MPI-ESM-LR_RCA4': 'MPI-M-MPI-ESM-LR RCA4'}

# for station_id in station_ids:
#     fig, axs = plt.subplots(1, 1, figsize=(5, 3))
#     for period in periods:
#         for cm in cms:
#             data = dict_meteo_ann[station_id][cm][period]
#             prec_avg = data.loc[:, 'PREC'].mean()
#             prec_std = data.loc[:, 'PREC'].std()
#             ta_avg = data.loc[:, 'TA'].mean()
#             ta_std = data.loc[:, 'TA'].std()
#             axs.errorbar(ta_avg, prec_avg, xerr=ta_std, yerr=prec_std, fmt='o', label=label[cm], color=color[f'{cm}_{period}'], ms=2.5)
#     axs.set_ylabel('Mean annual precipitation [mm]')
#     axs.set_xlabel('Mean annual air temperature [°C]')
#     axs.set_title(f'{station_label[station_id]} (station ID: {station_id})')
#     lines, labels = axs.get_legend_handles_labels()
#     fig.legend(lines[:7], labels[:7], loc='upper right', fontsize=6, frameon=False, bbox_to_anchor=(1.01, 0.9), title='1985-2005')
#     fig.legend(lines[7:], labels[7:], loc='upper right', fontsize=6, frameon=False, bbox_to_anchor=(1.01, 0.55), title='2016-2100')
#     fig.subplots_adjust(right=0.68)
#     file = base_path_figs / f'projected_annual_prec_and_ta_{station_label[station_id]}.png'
#     fig.savefig(file, dpi=250)
#     plt.close(fig=fig)

# for station_id in station_ids:
#     fig, axs = plt.subplots(1, 1, figsize=(5, 3))
#     data = dict_meteo_ann[station_id]['observed']['2016-2021']
#     prec_avg = data.loc[:, 'PREC'].mean()
#     ta_avg = data.loc[:, 'TA'].mean()
#     axs.scatter(ta_avg, prec_avg, label='observed', color='blue', s=5)
#     for cm in cms:
#         data = dict_meteo_ann[station_id][cm]['2016-2021']
#         prec_avg = data.loc[:, 'PREC'].mean()
#         ta_avg = data.loc[:, 'TA'].mean()
#         axs.scatter(ta_avg, prec_avg, label=label[cm], color=color[f'{cm}_future'], s=4)
#     axs.set_ylabel('Mean annual precipitation [mm]')
#     axs.set_xlabel('Mean annual air temperature [°C]')
#     axs.set_title(f'{station_label[station_id]} (station ID: {station_id})')
#     lines, labels = axs.get_legend_handles_labels()
#     fig.legend(lines, labels, loc='upper right', fontsize=6, frameon=False, bbox_to_anchor=(1.01, 0.9), title='2016-2021')
#     fig.subplots_adjust(right=0.68)
#     file = base_path_figs / f'annual_prec_and_ta_{station_label[station_id]}_obs_and_proj.png'
#     fig.savefig(file, dpi=250)
#     plt.close(fig=fig)

# --- add minimum TA and maximum TA --------------------------------
stations = ['freiburg', 'weingarten', 'ingelfingen']
for station, station_id in zip(stations, station_ids):
    file = base_path / 'dwd' / f'{station}' / 'meteo.txt'
    data = pd.read_csv(file, sep=';', na_values=-999)
    data.index = pd.to_datetime(data['MESS_DATUM'], format='%Y%m%d')
    data_2016_2021 = pd.DataFrame(index=idx_daily_2016_2021)
    data_2016_2021 = data_2016_2021.join(data)
    df_TA_min_max = pd.DataFrame(index=idx_daily_2016_2021)
    if station == 'freiburg':
        df_TA_min_max.loc[:, 'TA_min'] = data_2016_2021[' TNK'].values
        df_TA_min_max.loc[:, 'TA_max'] = data_2016_2021[' TXK'].values
    # correct for altitude effect since values are obtained from nearby station
    elif station == 'weingarten':
        df_TA_min_max.loc[:, 'TA_min'] = data_2016_2021[' TNK'].values + ((440 - 340) / 100) * 0.65
        df_TA_min_max.loc[:, 'TA_max'] = data_2016_2021[' TXK'].values + ((440 - 340) / 100) * 0.65
    elif station == 'ingelfingen':
        df_TA_min_max.loc[:, 'TA_min'] = data_2016_2021[' TNK'].values + ((385 - 541) / 100) * 0.65
        df_TA_min_max.loc[:, 'TA_max'] = data_2016_2021[' TXK'].values + ((385 - 541) / 100) * 0.65
    meteo_2016_2021 = dict_meteo_daily[station_id]['observed']['2016-2021']
    meteo_2016_2021 = meteo_2016_2021.join(df_TA_min_max)
    dict_meteo_daily[station_id]['observed']['2016-2021'] = meteo_2016_2021

for station_id in station_ids:
    for cm in ['MPI-M-MPI-ESM-LR_RCA4', 'CCCma-CanESM2_CCLM4-8-17']:
        file = base_path / 'climate_projections' / 'data' / 'daily' / f'tmin-max-daily_FullTs_{cm}_station-DWD_{station_id}.csv'
        data = pd.read_csv(file, sep=',', index_col=2)
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
        data = data.loc['1985':'2100', :]
        cond = (data['Var'].values == 'tasmax')
        data_ta_max = data.loc[cond, 'Center']
        cond = (data['Var'].values == 'tasmin')
        data_ta_min = data.loc[cond, 'Center']
        data1 = pd.DataFrame(index=data_ta_min.index, columns=['TA_min', 'TA_max'])
        data1.loc[:, 'TA_min'] = data_ta_min.values
        data1.loc[:, 'TA_max'] = data_ta_max.values
        data = pd.DataFrame(index=idx_daily_1985_2100)
        data = data.join(data1)

        data_1985_2005 = dict_meteo_daily[station_id][cm]['1985-2005']
        data_1985_2005 = data_1985_2005.join(data)
        # fill NaNs at 29th February  
        data_1985_2005.loc[:, 'TA_min'] = data_1985_2005['TA_min'].interpolate()
        data_1985_2005.loc[:, 'TA_max'] = data_1985_2005['TA_max'].interpolate()
        dict_meteo_daily[station_id][cm]['1985-2005'] = data_1985_2005
        data_2016_2021 = dict_meteo_daily[station_id][cm]['2016-2021']
        data_2016_2021 = data_2016_2021.join(data)
        # fill NaNs at 29th February  
        data_2016_2021.loc[:, 'TA_min'] = data_2016_2021['TA_min'].interpolate()
        data_2016_2021.loc[:, 'TA_max'] = data_2016_2021['TA_max'].interpolate()
        data_2040_2060 = dict_meteo_daily[station_id][cm]['2040-2060']
        data_2040_2060 = data_2040_2060.join(data)
        # fill NaNs at 29th February  
        data_2040_2060.loc[:, 'TA_min'] = data_2040_2060['TA_min'].interpolate()
        data_2040_2060.loc[:, 'TA_max'] = data_2040_2060['TA_max'].interpolate()
        data_2080_2100 = dict_meteo_daily[station_id][cm]['2080-2100']
        data_2080_2100 = data_2080_2100.join(data)
        # fill NaNs at 29th February  
        data_2080_2100.loc[:, 'TA_min'] = data_2080_2100['TA_min'].interpolate()
        data_2080_2100.loc[:, 'TA_max'] = data_2080_2100['TA_max'].interpolate()
        dict_meteo_daily[station_id][cm]['2016-2021'] = data_2016_2021
        dict_meteo_daily[station_id][cm]['2040-2060'] = data_2040_2060
        dict_meteo_daily[station_id][cm]['2080-2100'] = data_2080_2100

# --- downscale precipitation to 10 minutes ------------------------
dict_precip_10mins = {}
cm = 'MPI-M-MPI-ESM-LR_RCA4'
for station_id in station_ids:
    dict_precip_10mins[station_id] = {}
    dict_precip_10mins[station_id][cm] = {}
    file = base_path / 'climate_projections' / 'data' / 'subdaily' / f'pr_subdaily_FullTs_{cm}_station-DWD_{station_id}.csv'
    data = pd.read_csv(file, sep=',', index_col=0)
    file = base_path / 'climate_projections' / 'data' / 'subdaily' / 'datetime_MPI-RCA1hr.txt'
    data_idx = pd.read_csv(file, sep=';')
    data.index = pd.to_datetime(data_idx.iloc[:, 0].astype(str).values, format='%Y-%m-%d %H:%M')
    data = data.loc['1985':'2100', :]
    data_hourly = pd.DataFrame(index=idx_hourly_1985_2100)
    data_hourly.loc[:, 'PREC'] = data.loc[:, 'Center'].values
    data_10mins = pd.DataFrame(index=idx_10mins_1985_2100)
    # donwnscale hourly precipitation by linear interpolation
    data_10mins = data_10mins.join(data_hourly)
    data_10mins = data_10mins.ffill() / 6
    # replace numerical artefacts
    cond0 = (data_10mins['PREC'] < 0.001)
    data_10mins.loc[cond0, 'PREC'] = 0

    data_1985_2005 = pd.DataFrame(index=idx_10mins_1985_2005)
    data_1985_2005 = data_1985_2005.join(data_10mins)
    dict_precip_10mins[station_id][cm]['1985-2005'] = data_1985_2005
    data_2016_2021 = pd.DataFrame(index=idx_10mins_2016_2021)
    data_2016_2021 = data_2016_2021.join(data_10mins)
    data_2040_2060 = pd.DataFrame(index=idx_10mins_2040_2060)
    data_2040_2060 = data_2040_2060.join(data_10mins)
    data_2080_2100 = pd.DataFrame(index=idx_10mins_2080_2100)
    data_2080_2100 = data_2080_2100.join(data_10mins)
    dict_precip_10mins[station_id][cm]['2016-2021'] = data_2016_2021
    dict_precip_10mins[station_id][cm]['2040-2060'] = data_2040_2060
    dict_precip_10mins[station_id][cm]['2080-2100'] = data_2080_2100

cm = 'CCCma-CanESM2_CCLM4-8-17'
for station_id in station_ids:
    dict_precip_10mins[station_id][cm] = {}
    file = base_path / 'climate_projections' / 'data' / 'subdaily' / f'pr_subdaily_FullTs_{cm}_station-DWD_{station_id}.csv'
    data = pd.read_csv(file, sep=',', index_col=0)
    file = base_path / 'climate_projections' / 'data' / 'subdaily' / 'datetime_CANESM-CLM3hr.txt'
    data_idx = pd.read_csv(file, sep=';')
    data.index = pd.to_datetime(data_idx.iloc[:, 0].astype(str).values, format='%Y-%m-%d %H:%M')
    data = data.loc['1985':'2100', :]
    data_3hourly = pd.DataFrame(index=idx_3hourly_1985_2100c)
    data_3hourly = data_3hourly.join(data.loc[:, 'Center'].to_frame())
    data_3hourly.index = idx_3hourly_1985_2100
    data_3hourly.columns = ['PREC']
    # fill 29th February in leap years
    data_3hourly = data_3hourly.fillna(0)

    # donwnscale 3-hourly precipitation by linear interpolation
    data_hourly = pd.DataFrame(index=idx_hourly_1985_2100)
    data_hourly = data_hourly.join(data_3hourly)
    data_hourly = data_hourly.ffill() / 3
    data_10mins = pd.DataFrame(index=idx_10mins_1985_2100)
    data_10mins = data_10mins.join(data_hourly)
    data_10mins = data_10mins.ffill() / 6
    # replace numerical artefacts
    cond0 = (data_10mins['PREC'] < 0.001)
    data_10mins.loc[cond0, 'PREC'] = 0

    data_1985_2005 = pd.DataFrame(index=idx_10mins_1985_2005)
    data_1985_2005 = data_1985_2005.join(data_10mins)
    dict_precip_10mins[station_id][cm]['1985-2005'] = data_1985_2005
    data_2016_2021 = pd.DataFrame(index=idx_10mins_2016_2021)
    data_2016_2021 = data_2016_2021.join(data_10mins)
    data_2040_2060 = pd.DataFrame(index=idx_10mins_2040_2060)
    data_2040_2060 = data_2040_2060.join(data_10mins)
    data_2080_2100 = pd.DataFrame(index=idx_10mins_2080_2100)
    data_2080_2100 = data_2080_2100.join(data_10mins)
    dict_precip_10mins[station_id][cm]['2016-2021'] = data_2016_2021
    dict_precip_10mins[station_id][cm]['2040-2060'] = data_2040_2060
    dict_precip_10mins[station_id][cm]['2080-2100'] = data_2080_2100

# --- write input data to .txt -------------------------------------
stations = ['freiburg', 'kupferzell', 'altheim']
for station, station_id in zip(stations, station_ids):
    for cm in ['MPI-M-MPI-ESM-LR_RCA4', 'CCCma-CanESM2_CCLM4-8-17']:
        for period in ['1985-2005', '2016-2021', '2040-2060', '2080-2100']:
            data_precip = dict_precip_10mins[station_id][cm][period]
            data_meteo = dict_meteo_daily[station_id][cm][period]
            data_ta = data_meteo.loc[:, ['TA', 'TA_min', 'TA_max']]
            data_rs = data_meteo.loc[:, ['RS']]

            path_dir = base_path / "input" / station / cm / period
            if not os.path.exists(path_dir):
                os.mkdir(path_dir)

            idx_10mins = pd.date_range(start=str(data_precip.index[0]),
                                       end=str(data_precip.index[-1]), freq='10T')
            idx_daily = pd.date_range(start=str(data_ta.index[0]),
                                       end=str(data_ta.index[-1]), freq='d')
            df_PREC = pd.DataFrame(index=idx_10mins, columns=['YYYY', 'MM', 'DD', 'hh', 'mm', 'PREC'])
            df_PREC['YYYY'] = data_precip.index.year.values
            df_PREC['MM'] = data_precip.index.month.values
            df_PREC['DD'] = data_precip.index.day.values
            df_PREC['hh'] = data_precip.index.hour.values
            df_PREC['mm'] = data_precip.index.minute.values
            df_PREC['PREC'] = data_precip['PREC'].values
            path_txt = path_dir / "PREC.txt"
            df_PREC.to_csv(path_txt, header=True, index=False, sep="\t")
            nas = np.sum(np.isnan(data_precip['PREC'].values))
            print(f'{station}-{cm}-{period}-PREC: {nas}')

            df_TA = pd.DataFrame(index=idx_daily, columns=['YYYY', 'MM', 'DD', 'hh', 'mm'])
            df_TA['YYYY'] = data_ta.index.year.values
            df_TA['MM'] = data_ta.index.month.values
            df_TA['DD'] = data_ta.index.day.values
            df_TA['hh'] = data_ta.index.hour.values
            df_TA['mm'] = data_ta.index.minute.values
            df_TA['TA'] = data_ta['TA'].values
            df_TA['TA_min'] = data_ta['TA_min'].values
            df_TA['TA_max'] = data_ta['TA_max'].values
            path_txt = path_dir / "TA.txt"
            df_TA.to_csv(path_txt, header=True, index=False, sep="\t")
            nas = np.sum(np.isnan(data_ta['TA'].values))
            print(f'{station}-{cm}-{period}-TA: {nas} NaN values')
            nas = np.sum(np.isnan(data_ta['TA_min'].values))
            print(f'{station}-{cm}-{period}-TA_min: {nas} NaN values')
            nas = np.sum(np.isnan(data_ta['TA_max'].values))
            print(f'{station}-{cm}-{period}-TA_max: {nas} NaN values')

            df_RS = pd.DataFrame(index=idx_daily, columns=['YYYY', 'MM', 'DD', 'hh', 'mm'])
            df_RS['YYYY'] = data_rs.index.year.values
            df_RS['MM'] = data_rs.index.month.values
            df_RS['DD'] = data_rs.index.day.values
            df_RS['hh'] = data_rs.index.hour.values
            df_RS['mm'] = data_rs.index.minute.values
            df_RS['RS'] = data_rs['RS'].values * 0.0864  # convert watt (i.e. J/s) to MJ/day
            path_txt = path_dir / "RS.txt"
            df_RS.to_csv(path_txt, header=True, index=False, sep="\t")
            nas = np.sum(np.isnan(data_rs['RS'].values))
            print(f'{station}-{cm}-{period}-RS: {nas} NaN values')

stations = ['freiburg', 'kupferzell', 'altheim']
for station, station_id in zip(stations, station_ids):
    data_meteo = dict_meteo_daily[station_id]['observed']['2016-2021']
    data_ta = data_meteo.loc[:, ['TA', 'TA_min', 'TA_max']]

    path_dir = base_path / "input" / station / 'observed' / '2016-2021'
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)

    idx_daily = pd.date_range(start=str(data_ta.index[0]),
                                end=str(data_ta.index[-1]), freq='d')

    df_TA = pd.DataFrame(index=idx_daily, columns=['YYYY', 'MM', 'DD', 'hh', 'mm'])
    df_TA['YYYY'] = data_ta.index.year.values
    df_TA['MM'] = data_ta.index.month.values
    df_TA['DD'] = data_ta.index.day.values
    df_TA['hh'] = data_ta.index.hour.values
    df_TA['mm'] = data_ta.index.minute.values
    df_TA['TA'] = data_ta['TA'].values
    df_TA['TA_min'] = data_ta['TA_min'].values
    df_TA['TA_max'] = data_ta['TA_max'].values
    path_txt = path_dir / "TA.txt"
    df_TA.to_csv(path_txt, header=True, index=False, sep="\t")
    nas = np.sum(np.isnan(data_ta['TA'].values))
    print(f'{station}-observed-{period}-TA: {nas} NaN values')
    nas = np.sum(np.isnan(data_ta['TA_min'].values))
    print(f'{station}-observed-{period}-TA_min: {nas} NaN values')
    nas = np.sum(np.isnan(data_ta['TA_max'].values))
    print(f'{station}-observed-{period}-TA_max: {nas} NaN values')

    data_precip = data_meteo.loc[:, 'PREC'].to_frame()
    nas = np.sum(np.isnan(data_precip['PREC'].values))
    print(f'{station}-observed-{period}-PREC: {nas} NaN values')

    data_pet = data_meteo.loc[:, 'PET'].to_frame()
    nas = np.sum(np.isnan(data_pet['PET'].values))
    print(f'{station}-observed-{period}-PET: {nas} NaN values')