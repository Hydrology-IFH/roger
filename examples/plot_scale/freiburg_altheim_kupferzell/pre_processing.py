from pathlib import Path
import os
import glob
import h5netcdf
import datetime
from cftime import date2num
import pandas as pd
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

# --- time index to join data --------------------------------------------------
idx_daily_2003_2005 = pd.date_range(start='2003-01-01',
                                    end='2005-12-31', freq='d')
idx_daily_2015_2021 = pd.date_range(start='2015-01-01',
                                    end='2021-12-31', freq='d')
idx_daily_1994_2005 = pd.date_range(start='1994-01-01',
                                    end='2005-12-31', freq='d')
idx_daily_1985_2005 = pd.date_range(start='1985-01-01',
                                    end='2005-12-31', freq='d')
idx_daily_2040_2060 = pd.date_range(start='2040-01-01',
                                    end='2060-12-31', freq='d')
idx_daily_2080_2100 = pd.date_range(start='2080-01-01',
                                    end='2100-12-31', freq='d')

idx_hourly_2003_2005 = pd.date_range(start='2003-01-01 00:00:00',
                                    end='2005-12-31 23:00:00', freq='h')
idx_hourly_2015_2021 = pd.date_range(start='2015-01-01 00:00:00',
                                    end='2021-12-31 23:00:00', freq='h')
idx_hourly_1994_2005 = pd.date_range(start='1994-01-01 00:00:00',
                                    end='2005-12-31 23:00:00', freq='h')
idx_hourly_1985_2005 = pd.date_range(start='1985-01-01 00:00:00',
                                    end='2005-12-31 23:00:00', freq='h')
idx_hourly_2040_2060 = pd.date_range(start='2040-01-01 00:00:00',
                                    end='2060-12-31 23:00:00', freq='h')
idx_hourly_2080_2100 = pd.date_range(start='2080-01-01 00:00:00',
                                    end='2100-12-31 23:00:00', freq='h')

idx_10mins_2003_2005 = pd.date_range(start='2003-01-01 00:00:00',
                                    end='2005-12-31 23:50:00', freq='10T')
idx_10mins_2015_2021 = pd.date_range(start='2015-01-01 00:00:00',
                                    end='2021-12-31 23:50:00', freq='10T')
idx_10mins_1994_2005 = pd.date_range(start='1994-01-01 00:00:00',
                                    end='2005-12-31 23:50:00', freq='10T')
idx_10mins_1985_2005 = pd.date_range(start='1985-01-01 00:00:00',
                                    end='2005-12-31 23:50:00', freq='10T')
idx_10mins_2040_2060 = pd.date_range(start='2040-01-01 00:00:00',
                                    end='2060-12-31 23:50:00', freq='10T')
idx_10mins_2080_2100 = pd.date_range(start='2080-01-01 00:00:00',
                                    end='2100-12-31 23:50:00', freq='10T')

periods = ['hist', 'future']
station_ids = [1443, 2787, 4189]
cms = ['CCCma-CanESM2_CCLM4-8-17', 'ICHEC-EC-EARTH_CCLM4-8-17', 
       'ICHEC-EC-EARTH_RCA4', 'IPSL-IPSL-CM5A-MR_RCA4',
       'MIROC-MIROC5_CCLM4-8-17', 'MPI-M-MPI-ESM-LR_CCLM4-8-17', 
       'MPI-M-MPI-ESM-LR_RCA4']
station_label1 = {1443: 'freiburg',
                  2787: 'kupferzell',
                  4189: 'altheim',
}

# --- load climate projections ---------------------------------------------------
dict_meteo = {}
for station_id in station_ids:
    dict_meteo[station_id] = {}
    for cm in cms:
        dict_meteo[station_id][cm] = {}
        for period in periods:
            file = base_path / 'climate_projections' / 'data' / f'BC-Neu-pcmgthr-MBCn_{cm}_DWD_{station_id}_{period}.csv'
            data = pd.read_csv(file, sep=',', index_col=0)
            data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
            data.columns = ['PREC', 'TA', 'TAD', 'RS']
            if period == 'hist':
                data_1985_2005 = pd.DataFrame(index=idx_daily_1985_2005)
                data_1985_2005 = data_1985_2005.join(data)
                dict_meteo[station_id][cm]['1985-2005'] = data_1985_2005
            elif period == 'future':
                data_2015_2021 = pd.DataFrame(index=idx_daily_2015_2021)
                data_2015_2021 = data_2015_2021.join(data)
                data_2040_2060 = pd.DataFrame(index=idx_daily_2040_2060)
                data_2040_2060 = data_2040_2060.join(data)
                data_2080_2100 = pd.DataFrame(index=idx_daily_2080_2100)
                data_2080_2100 = data_2080_2100.join(data)
                dict_meteo[station_id][cm]['2015-2021'] = data_2015_2021
                dict_meteo[station_id][cm]['2040-2060'] = data_2040_2060
                dict_meteo[station_id][cm]['2080-2100'] = data_2080_2100

    dict_meteo[station_id]['observed'] = {}
    file = base_path / 'input' / f'{station_label1[station_id]}' / 'observed' / '2015_2021' / 'PREC.txt'
    data_prec = pd.read_csv(file, sep='\t')
    data_prec.index = idx_10mins_2015_2021
    file = base_path / 'input' / f'{station_label1[station_id]}' / 'observed' / '2015_2021' / 'TA.txt'
    data_ta = pd.read_csv(file, sep='\t')
    data_2015_2021 = pd.DataFrame(index=idx_daily_2015_2021, columns=['PREC', 'TA'])
    data_2015_2021.loc[:, 'PREC'] = data_prec.loc[:, 'PREC'].resample('1D').sum().values
    data_2015_2021.loc[:, 'TA'] = data_ta.loc[:, 'TA'].values
    dict_meteo[station_id]['observed']['2015-2021'] = data_2015_2021

# --- projected annual air temperature and precipitation --------------------------------
dict_meteo_ann = {}
for station_id in station_ids:
    dict_meteo_ann[station_id] = {}
    for cm in cms:
        dict_meteo_ann[station_id][cm] = {}
        for period in periods:
            file = file = base_path / 'climate_projections' / 'data' / f'BC-Neu-pcmgthr-MBCn_{cm}_DWD_{station_id}_{period}.csv'
            data = pd.read_csv(file, sep=',', index_col=0)
            data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
            data.columns = ['PREC', 'TA', 'TAD', 'RS']
            data_ann = data.loc[:, 'PREC'].resample('1Y').sum().to_frame().join(data.loc[:, 'TA'].resample('1Y').mean().to_frame())
            dict_meteo_ann[station_id][cm][period] = data_ann
            
# --- observed annual air temperature and precipitation --------------------------------
for station_id in station_ids:
    for cm in cms:
        file = file = base_path / 'climate_projections' / 'data' / f'BC-Neu-pcmgthr-MBCn_{cm}_DWD_{station_id}_future.csv'
        data = pd.read_csv(file, sep=',', index_col=0)
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
        data.columns = ['PREC', 'TA', 'TAD', 'RS']
        data_2015_2021 = pd.DataFrame(index=idx_daily_2015_2021)
        data_2015_2021 = data_2015_2021.join(data)
        data_ann = data_2015_2021.loc[:, 'PREC'].resample('1Y').sum().to_frame().join(data_2015_2021.loc[:, 'TA'].resample('1Y').mean().to_frame())
        dict_meteo_ann[station_id][cm]['2015-2021'] = data_ann
    data_2015_2021 = dict_meteo[station_id]['observed']['2015-2021']
    data_ann = data_2015_2021.loc[:, 'PREC'].resample('1Y').sum().to_frame().join(data_2015_2021.loc[:, 'TA'].resample('1Y').mean().to_frame())
    dict_meteo_ann[station_id]['observed'] = {}
    dict_meteo_ann[station_id]['observed']['2015-2021'] = data_ann

# --- compare projected annual air temperature and precipitation ------------------------
color = {'CCCma-CanESM2_CCLM4-8-17_hist': '#eff3ff',
         'ICHEC-EC-EARTH_CCLM4-8-17_hist': '#c6dbef', 
         'ICHEC-EC-EARTH_RCA4_hist': '#9ecae1', 
         'IPSL-IPSL-CM5A-MR_RCA4_hist': '#6baed6',
         'MIROC-MIROC5_CCLM4-8-17_hist': '#4292c6', 
         'MPI-M-MPI-ESM-LR_CCLM4-8-17_hist': '#2171b5', 
         'MPI-M-MPI-ESM-LR_RCA4_hist': '#084594',
         'CCCma-CanESM2_CCLM4-8-17_future': '#fee5d9',
         'ICHEC-EC-EARTH_CCLM4-8-17_future': '#fcbba1', 
         'ICHEC-EC-EARTH_RCA4_future': '#fc9272', 
         'IPSL-IPSL-CM5A-MR_RCA4_future': '#fb6a4a',
         'MIROC-MIROC5_CCLM4-8-17_future': '#ef3b2c', 
         'MPI-M-MPI-ESM-LR_CCLM4-8-17_future': '#cb181d', 
         'MPI-M-MPI-ESM-LR_RCA4_future': '#99000d'}

label = {'CCCma-CanESM2_CCLM4-8-17': 'CCCma-CanESM2 CCLM4.8.17',
         'ICHEC-EC-EARTH_CCLM4-8-17': 'ICHEC-EC-EARTH CCLM4.8.17', 
         'ICHEC-EC-EARTH_RCA4': 'ICHEC-EC-EARTH RCA4', 
         'IPSL-IPSL-CM5A-MR_RCA4': 'IPSL-IPSL-CM5A-MR RCA4',
         'MIROC-MIROC5_CCLM4-8-17': 'MIROC-MIROC5 CCLM4.8.17', 
         'MPI-M-MPI-ESM-LR_CCLM4-8-17': 'MPI-M-MPI-ESM-LR CCLM4.8.17', 
         'MPI-M-MPI-ESM-LR_RCA4': 'MPI-M-MPI-ESM-LR RCA4'}
        
station_label = {1443: 'Freiburg',
                2787: 'Kupferzell',
                4189: 'Altheim',

}


for station_id in station_ids:
    fig, axs = plt.subplots(1, 1, figsize=(5, 3))
    for period in periods:
        for cm in cms:
            data = dict_meteo_ann[station_id][cm][period]
            prec_avg = data.loc[:, 'PREC'].mean()
            prec_std = data.loc[:, 'PREC'].std()
            ta_avg = data.loc[:, 'TA'].mean()
            ta_std = data.loc[:, 'TA'].std()
            axs.errorbar(ta_avg, prec_avg, xerr=ta_std, yerr=prec_std, fmt='o', label=label[cm], color=color[f'{cm}_{period}'], ms=2.5)
    axs.set_ylabel('Mean annual precipitation [mm]')
    axs.set_xlabel('Mean annual air temperature [°C]')
    axs.set_title(f'{station_label[station_id]} (station ID: {station_id})')
    lines, labels = axs.get_legend_handles_labels()
    fig.legend(lines[:7], labels[:7], loc='upper right', fontsize=6, frameon=False, bbox_to_anchor=(1.01, 0.9), title='1985-2005')
    fig.legend(lines[7:], labels[7:], loc='upper right', fontsize=6, frameon=False, bbox_to_anchor=(1.01, 0.55), title='2016-2100')
    fig.subplots_adjust(right=0.68)
    file = base_path_figs / f'projected_annual_prec_and_ta_{station_label[station_id]}.png'
    fig.savefig(file, dpi=250)
    plt.close(fig=fig)

for station_id in station_ids:
    fig, axs = plt.subplots(1, 1, figsize=(5, 3))
    data = dict_meteo_ann[station_id]['observed']['2015-2021']
    prec_avg = data.loc[:, 'PREC'].mean()
    ta_avg = data.loc[:, 'TA'].mean()
    axs.scatter(ta_avg, prec_avg, label='observed', color='blue', s=5)
    for cm in cms:
        data = dict_meteo_ann[station_id][cm]['2015-2021']
        prec_avg = data.loc[:, 'PREC'].mean()
        ta_avg = data.loc[:, 'TA'].mean()
        axs.scatter(ta_avg, prec_avg, label=label[cm], color=color[f'{cm}_future'], s=4)
    axs.set_ylabel('Mean annual precipitation [mm]')
    axs.set_xlabel('Mean annual air temperature [°C]')
    axs.set_title(f'{station_label[station_id]} (station ID: {station_id})')
    lines, labels = axs.get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper right', fontsize=6, frameon=False, bbox_to_anchor=(1.01, 0.9), title='2015-2021')
    fig.subplots_adjust(right=0.68)
    file = base_path_figs / f'annual_prec_and_ta_{station_label[station_id]}_obs_and_proj.png'
    fig.savefig(file, dpi=250)
    plt.close(fig=fig)

# # --- add minimum TA and maximum TA --------------------------------
# stations_for_ta = ['freiburg', 'weingarten', 'ingelfingen']
# for station_for_ta, station_id in zip(stations_for_ta, station_ids):
#     file = base_path / 'dwd' / f'{station_for_ta}' / 'meteo.txt'
#     data = pd.read_csv(file, sep=';', na_values=-999)
#     data.index = pd.to_datetime(data['MESS_DATUM'], format='%Y%m%d')
#     data_2015_2021 = pd.DataFrame(index=idx_daily_2015_2021)
#     data_2015_2021 = data_2015_2021.join(data)
#     file = base_path / 'input' / f'{station_label[station_id]}' / 'observed' / '2015_2021' / 'TA.txt'
#     df_TA = pd.read_csv(file, sep="\t", na_values=-999)
#     df_TA.loc[:, 'TA_min'] = data_2015_2021[' TNK'].values
#     df_TA.loc[:, 'TA_max'] = data_2015_2021[' TXK'].values
#     file = base_path / 'input' / f'{station_label[station_id]}' / 'observed' / '2015_2021' / 'TA.txt'
#     df_TA.to_csv(file, header=True, index=False, sep="\t")

# stations_for_ta = ['freiburg', 'weingarten']
# for station_for_ta, station_id in zip(stations_for_ta, [1443, 4189]):
#     file = base_path / 'dwd' / f'{station_for_ta}' / 'meteo.txt'
#     data = pd.read_csv(file, sep=';', na_values=-999)
#     data.index = pd.to_datetime(data['MESS_DATUM'], format='%Y%m%d')
#     data_1994_2005 = pd.DataFrame(index=idx_daily_1994_2005)
#     data_1994_2005 = data_1994_2005.join(data)
#     file = base_path / 'input' / f'{station_label[station_id]}' / 'observed' / '1994_2005' / 'TA.txt'
#     df_TA = pd.read_csv(file, sep="\t", na_values=-999)
#     df_TA.loc[:, 'TA_min'] = data_1994_2005[' TNK'].values
#     df_TA.loc[:, 'TA_max'] = data_1994_2005[' TXK'].values
#     file = base_path / 'input' / f'{station_label[station_id]}' / '1994_2005' / 'TA.txt'
#     df_TA.to_csv(file, header=True, index=False, sep="\t")

# station_for_ta = 'ingelfingen'
# station_id  = 2787
# file = base_path / 'dwd' / f'{station_for_ta}' / 'meteo.txt'
# data = pd.read_csv(file, sep=';', na_values=-999)
# data.index = pd.to_datetime(data['MESS_DATUM'], format='%Y%m%d')
# data_2003_2005 = pd.DataFrame(index=idx_daily_2003_2005)
# data_2003_2005 = data_2003_2005.join(data)
# file = base_path / 'input' / f'{station_label[station_id]}' / 'observed' / '2003_2005' / 'TA.txt'
# df_TA = pd.read_csv(file, sep="\t", na_values=-999)
# df_TA.loc[:, 'TA_min'] = data_2003_2005[' TNK'].values
# df_TA.loc[:, 'TA_max'] = data_2003_2005[' TXK'].values
# file = base_path / 'input' / f'{station_label[station_id]}' / 'observed' / '2003_2005' / 'TA.txt'
# df_TA.to_csv(file, header=True, index=False, sep="\t")