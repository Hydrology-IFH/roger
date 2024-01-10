import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import pickle

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

mpl.rcParams["font.size"] = 7
mpl.rcParams["axes.titlesize"] = 8
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["xtick.labelsize"] = 7
mpl.rcParams["ytick.labelsize"] = 7
mpl.rcParams["legend.fontsize"] = 7
mpl.rcParams["legend.title_fontsize"] = 8
sns.set_style("ticks")
sns.plotting_context(
    "paper",
    font_scale=1,
    rc={
        "font.size": 7.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "legend.title_fontsize": 8.0,
    },
)

base_path = Path(__file__).parent
# directory of figures
base_path_figs = base_path / "figures" / "climate_projections"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

station_ids = [2812, 259,
               4881, 1711,
               1255, 731, 710,
               3418, 2814, 2072,
               3761, 5206,
              ]
stations = ["lahr", "muellheim", 
            "stockach", "gottmadingen",
            "eppingen-elsenz", "bruchsal-heidelsheim", "bretten",
            "ehingen-kirchen", "merklingen", "hayingen",
            "oehringen", "vellberg-kleinaltdorf"]
cms = [
    "CCCma-CanESM2_CCLM4-8-17",
    "MPI-M-MPI-ESM-LR_RCA4",
]
station_label = {
    4881: "Stockach",
    1255: "Eppingen-Elsenz",
    731: "Bruchsal-Heidelsheim",
    2812: "Lahr",
    3418: "Ehingen-Kirchen",
    2814: "Merklingen",
    259: "Muellheim",
    3761: "Oehringen",
    1711: "Gottmadingen",
    2787: "Kupferzell",
    4094: "Weingarten",
    710: "Bretten",  # 7490
    2072: "Hayingen",
    5206: "Vellberg-Kleinaltdorf",
}

station_label1 = {
    4881: "stockach",
    1255: "eppingen-elsenz",
    731: "bruchsal-heidelsheim",
    2812: "lahr",
    3418: "ehingen-kirchen",
    2814: "merklingen",
    259: "muellheim",
    3761: "oehringen",
    1711: "gottmadingen",
    2787: "kupferzell",
    4094: "weingarten",
    710: "bretten",
    2072: "hayingen",
    5206: "vellberg-kleinaltdorf",
}

color = {
    "CCCma-CanESM2_CCLM4-8-17_hist": "#eff3ff",
    "MPI-M-MPI-ESM-LR_RCA4_hist": "#084594",
    "CCCma-CanESM2_CCLM4-8-17_future": "#fee5d9",
    "MPI-M-MPI-ESM-LR_RCA4_future": "#99000d",
}

label = {
    "CCCma-CanESM2_CCLM4-8-17": "CCCma-CanESM2 CCLM4.8.17",
    "MPI-M-MPI-ESM-LR_RCA4": "MPI-M-MPI-ESM-LR RCA4",
}


def _calc_pet_with_makkink(rs, ta, z, c1=0.63, c2=-0.05):
    """Calculate potential evapotranspiration according to Makkink.

    Args
    ----------
    rs : np.ndarray
        solar radiation (in MJ m-2 day-1)

    ta : np.ndarray
        air temperature (in celsius)

    z : float
        elevation above sea level (in m)

    c1 : float, optional
        Makkink coefficient (-)

    c2 : float, optional
        Makkink coefficient (-)

    Reference
    ----------
    Makkink, G. F., Testing the Penman formula by means of lysimeters,
    J. Inst. Wat. Engrs, 11, 277-288, 1957.

    Returns
    ----------
    pet : np.ndarray
        potential evapotranspiration
    """
    # slope of saturation vapour pressure curve (in kPa celsius-1)
    svpc = 4098 * (0.6108 * np.exp((17.27 * ta) / (ta + 237.3))) / (ta + 237.3) ** 2

    # atmospheric pressure (in kPa)
    p = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26

    # psychometric constant (in kPa celsius-1)
    gam = 0.665 * 1e-3 * p

    # special heat of evaporation (in MJ m-2 mm-1)
    lam = 0.0864 * (28.4 - 0.028 * ta)

    # potential evapotranspiration (in mm)
    pet = (svpc / (svpc + gam)) * ((c1 * rs / lam) + c2)

    return np.where(pet < 0, 0, pet)


# --- time index to join data --------------------------------------------------
idx_annually_1985_2099 = pd.date_range(start="1985-01-01", end="2099-12-31", freq="y")
idx_annually_1985_2014 = pd.date_range(start="1985-01-01", end="2014-12-31", freq="y")
idx_annually_2030_2059 = pd.date_range(start="2030-01-01", end="2059-12-31", freq="y")
idx_annually_2070_2099 = pd.date_range(start="2070-01-01", end="2099-12-31", freq="y")

idx_seasonally_1985_2099 = pd.date_range(start="1985-03-01", end="2099-09-01", freq="3MS")
idx_seasonally_1985_2014 = pd.date_range(start="1985-03-01", end="2015-09-01", freq="3MS")
idx_seasonally_2030_2059 = pd.date_range(start="2030-03-01", end="2059-09-01", freq="3MS")
idx_seasonally_2070_2099 = pd.date_range(start="2070-03-01", end="2099-09-01", freq="3MS")

idx_daily_1985_2099 = pd.date_range(start="1985-01-01", end="2099-12-31", freq="d")
idx_daily_1985_2014 = pd.date_range(start="1985-01-01", end="2014-12-31", freq="d")
idx_daily_2030_2059 = pd.date_range(start="2030-01-01", end="2059-12-31", freq="d")
idx_daily_2070_2099 = pd.date_range(start="2070-01-01", end="2099-12-31", freq="d")

idx_hourly_1985_2099 = pd.date_range(start="1985-01-01 00:00:00", end="2099-12-31 23:00:00", freq="h")
idx_hourly_1985_2014 = pd.date_range(start="1985-01-01 00:00:00", end="2014-12-31 23:00:00", freq="h")
idx_hourly_2030_2059 = pd.date_range(start="2030-01-01 00:00:00", end="2059-12-31 23:00:00", freq="h")
idx_hourly_2070_2099 = pd.date_range(start="2070-01-01 00:00:00", end="2099-12-31 23:00:00", freq="h")

idx_3hourly_1985_2099c = pd.date_range(start="1985-01-01 01:30:00", end="2099-12-31 22:30:00", freq="3h")
idx_3hourly_1985_2099 = pd.date_range(start="1985-01-01 00:00:00", end="2099-12-31 23:00:00", freq="3h")

idx_10mins_1985_2099 = pd.date_range(start="1985-01-01 00:00:00", end="2100-12-31 23:50:00", freq="10T")
idx_10mins_1985_2014 = pd.date_range(start="1985-01-01 00:00:00", end="2014-12-31 23:50:00", freq="10T")
idx_10mins_2030_2059 = pd.date_range(start="2030-01-01 00:00:00", end="2059-12-31 23:50:00", freq="10T")
idx_10mins_2070_2099 = pd.date_range(start="2070-01-01 00:00:00", end="2099-12-31 23:50:00", freq="10T")

# --- load climate projections ---------------------------------------------------
dict_bc_meteo_daily = {}
for station_id in station_ids:
    if station_label1[station_id] == "altheim":
        elevation = 534
    elif station_label1[station_id] == "bretten":
        elevation = 281
    elif station_label1[station_id] == "bruchsal-heidelsheim":
        elevation = 130
    elif station_label1[station_id] == "eppingen-elsenz":
        elevation = 226
    elif station_label1[station_id] == "ehingen-kirchen":
        elevation = 594
    elif station_label1[station_id] == "freiburg":
        elevation = 236
    elif station_label1[station_id] == "gottmadingen":
        elevation = 438
    elif station_label1[station_id] == "hayingen":
        elevation = 665
    elif station_label1[station_id] == "kupferzell":
        elevation = 355
    elif station_label1[station_id] == "lahr":
        elevation = 155
    elif station_label1[station_id] == "merklingen":
        elevation = 685
    elif station_label1[station_id] == "muellheim":
        elevation = 275
    elif station_label1[station_id] == "oehringen":
        elevation = 276
    elif station_label1[station_id] == "stockach":
        elevation = 532
    elif station_label1[station_id] == "vellberg-kleinaltdorf":
        elevation = 396
    elif station_label1[station_id] == "weingarten":
        elevation = 440

    dict_bc_meteo_daily[station_label1[station_id]] = {}
    for cm in cms:
        dict_bc_meteo_daily[station_label1[station_id]][cm] = {}
        file = (
            base_path / "climate_projections" / "corrected" / f"BC_pr-tas-hurs-rsds-daily_FullTs_{cm}_station-DWD_{station_id}.csv"
        )
        data = pd.read_csv(file, sep=",", index_col=1)
        data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
        data = data.iloc[:, 1:]
        data.columns = ["PREC", "TA", "TAD", "RS"]
        data_1985_2099 = pd.DataFrame(index=idx_daily_1985_2099)
        data_1985_2099 = data_1985_2099.join(data)
        # fill NaNs at 29th February
        data_1985_2099.loc[:, "TA"] = data_1985_2099["TA"].interpolate()
        data_1985_2099.loc[:, "RS"] = data_1985_2099["RS"].interpolate()
        data_1985_2099.loc[:, "PET"] = _calc_pet_with_makkink(
            data_1985_2099.loc[:, "RS"].values, data_1985_2099.loc[:, "TA"].values, elevation
        )
        dict_bc_meteo_daily[station_label1[station_id]][cm]["1985-2099"] = data_1985_2099

        data_1985_2014 = pd.DataFrame(index=idx_daily_1985_2014)
        data_1985_2014 = data_1985_2014.join(data)
        # fill NaNs at 29th February
        data_1985_2014.loc[:, "TA"] = data_1985_2014["TA"].interpolate()
        data_1985_2014.loc[:, "RS"] = data_1985_2014["RS"].interpolate()
        data_1985_2014.loc[:, "PET"] = _calc_pet_with_makkink(
            data_1985_2014.loc[:, "RS"].values, data_1985_2014.loc[:, "TA"].values, elevation
        )
        dict_bc_meteo_daily[station_label1[station_id]][cm]["1985-2014"] = data_1985_2014
        data_2030_2059 = pd.DataFrame(index=idx_daily_2030_2059)
        data_2030_2059 = data_2030_2059.join(data)
        # fill NaNs at 29th February
        data_2030_2059.loc[:, "TA"] = data_2030_2059["TA"].interpolate()
        data_2030_2059.loc[:, "RS"] = data_2030_2059["RS"].interpolate()
        data_2030_2059.loc[:, "PET"] = _calc_pet_with_makkink(
            data_2030_2059.loc[:, "RS"].values, data_2030_2059.loc[:, "TA"].values, elevation
        )
        data_2070_2099 = pd.DataFrame(index=idx_daily_2070_2099)
        data_2070_2099 = data_2070_2099.join(data)
        # fill NaNs at 29th February
        data_2070_2099.loc[:, "TA"] = data_2070_2099["TA"].interpolate()
        data_2070_2099.loc[:, "RS"] = data_2070_2099["RS"].interpolate()
        data_2070_2099.loc[:, "PET"] = _calc_pet_with_makkink(
            data_2070_2099.loc[:, "RS"].values, data_2070_2099.loc[:, "TA"].values, elevation
        )
        dict_bc_meteo_daily[station_label1[station_id]][cm]["1985-2014"] = data_1985_2014
        dict_bc_meteo_daily[station_label1[station_id]][cm]["2030-2059"] = data_2030_2059
        dict_bc_meteo_daily[station_label1[station_id]][cm]["2070-2099"] = data_2070_2099


# --- correction of daily minimum air temperature and daily maximum air temperature -------
for station_id in station_ids:
    for cm in ["MPI-M-MPI-ESM-LR_RCA4", "CCCma-CanESM2_CCLM4-8-17"]:
        # uncorrected daily minimum and maximum air temperature
        file = (
            base_path
            / "climate_projections"
            / "uncorrected"
            / f"tmin-max-daily_FullTs_{cm}_station-DWD_{station_id}.csv"
        )
        data = pd.read_csv(file, sep=",", index_col=2)
        data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
        data = data.iloc[:, 2:]
        cond = data["Var"].values == "tasmax"
        data_ta_max = data.loc[cond, "Center"]
        cond = data["Var"].values == "tasmin"
        data_ta_min = data.loc[cond, "Center"]
        data1 = pd.DataFrame(index=data_ta_min.index, columns=["TA_min", "TA_max"])
        data1.loc[:, "TA_min"] = data_ta_min.values
        data1.loc[:, "TA_max"] = data_ta_max.values
        data_uc_ta_min_max_1985_2099 = pd.DataFrame(index=idx_daily_1985_2099)
        data_uc_ta_min_max_1985_2099 = data_uc_ta_min_max_1985_2099.join(data1)
        # uncorrected daily average air temperature
        file = base_path / "climate_projections" / "uncorrected" / f"UC_pr-tas-hurs-rsds-daily_FullTs_{cm}_station-DWD_{station_id}.csv"
        data = pd.read_csv(file, sep=",", index_col=1)
        data = data.iloc[:, 1:]
        data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
        data.columns = ["PREC", "TA", "TAD", "RS"]
        data_uc_1985_2099 = pd.DataFrame(index=idx_daily_1985_2099)
        data_uc_1985_2099 = data_uc_1985_2099.join(data)
        # fill NaNs at 29th February
        data_uc_1985_2099.loc[:, "TA"] = data_uc_1985_2099["TA"].interpolate()
        data_uc_1985_2099.loc[:, "RS"] = data_uc_1985_2099["RS"].interpolate()
        data_uc_1985_2099 = data_uc_1985_2099.join(data_uc_ta_min_max_1985_2099)
        # bias correction daily minimum and maximum air temperature
        # with difference between bias-corrected daily average air temperature and uncorrected air temperature
        data_uc_1985_2099.loc[:, "scale"] = (
            dict_bc_meteo_daily[station_label1[station_id]][cm]["1985-2099"].loc[:, "TA"] - data_uc_1985_2099.loc[:, "TA"]
        )
        data_1985_2099 = dict_bc_meteo_daily[station_label1[station_id]][cm]["1985-2099"]
        data_1985_2099.loc[:, "TA_min"] = data_uc_1985_2099.loc[:, "TA_min"] + data_uc_1985_2099.loc[:, "scale"]
        data_1985_2099.loc[:, "TA_max"] = data_uc_1985_2099.loc[:, "TA_max"] + data_uc_1985_2099.loc[:, "scale"]
        # fill NaNs at 29th February
        data_1985_2099.loc[:, "TA_min"] = data_1985_2099["TA_min"].interpolate()
        data_1985_2099.loc[:, "TA_max"] = data_1985_2099["TA_max"].interpolate()
        data_ta_min_max_1985_2099 = data_1985_2099.loc[:, "TA_min":"TA_max"]

        data_1985_2014 = dict_bc_meteo_daily[station_label1[station_id]][cm]["1985-2014"]
        data_1985_2014 = data_1985_2014.join(data_ta_min_max_1985_2099)
        # fill NaNs at 29th February
        data_1985_2014.loc[:, "TA_min"] = data_1985_2014["TA_min"].interpolate()
        data_1985_2014.loc[:, "TA_max"] = data_1985_2014["TA_max"].interpolate()
        dict_bc_meteo_daily[station_label1[station_id]][cm]["1985-2014"] = data_1985_2014
        data_2030_2059 = dict_bc_meteo_daily[station_label1[station_id]][cm]["2030-2059"]
        data_2030_2059 = data_2030_2059.join(data_ta_min_max_1985_2099)
        # fill NaNs at 29th February
        data_2030_2059.loc[:, "TA_min"] = data_2030_2059["TA_min"].interpolate()
        data_2030_2059.loc[:, "TA_max"] = data_2030_2059["TA_max"].interpolate()
        data_2070_2099 = dict_bc_meteo_daily[station_label1[station_id]][cm]["2070-2099"]
        data_2070_2099 = data_2070_2099.join(data_ta_min_max_1985_2099)
        # fill NaNs at 29th February
        data_2070_2099.loc[:, "TA_min"] = data_2070_2099["TA_min"].interpolate()
        data_2070_2099.loc[:, "TA_max"] = data_2070_2099["TA_max"].interpolate()
        dict_bc_meteo_daily[station_label1[station_id]][cm]["2030-2059"] = data_2030_2059
        dict_bc_meteo_daily[station_label1[station_id]][cm]["2070-2099"] = data_2070_2099
        dict_bc_meteo_daily[station_label1[station_id]][cm]["1985-2099"] = data_1985_2099

# --- downscale uncorrected precipitation to 10 minutes ------------------------
dict_precip_10mins = {}
cm = "MPI-M-MPI-ESM-LR_RCA4"
for station_id in station_ids:
    dict_precip_10mins[station_label1[station_id]] = {}
    dict_precip_10mins[station_label1[station_id]][cm] = {}
    file = (
        base_path
        / "climate_projections"
        / "uncorrected"
        / f"pr_hourly_FullTs_{cm}_station-DWD_{station_id}.csv"
    )
    data = pd.read_csv(file, sep=",", index_col=0)

    file = base_path / "climate_projections" / "uncorrected" / "datetime_MPI-RCA1hr.txt"
    data_idx = pd.read_csv(file, sep=";")
    data.index = pd.to_datetime(data_idx.iloc[:, 0].astype(str).values, format="%Y-%m-%d %H:%M")
    data_hourly = pd.DataFrame(index=idx_hourly_1985_2099)
    data_hourly.loc[:, "PREC_hourly"] = data.loc["1985":"2099", "Center"].values
    # resample to daily precipitation
    data_daily = data_hourly.resample("1D").sum()
    data_daily.columns = ["PREC_daily"]

    data_10mins = pd.DataFrame(index=idx_10mins_1985_2099)
    data_10mins = data_10mins.join([data_daily, data_hourly])
    data_10mins = data_10mins.ffill()
    # donwnscale hourly precipitation by linear interpolation
    data_10mins.loc[:, "PREC"] = data_10mins.loc[:, "PREC_hourly"] / 6
    # scaling factor derived from uncorrected precipitation for downscaling of daily bias-corrected precipitation
    data_10mins.loc[:, "scale"] = data_10mins.loc[:, "PREC"] / data_10mins.loc[:, "PREC_daily"]
    data_10mins = data_10mins.fillna(0)

    data_1985_2014 = pd.DataFrame(index=idx_10mins_1985_2014)
    data_1985_2014 = data_1985_2014.join(data_10mins)
    dict_precip_10mins[station_label1[station_id]][cm]["1985-2014"] = data_1985_2014
    data_2030_2059 = pd.DataFrame(index=idx_10mins_2030_2059)
    data_2030_2059 = data_2030_2059.join(data_10mins)
    data_2070_2099 = pd.DataFrame(index=idx_10mins_2070_2099)
    data_2070_2099 = data_2070_2099.join(data_10mins)
    dict_precip_10mins[station_label1[station_id]][cm]["2030-2059"] = data_2030_2059
    dict_precip_10mins[station_label1[station_id]][cm]["2070-2099"] = data_2070_2099
    dict_precip_10mins[station_label1[station_id]][cm]["1985-2099"] = data_10mins

cm = "CCCma-CanESM2_CCLM4-8-17"
for station_id in station_ids:
    dict_precip_10mins[station_label1[station_id]][cm] = {}
    file = (
        base_path
        / "climate_projections"
        / "uncorrected"
        / f"pr-Subdaily_FullTs_{cm}_station-DWD_{station_id}.csv"
    )
    data = pd.read_csv(file, sep=",", index_col=0)

    file = base_path / "climate_projections" / "uncorrected" / "datetime_CANESM-CLM3hr.txt"
    data_idx = pd.read_csv(file, sep=";")
    data.index = pd.to_datetime(data_idx.iloc[:, 0].astype(str).values, format="%Y-%m-%d %H:%M")
    data_3hourly = pd.DataFrame(index=idx_3hourly_1985_2099c)
    data_3hourly = data_3hourly.join(data.loc[:, "Center"].to_frame())
    data_3hourly.index = idx_3hourly_1985_2099
    data_3hourly.columns = ["PREC_3hourly"]
    data_3hourly.loc[:, "PREC_3hourly"] = data_3hourly.loc[:, "PREC_3hourly"].values
    # fill 29th February in leap years
    data_3hourly = data_3hourly.fillna(0)
    # resample to daily precipitation
    data_daily = data_3hourly.resample("1D").sum()
    data_daily.columns = ["PREC_daily"]

    data_10mins = pd.DataFrame(index=idx_10mins_1985_2099)
    data_10mins = data_10mins.join([data_daily, data_3hourly])
    data_10mins = data_10mins.ffill()
    # donwnscale 3-hourly precipitation by linear interpolation
    data_10mins.loc[:, "PREC"] = data_10mins.loc[:, "PREC_3hourly"] / 18
    # scaling factor derived from uncorrected precipitation for downscaling of daily bias-corrected precipitation
    data_10mins.loc[:, "scale"] = data_10mins.loc[:, "PREC"] / data_10mins.loc[:, "PREC_daily"]
    data_10mins = data_10mins.fillna(0)

    data_1985_2014 = pd.DataFrame(index=idx_10mins_1985_2014)
    data_1985_2014 = data_1985_2014.join(data_10mins)
    dict_precip_10mins[station_label1[station_id]][cm]["1985-2014"] = data_1985_2014
    data_2030_2059 = pd.DataFrame(index=idx_10mins_2030_2059)
    data_2030_2059 = data_2030_2059.join(data_10mins)
    data_2070_2099 = pd.DataFrame(index=idx_10mins_2070_2099)
    data_2070_2099 = data_2070_2099.join(data_10mins)
    dict_precip_10mins[station_label1[station_id]][cm]["2030-2059"] = data_2030_2059
    dict_precip_10mins[station_label1[station_id]][cm]["2070-2099"] = data_2070_2099
    dict_precip_10mins[station_label1[station_id]][cm]["1985-2099"] = data_10mins

# --- downscale bias-corrected precipitation to 10 minutes ------------------------
dict_bc_precip_10mins = {}
for station_id in station_ids:
    dict_bc_precip_10mins[station_label1[station_id]] = {}
    for cm in ["MPI-M-MPI-ESM-LR_RCA4", "CCCma-CanESM2_CCLM4-8-17"]:
        dict_bc_precip_10mins[station_label1[station_id]][cm] = {}
        data_daily = dict_bc_meteo_daily[station_label1[station_id]][cm]["1985-2099"].loc[:, "PREC"].to_frame()

        data_10mins = pd.DataFrame(index=idx_10mins_1985_2099)
        # donwnscale daily precipitation by subdaily scaling factors
        data_10mins = data_10mins.join(data_daily)
        data_10mins = data_10mins.ffill()
        data_10mins.loc[:, "PREC"] = (
            data_10mins.loc[:, "PREC"] * dict_precip_10mins[station_label1[station_id]][cm]["1985-2099"].loc[:, "scale"]
        )
        # replace numerical artefacts
        cond0 = data_10mins["PREC"] < 0.001
        data_10mins.loc[cond0, "PREC"] = 0

        data_1985_2014 = pd.DataFrame(index=idx_10mins_1985_2014)
        data_1985_2014 = data_1985_2014.join(data_10mins)
        dict_bc_precip_10mins[station_label1[station_id]][cm]["1985-2014"] = data_1985_2014
        data_2030_2059 = pd.DataFrame(index=idx_10mins_2030_2059)
        data_2030_2059 = data_2030_2059.join(data_10mins)
        data_2070_2099 = pd.DataFrame(index=idx_10mins_2070_2099)
        data_2070_2099 = data_2070_2099.join(data_10mins)
        dict_bc_precip_10mins[station_label1[station_id]][cm]["2030-2059"] = data_2030_2059
        dict_bc_precip_10mins[station_label1[station_id]][cm]["2070-2099"] = data_2070_2099

# # --- write input data to .txt -------------------------------------
# for station_id in station_ids:
#     station = station_label1[station_id]
#     for cm in ["MPI-M-MPI-ESM-LR_RCA4", "CCCma-CanESM2_CCLM4-8-17"]:
#         for period in ["1985-2014", "2030-2059", "2070-2099"]:
#             data_precip = dict_bc_precip_10mins[station_label1[station_id]][cm][period]
#             data_meteo = dict_bc_meteo_daily[station_label1[station_id]][cm][period]
#             data_ta = data_meteo.loc[:, ["TA", "TA_min", "TA_max"]]
#             data_rs = data_meteo.loc[:, ["RS"]]

#             path_dir = base_path / "input"
#             if not os.path.exists(path_dir):
#                 os.mkdir(path_dir)

#             path_dir = base_path / "input" / station
#             if not os.path.exists(path_dir):
#                 os.mkdir(path_dir)

#             path_dir = base_path / "input" / station / cm
#             if not os.path.exists(path_dir):
#                 os.mkdir(path_dir)

#             path_dir = base_path / "input" / station / cm / period
#             if not os.path.exists(path_dir):
#                 os.mkdir(path_dir)

#             idx_10mins = pd.date_range(start=str(data_precip.index[0]), end=str(data_precip.index[-1]), freq="10T")
#             idx_daily = pd.date_range(start=str(data_ta.index[0]), end=str(data_ta.index[-1]), freq="d")
#             df_PREC = pd.DataFrame(index=idx_10mins, columns=["YYYY", "MM", "DD", "hh", "mm", "PREC"])
#             df_PREC["YYYY"] = data_precip.index.year.values
#             df_PREC["MM"] = data_precip.index.month.values
#             df_PREC["DD"] = data_precip.index.day.values
#             df_PREC["hh"] = data_precip.index.hour.values
#             df_PREC["mm"] = data_precip.index.minute.values
#             df_PREC["PREC"] = data_precip["PREC"].values
#             path_txt = path_dir / "PREC.txt"
#             df_PREC.to_csv(path_txt, header=True, index=False, sep="\t")
#             nas = np.sum(np.isnan(data_precip["PREC"].values))
#             print(f"{station}-{cm}-{period}-PREC: {nas}")

#             df_TA = pd.DataFrame(index=idx_daily, columns=["YYYY", "MM", "DD", "hh", "mm"])
#             df_TA["YYYY"] = data_ta.index.year.values
#             df_TA["MM"] = data_ta.index.month.values
#             df_TA["DD"] = data_ta.index.day.values
#             df_TA["hh"] = data_ta.index.hour.values
#             df_TA["mm"] = data_ta.index.minute.values
#             df_TA["TA"] = data_ta["TA"].values
#             df_TA["TA_min"] = data_ta["TA_min"].values
#             df_TA["TA_max"] = data_ta["TA_max"].values
#             path_txt = path_dir / "TA.txt"
#             df_TA.to_csv(path_txt, header=True, index=False, sep="\t")
#             nas = np.sum(np.isnan(data_ta["TA"].values))
#             print(f"{station}-{cm}-{period}-TA: {nas} NaN values")
#             nas = np.sum(np.isnan(data_ta["TA_min"].values))
#             print(f"{station}-{cm}-{period}-TA_min: {nas} NaN values")
#             nas = np.sum(np.isnan(data_ta["TA_max"].values))
#             print(f"{station}-{cm}-{period}-TA_max: {nas} NaN values")

#             df_RS = pd.DataFrame(index=idx_daily, columns=["YYYY", "MM", "DD", "hh", "mm"])
#             df_RS["YYYY"] = data_rs.index.year.values
#             df_RS["MM"] = data_rs.index.month.values
#             df_RS["DD"] = data_rs.index.day.values
#             df_RS["hh"] = data_rs.index.hour.values
#             df_RS["mm"] = data_rs.index.minute.values
#             df_RS["RS"] = data_rs["RS"].values * 0.0864  # convert watt (i.e. J/s) to MJ/day
#             path_txt = path_dir / "RS.txt"
#             df_RS.to_csv(path_txt, header=True, index=False, sep="\t")
#             nas = np.sum(np.isnan(data_rs["RS"].values))
#             print(f"{station}-{cm}-{period}-RS: {nas} NaN values")

# --- plot time series of daily projected air temperature, potential evapotranspiration and precipitation ----
for station_id in station_ids:
    for cm in ["MPI-M-MPI-ESM-LR_RCA4", "CCCma-CanESM2_CCLM4-8-17"]:
        for period in ["1985-2014", "2030-2059", "2070-2099"]:
            data_precip = dict_bc_precip_10mins[station_label1[station_id]][cm][period]
            data_precip_uc = dict_precip_10mins[station_label1[station_id]][cm][period]
            data_meteo = dict_bc_meteo_daily[station_label1[station_id]][cm][period]
            data_ta = data_meteo.loc[:, ["TA", "TA_min", "TA_max"]]

            fig, axs = plt.subplots(1, 1, figsize=(6, 2))
            axs.plot(data_precip_uc.index, data_precip_uc["scale"].values, color="black", lw=0.5)
            axs.set_xlim(data_precip_uc.index[0], data_precip_uc.index[-1])
            axs.set_ylim(
                0,
            )
            axs.set_xlabel("Time [year]")
            axs.set_ylabel("Scale [-]")
            fig.tight_layout()
            file = base_path_figs / f"precip_scale_{station_label1[station_id]}_{cm}_{period}.png"
            fig.savefig(file, dpi=300)
            plt.close(fig=fig)

            fig, axs = plt.subplots(1, 1, figsize=(6, 2))
            axs.plot(data_precip.index, data_precip["PREC"].values, color="blue", lw=1)
            axs.plot(data_precip_uc.index, data_precip_uc["PREC"].values, ls="--", color="blue", lw=1, alpha=0.5)
            axs.set_xlim(data_precip.index[0], data_precip.index[-1])
            axs.set_ylim(
                0,
            )
            axs.set_xlabel("Time [year]")
            axs.set_ylabel("PRECIP [mm/10 minutes]")
            fig.tight_layout()
            file = base_path_figs / f"precip_{station_label1[station_id]}_{cm}_{period}.png"
            fig.savefig(file, dpi=300)
            plt.close(fig=fig)

            fig, axs = plt.subplots(1, 1, figsize=(6, 2))
            axs.plot(data_precip.index, data_precip["PREC"].cumsum().values, color="blue", lw=1)
            axs.plot(
                data_precip_uc.index, data_precip_uc["PREC"].cumsum().values, ls="--", color="blue", lw=1, alpha=0.5
            )
            axs.set_xlim(data_precip.index[0], data_precip.index[-1])
            axs.set_ylim(
                0,
            )
            axs.set_xlabel("Time [year]")
            axs.set_ylabel("PRECIP [mm]")
            fig.tight_layout()
            file = base_path_figs / f"precip_{station_label1[station_id]}_{cm}_{period}_cumulated.png"
            fig.savefig(file, dpi=300)
            plt.close(fig=fig)

            fig, axs = plt.subplots(1, 1, figsize=(6, 2))
            axs.plot(data_precip.index, data_precip["PREC"].cumsum().values, color="blue", lw=1)
            axs.set_xlim(data_precip.index[0], data_precip.index[-1])
            axs.set_ylim(
                0,
            )
            axs.set_xlabel("Time [year]")
            axs.set_ylabel("PRECIP [mm]")
            fig.tight_layout()
            file = base_path_figs / f"precip_bc_{station_label1[station_id]}_{cm}_{period}_cumulated.png"
            fig.savefig(file, dpi=300)
            plt.close(fig=fig)

            fig, axs = plt.subplots(1, 1, figsize=(6, 2))
            axs.plot(data_ta.index, data_ta["TA"].values, color="red", lw=0.5)
            axs.fill_between(
                data_ta.index,
                data_ta["TA_min"].values,
                data_ta["TA_max"].values,
                color="red",
                edgecolor=None,
                alpha=0.3,
            )

            axs.set_xlim(data_ta.index[0], data_ta.index[-1])
            axs.set_xlabel("Time [year]")
            axs.set_ylabel("TA [degC]")
            fig.tight_layout()
            file = base_path_figs / f"ta_{station_label1[station_id]}_{cm}_{period}.png"
            fig.savefig(file, dpi=300)
            plt.close(fig=fig)

            fig, axs = plt.subplots(1, 1, figsize=(6, 2))
            axs.fill_between(
                data_ta.index,
                data_ta["TA_min"].values,
                data_ta["TA_max"].values,
                color="red",
                edgecolor=None,
                alpha=0.3,
            )
            axs.set_xlim(data_ta.index[0], data_ta.index[-1])
            axs.set_xlabel("Time [year]")
            axs.set_ylabel("TA [degC]")
            fig.tight_layout()
            file = base_path_figs / f"ta_min_max_{station_label1[station_id]}_{cm}_{period}.png"
            fig.savefig(file, dpi=300)
            plt.close(fig=fig)

# --- aggregate daily projected air temperature, potential evapotranspiration and precipitation to annual --------
dict_meteo_ann = {}
for station_id in station_ids:
    dict_meteo_ann[station_label1[station_id]] = {}
    for cm in cms:
        dict_meteo_ann[station_label1[station_id]][cm] = {}
        data = dict_bc_meteo_daily[station_label1[station_id]][cm]["1985-2099"]
        data_ann = (
            data.loc[:, "PREC"]
            .resample("1Y")
            .sum()
            .to_frame()
            .join(data.loc[:, "TA"].resample("1Y").mean().to_frame())
            .join(data.loc[:, "PET"].resample("1Y").sum().to_frame())
        )
        dict_meteo_ann[station_label1[station_id]][cm]["1985-2099"] = data_ann
        data_ann_1985_2014 = pd.DataFrame(index=idx_annually_1985_2014)
        data_ann_1985_2014 = data_ann_1985_2014.join(data_ann)
        dict_meteo_ann[station_label1[station_id]][cm]["1985-2014"] = data_ann_1985_2014
        data_ann_2030_2059 = pd.DataFrame(index=idx_annually_2030_2059)
        data_ann_2030_2059 = data_ann_2030_2059.join(data_ann)
        data_ann_2070_2099 = pd.DataFrame(index=idx_annually_2070_2099)
        data_ann_2070_2099 = data_ann_2070_2099.join(data_ann)
        dict_meteo_ann[station_label1[station_id]][cm]["2030-2059"] = data_ann_2030_2059
        dict_meteo_ann[station_label1[station_id]][cm]["2070-2099"] = data_ann_2070_2099

# --- calculate total deltas -------------------------------------------------
dict_deltas_climate = {}
for station_id in station_ids:
    dict_deltas_climate[station_label1[station_id]] = {}
    for cm in cms:
        dict_deltas_climate[station_label1[station_id]][cm] = {}
        data_ref = dict_meteo_ann[station_label1[station_id]][cm]["1985-2014"]
        data_nf = dict_meteo_ann[station_label1[station_id]][cm]["2030-2059"]
        data_ff = dict_meteo_ann[station_label1[station_id]][cm]["2070-2099"]
        prec_avg_ref = data_ref.loc[:, "PREC"].mean()
        prec_ipr_ref = np.nanpercentile(data_ref.loc[:, "PREC"], 90) - np.nanpercentile(data_ref.loc[:, "PREC"], 10)
        ta_avg_ref = data_ref.loc[:, "TA"].mean()
        ta_ipr_ref = np.nanpercentile(data_ref.loc[:, "TA"], 90) - np.nanpercentile(data_ref.loc[:, "TA"], 10)
        pet_avg_ref = data_ref.loc[:, "PET"].mean()
        pet_ipr_ref = np.nanpercentile(data_ref.loc[:, "PET"], 90) - np.nanpercentile(data_ref.loc[:, "PET"], 10)

        prec_avg_nf = data_nf.loc[:, "PREC"].mean()
        prec_ipr_nf = np.nanpercentile(data_nf.loc[:, "PREC"], 90) - np.nanpercentile(data_nf.loc[:, "PREC"], 10)
        ta_avg_nf = data_nf.loc[:, "TA"].mean()
        ta_ipr_nf = np.nanpercentile(data_nf.loc[:, "TA"], 90) - np.nanpercentile(data_nf.loc[:, "TA"], 10)
        pet_avg_nf = data_nf.loc[:, "PET"].mean()
        pet_ipr_nf = np.nanpercentile(data_nf.loc[:, "PET"], 90) - np.nanpercentile(data_nf.loc[:, "PET"], 10)

        prec_avg_ff = data_ff.loc[:, "PREC"].mean()
        prec_ipr_ff = np.nanpercentile(data_ff.loc[:, "PREC"], 90) - np.nanpercentile(data_ff.loc[:, "PREC"], 10)
        ta_avg_ff = data_ff.loc[:, "TA"].mean()
        ta_ipr_ff = np.nanpercentile(data_ff.loc[:, "TA"], 90) - np.nanpercentile(data_ff.loc[:, "TA"], 10)
        pet_avg_ff = data_ff.loc[:, "PET"].mean()
        pet_ipr_ff = np.nanpercentile(data_ff.loc[:, "PET"], 90) - np.nanpercentile(data_ff.loc[:, "PET"], 10)

        df_prec = pd.DataFrame(index=[0], columns=["dAvg_nf", "dIPR_nf", "dAvg_ff", "dIPR_ff"])
        df_prec.loc[0, "dAvg_nf"] = (prec_avg_nf - prec_avg_ref) / prec_avg_ref
        df_prec.loc[0, "dAvg_ff"] = (prec_avg_ff - prec_avg_ref) / prec_avg_ref
        df_prec.loc[0, "dIPR_nf"] = (prec_ipr_nf - prec_ipr_ref) / prec_ipr_ref
        df_prec.loc[0, "dIPR_ff"] = (prec_ipr_ff - prec_ipr_ref) / prec_ipr_ref
        df_ta = pd.DataFrame(index=[0], columns=["dAvg_nf", "dIPR_nf", "dAvg_ff", "dIPR_ff"])
        df_ta.loc[0, "dAvg_nf"] = (ta_avg_nf - ta_avg_ref) / ta_avg_ref
        df_ta.loc[0, "dAvg_ff"] = (ta_avg_ff - ta_avg_ref) / ta_avg_ref
        df_ta.loc[0, "dIPR_nf"] = (ta_ipr_nf - ta_ipr_ref) / ta_ipr_ref
        df_ta.loc[0, "dIPR_ff"] = (ta_ipr_ff - ta_ipr_ref) / ta_ipr_ref
        df_pet = pd.DataFrame(index=[0], columns=["dAvg_nf", "dIPR_nf", "dAvg_ff", "dIPR_ff"])
        df_pet.loc[0, "dAvg_nf"] = (pet_avg_nf - pet_avg_ref) / pet_avg_ref
        df_pet.loc[0, "dAvg_ff"] = (pet_avg_ff - pet_avg_ref) / pet_avg_ref
        df_pet.loc[0, "dIPR_nf"] = (pet_ipr_nf - pet_ipr_ref) / pet_ipr_ref
        df_pet.loc[0, "dIPR_ff"] = (pet_ipr_ff - pet_ipr_ref) / pet_ipr_ref

        dict_deltas_climate[station_label1[station_id]][cm]["prec"] = {}
        dict_deltas_climate[station_label1[station_id]][cm]["prec"] = df_prec
        dict_deltas_climate[station_label1[station_id]][cm]["ta"] = {}
        dict_deltas_climate[station_label1[station_id]][cm]["ta"] = df_ta
        dict_deltas_climate[station_label1[station_id]][cm]["pet"] = {}
        dict_deltas_climate[station_label1[station_id]][cm]["pet"] = df_pet

# Store data (serialize)
file = base_path / "figures" / "delta_changes_climate.pkl"
with open(file, "wb") as handle:
    pickle.dump(dict_deltas_climate, handle, protocol=pickle.HIGHEST_PROTOCOL)

# --- plot delta changes of precipitation and temperature ----------------------
for i, station_id in enumerate(station_ids):
    fig, axs = plt.subplots(1, 1, figsize=(4, 2), sharex=True, sharey=True)
    for cm in cms:
        dta_avg_nf = dict_deltas_climate[station_label1[station_id]][cm]["ta"].loc[0, "dAvg_nf"] * 100
        dprec_avg_nf = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dAvg_nf"] * 100
        axs.scatter(dta_avg_nf, dprec_avg_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
    for cm in cms:
        dta_avg_ff = dict_deltas_climate[station_label1[station_id]][cm]["ta"].loc[0, "dAvg_ff"] * 100
        dprec_avg_ff = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dAvg_ff"] * 100
        axs.scatter(dta_avg_ff, dprec_avg_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
    axs.set_xlabel(r"$\overline{\Delta}$ TA [%]")
    axs.set_ylabel(r"$\overline{\Delta}$ PREC [%]")
    lines, labels = axs[-1].get_legend_handles_labels()
    fig.legend(
        lines[:7],
        labels[:7],
        loc="upper right",
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(1.0, 1.01),
    )
    fig.legend(
        lines[7:],
        labels[7:],
        loc="upper right",
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(1.0, 0.53),
    )
    fig.subplots_adjust(left=0.08, right=0.72, bottom=0.2)
    file = base_path_figs / f"projected_annual_dprec_and_dta_avg_{station_label1[station_id]}.png"
    fig.savefig(file, dpi=300)
    file = base_path_figs / f"projected_annual_dprec_and_dta_avg_{station_label1[station_id]}.pdf"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

for i, station_id in enumerate(station_ids):
    fig, axs = plt.subplots(2, 1, figsize=(6, 2), sharex="row", sharey="row")
    for cm in cms:
        dta_avg_nf = dict_deltas_climate[station_label1[station_id]][cm]["ta"].loc[0, "dAvg_nf"] * 100
        dprec_avg_nf = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dAvg_nf"] * 100
        axs[0].scatter(dta_avg_nf, dprec_avg_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
        dta_ipr_nf = dict_deltas_climate[station_label1[station_id]][cm]["ta"].loc[0, "dIPR_nf"] * 100
        dprec_ipr_nf = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dIPR_nf"] * 100
        axs[1].scatter(dta_ipr_nf, dprec_ipr_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
    for cm in cms:
        dta_avg_ff = dict_deltas_climate[station_label1[station_id]][cm]["ta"].loc[0, "dAvg_ff"] * 100
        dprec_avg_ff = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dAvg_ff"] * 100
        axs[0].scatter(dta_avg_ff, dprec_avg_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
        dta_ipr_ff = dict_deltas_climate[station_label1[station_id]][cm]["ta"].loc[0, "dIPR_ff"] * 100
        dprec_ipr_ff = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dIPR_ff"] * 100
        axs[1].scatter(dta_ipr_ff, dprec_ipr_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
    axs[0].set_xlabel(r"$\overline{\Delta}$ TA [%]")
    axs[1].set_xlabel(r"$\Delta$$IPR$ TA [%]")
    axs[0].set_ylabel(r"$\overline{\Delta}$ PREC [%]")
    axs[1].set_ylabel(r"$\Delta$$IPR$ PREC [%]")
    axs[0].text(
        0.9, 0.9, "(a)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 0].transAxes
    )
    axs[1].text(
        0.9, 0.9, "(b)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 1].transAxes
    )
    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        lines[:7],
        labels[:7],
        loc="upper right",
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(1.0, 0.9),
        title="2030-2059",
    )
    fig.legend(
        lines[7:],
        labels[7:],
        loc="upper right",
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(1.0, 0.64),
        title="2070-2099",
    )
    fig.subplots_adjust(left=0.08, right=0.72, bottom=0.15, wspace=0.1, hspace=0.35)
    file = base_path_figs / f"projected_annual_dprec_and_dta_{station_label1[station_id]}.png"
    fig.savefig(file, dpi=300)
    file = base_path_figs / f"projected_annual_dprec_and_dta_{station_label1[station_id]}.pdf"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

# --- plot delta changes of precipitation and potential evapotranspiration ----------------------
for i, station_id in enumerate(station_ids):
    fig, axs = plt.subplots(2, 1, figsize=(4, 2), sharex="row", sharey="row")
    for cm in cms:
        dpet_avg_nf = dict_deltas_climate[station_label1[station_id]][cm]["pet"].loc[0, "dAvg_nf"] * 100
        dprec_avg_nf = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dAvg_nf"] * 100
        axs[0].scatter(dpet_avg_nf, dprec_avg_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
        dpet_ipr_nf = dict_deltas_climate[station_label1[station_id]][cm]["pet"].loc[0, "dIPR_nf"] * 100
        dprec_ipr_nf = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dIPR_nf"] * 100
        axs[1].scatter(dpet_ipr_nf, dprec_ipr_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
    for cm in cms:
        dpet_avg_ff = dict_deltas_climate[station_label1[station_id]][cm]["pet"].loc[0, "dAvg_ff"] * 100
        dprec_avg_ff = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dAvg_ff"] * 100
        axs[0].scatter(dpet_avg_ff, dprec_avg_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
        dpet_ipr_ff = dict_deltas_climate[station_label1[station_id]][cm]["pet"].loc[0, "dIPR_ff"] * 100
        dprec_ipr_ff = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dIPR_ff"] * 100
        axs[1].scatter(dpet_ipr_ff, dprec_ipr_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
    axs[0].set_ylabel("")
    axs[0].set_xlabel(r"$\overline{\Delta}$ PET [%]")
    axs[1].set_xlabel(r"$\Delta$$IPR$ PET [%]")
    axs[0].set_ylabel(r"$\overline{\Delta}$ PREC [%]")
    axs[1].set_ylabel(r"$\Delta$$IPR$ PREC [%]")
    axs[0].text(
        0.9, 0.9, "(a)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 0].transAxes
    )
    axs[1].text(
        0.9, 0.9, "(b)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 1].transAxes
    )
    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        lines[:7],
        labels[:7],
        loc="upper right",
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(1.0, 0.9),
        title="2030-2059",
    )
    fig.legend(
        lines[7:],
        labels[7:],
        loc="upper right",
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(1.0, 0.64),
        title="2070-2099",
    )
    fig.subplots_adjust(left=0.08, right=0.72, bottom=0.15, wspace=0.1, hspace=0.4)
    file = base_path_figs / f"projected_annual_dprec_and_dpet_{station_label1[station_id]}.png"
    fig.savefig(file, dpi=300)
    file = base_path_figs / f"projected_annual_dprec_and_dpet_{station_label1[station_id]}.pdf"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)
