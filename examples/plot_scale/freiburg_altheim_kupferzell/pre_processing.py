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

periods = ["hist", "future"]
station_ids = [1443, 2787, 4189]
cms = [
    "CCCma-CanESM2_CCLM4-8-17",
    "ICHEC-EC-EARTH_CCLM4-8-17",
    "ICHEC-EC-EARTH_RCA4",
    "IPSL-IPSL-CM5A-MR_RCA4",
    "MIROC-MIROC5_CCLM4-8-17",
    "MPI-M-MPI-ESM-LR_CCLM4-8-17",
    "MPI-M-MPI-ESM-LR_RCA4",
]
station_label = {
    1443: "Freiburg",
    2787: "Kupferzell",
    4189: "Altheim",
}
station_label1 = {
    1443: "freiburg",
    2787: "kupferzell",
    4189: "altheim",
}

color = {
    "CCCma-CanESM2_CCLM4-8-17_hist": "#eff3ff",
    "ICHEC-EC-EARTH_CCLM4-8-17_hist": "#c6dbef",
    "ICHEC-EC-EARTH_RCA4_hist": "#9ecae1",
    "IPSL-IPSL-CM5A-MR_RCA4_hist": "#6baed6",
    "MIROC-MIROC5_CCLM4-8-17_hist": "#4292c6",
    "MPI-M-MPI-ESM-LR_CCLM4-8-17_hist": "#2171b5",
    "MPI-M-MPI-ESM-LR_RCA4_hist": "#084594",
    "CCCma-CanESM2_CCLM4-8-17_future": "#fee5d9",
    "ICHEC-EC-EARTH_CCLM4-8-17_future": "#fcbba1",
    "ICHEC-EC-EARTH_RCA4_future": "#fc9272",
    "IPSL-IPSL-CM5A-MR_RCA4_future": "#fb6a4a",
    "MIROC-MIROC5_CCLM4-8-17_future": "#ef3b2c",
    "MPI-M-MPI-ESM-LR_CCLM4-8-17_future": "#cb181d",
    "MPI-M-MPI-ESM-LR_RCA4_future": "#99000d",
}

label = {
    "CCCma-CanESM2_CCLM4-8-17": "CCCma-CanESM2 CCLM4.8.17",
    "ICHEC-EC-EARTH_CCLM4-8-17": "ICHEC-EC-EARTH CCLM4.8.17",
    "ICHEC-EC-EARTH_RCA4": "ICHEC-EC-EARTH RCA4",
    "IPSL-IPSL-CM5A-MR_RCA4": "IPSL-IPSL-CM5A-MR RCA4",
    "MIROC-MIROC5_CCLM4-8-17": "MIROC-MIROC5 CCLM4.8.17",
    "MPI-M-MPI-ESM-LR_CCLM4-8-17": "MPI-M-MPI-ESM-LR CCLM4.8.17",
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
idx_annually_1985_2100 = pd.date_range(start="1985-01-01", end="2099-12-31", freq="y")
idx_annually_2003_2005 = pd.date_range(start="2003-01-01", end="2005-12-31", freq="y")
idx_annually_2016_2021 = pd.date_range(start="2016-01-01", end="2021-12-31", freq="y")
idx_annually_1994_2005 = pd.date_range(start="1994-01-01", end="2005-12-31", freq="y")
idx_annually_1985_2015 = pd.date_range(start="1985-01-01", end="2014-12-31", freq="y")
idx_annually_2030_2060 = pd.date_range(start="2030-01-01", end="2059-12-31", freq="y")
idx_annually_2070_2100 = pd.date_range(start="2070-01-01", end="2099-12-31", freq="y")

idx_seasonally_1985_2100 = pd.date_range(start="1985-03-01", end="2099-09-01", freq="3MS")
idx_seasonally_2003_2005 = pd.date_range(start="2003-03-01", end="2005-09-01", freq="3MS")
idx_seasonally_2016_2021 = pd.date_range(start="2016-03-01", end="2021-09-01", freq="3MS")
idx_seasonally_1994_2005 = pd.date_range(start="1994-03-01", end="2005-09-01", freq="3MS")
idx_seasonally_1985_2015 = pd.date_range(start="1985-03-01", end="2015-09-01", freq="3MS")
idx_seasonally_2030_2060 = pd.date_range(start="2030-03-01", end="2059-09-01", freq="3MS")
idx_seasonally_2070_2100 = pd.date_range(start="2070-03-01", end="2099-09-01", freq="3MS")

idx_daily_1985_2100 = pd.date_range(start="1985-01-01", end="2099-12-31", freq="d")
idx_daily_2003_2005 = pd.date_range(start="2003-01-01", end="2005-12-31", freq="d")
idx_daily_2016_2021 = pd.date_range(start="2016-01-01", end="2021-12-31", freq="d")
idx_daily_1994_2005 = pd.date_range(start="1994-01-01", end="2005-12-31", freq="d")
idx_daily_1985_2015 = pd.date_range(start="1985-01-01", end="2014-12-31", freq="d")
idx_daily_2030_2060 = pd.date_range(start="2030-01-01", end="2059-12-31", freq="d")
idx_daily_2070_2100 = pd.date_range(start="2070-01-01", end="2099-12-31", freq="d")

idx_hourly_1985_2100 = pd.date_range(start="1985-01-01 00:00:00", end="2099-12-31 23:00:00", freq="h")
idx_hourly_2003_2005 = pd.date_range(start="2003-01-01 00:00:00", end="2004-12-31 23:00:00", freq="h")
idx_hourly_2016_2021 = pd.date_range(start="2016-01-01 00:00:00", end="2021-12-31 23:00:00", freq="h")
idx_hourly_1994_2005 = pd.date_range(start="1994-01-01 00:00:00", end="2004-12-31 23:00:00", freq="h")
idx_hourly_1985_2015 = pd.date_range(start="1985-01-01 00:00:00", end="2014-12-31 23:00:00", freq="h")
idx_hourly_2030_2060 = pd.date_range(start="2030-01-01 00:00:00", end="2059-12-31 23:00:00", freq="h")
idx_hourly_2070_2100 = pd.date_range(start="2070-01-01 00:00:00", end="2099-12-31 23:00:00", freq="h")

idx_3hourly_1985_2100c = pd.date_range(start="1985-01-01 01:30:00", end="2099-12-31 22:30:00", freq="3h")
idx_3hourly_1985_2100 = pd.date_range(start="1985-01-01 00:00:00", end="2099-12-31 23:00:00", freq="3h")

idx_10mins_1985_2100 = pd.date_range(start="1985-01-01 00:00:00", end="2100-12-31 23:50:00", freq="10T")
idx_10mins_2003_2005 = pd.date_range(start="2003-01-01 00:00:00", end="2004-12-31 23:50:00", freq="10T")
idx_10mins_2016_2021 = pd.date_range(start="2016-01-01 00:00:00", end="2021-12-31 23:50:00", freq="10T")
idx_10mins_1994_2005 = pd.date_range(start="1994-01-01 00:00:00", end="2004-12-31 23:50:00", freq="10T")
idx_10mins_1985_2015 = pd.date_range(start="1985-01-01 00:00:00", end="2004-12-31 23:50:00", freq="10T")
idx_10mins_2030_2060 = pd.date_range(start="2030-01-01 00:00:00", end="2059-12-31 23:50:00", freq="10T")
idx_10mins_2070_2100 = pd.date_range(start="2070-01-01 00:00:00", end="2099-12-31 23:50:00", freq="10T")

# --- load bias-corrected climate projections ---------------------------------------------------
dict_bc_meteo_daily = {}
for station_id in station_ids:
    if station_id == 1443:
        elevation = 236
    elif station_id == 2787:
        elevation = 340
    elif station_id == 4189:
        elevation = 541
    dict_bc_meteo_daily[station_id] = {}
    for cm in cms:
        dict_bc_meteo_daily[station_id][cm] = {}
        file = (
            base_path / "climate_projections" / "data" / "daily" / f"BC-Neu-pcmgthr-MBCn_{cm}_DWD_{station_id}_hist.csv"
        )
        data_hist = pd.read_csv(file, sep=",", index_col=0)
        data_hist.index = pd.to_datetime(data_hist.index, format="%Y-%m-%d")
        data_hist.columns = ["PREC", "TA", "TAD", "RS"]
        file = (
            base_path
            / "climate_projections"
            / "data"
            / "daily"
            / f"BC-Neu-pcmgthr-MBCn_{cm}_DWD_{station_id}_future.csv"
        )
        data_future = pd.read_csv(file, sep=",", index_col=0)
        data_future.index = pd.to_datetime(data_future.index, format="%Y-%m-%d")
        data_future.columns = ["PREC", "TA", "TAD", "RS"]
        data_hist_future = pd.concat([data_hist, data_future])
        data_1985_2100 = pd.DataFrame(index=idx_daily_1985_2100)
        data_1985_2100 = data_1985_2100.join(data_hist_future)
        # fill NaNs at 29th February
        data_1985_2100.loc[:, "TA"] = data_1985_2100["TA"].interpolate()
        data_1985_2100.loc[:, "RS"] = data_1985_2100["RS"].interpolate()
        data_1985_2100.loc[:, "PET"] = _calc_pet_with_makkink(
            data_1985_2100.loc[:, "RS"].values, data_1985_2100.loc[:, "TA"].values, elevation
        )
        dict_bc_meteo_daily[station_id][cm]["1985-2100"] = data_1985_2100

        for period in periods:
            file = (
                base_path
                / "climate_projections"
                / "data"
                / "daily"
                / f"BC-Neu-pcmgthr-MBCn_{cm}_DWD_{station_id}_{period}.csv"
            )
            data = pd.read_csv(file, sep=",", index_col=0)
            data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
            data.columns = ["PREC", "TA", "TAD", "RS"]
            if period == "hist":
                data_1985_2015 = pd.DataFrame(index=idx_daily_1985_2015)
                data_1985_2015 = data_1985_2015.join(data)
                # fill NaNs at 29th February
                data_1985_2015.loc[:, "TA"] = data_1985_2015["TA"].interpolate()
                data_1985_2015.loc[:, "RS"] = data_1985_2015["RS"].interpolate()
                data_1985_2015.loc[:, "PET"] = _calc_pet_with_makkink(
                    data_1985_2015.loc[:, "RS"].values, data_1985_2015.loc[:, "TA"].values, elevation
                )
                dict_bc_meteo_daily[station_id][cm]["1985-2015"] = data_1985_2015
            elif period == "future":
                data_2016_2021 = pd.DataFrame(index=idx_daily_2016_2021)
                data_2016_2021 = data_2016_2021.join(data)
                # fill NaNs at 29th February
                data_2016_2021.loc[:, "TA"] = data_2016_2021["TA"].interpolate()
                data_2016_2021.loc[:, "RS"] = data_2016_2021["RS"].interpolate()
                data_2016_2021.loc[:, "PET"] = _calc_pet_with_makkink(
                    data_2016_2021.loc[:, "RS"].values, data_2016_2021.loc[:, "TA"].values, elevation
                )
                data_2030_2060 = pd.DataFrame(index=idx_daily_2030_2060)
                data_2030_2060 = data_2030_2060.join(data)
                # fill NaNs at 29th February
                data_2030_2060.loc[:, "TA"] = data_2030_2060["TA"].interpolate()
                data_2030_2060.loc[:, "RS"] = data_2030_2060["RS"].interpolate()
                data_2030_2060.loc[:, "PET"] = _calc_pet_with_makkink(
                    data_2030_2060.loc[:, "RS"].values, data_2030_2060.loc[:, "TA"].values, elevation
                )
                data_2070_2100 = pd.DataFrame(index=idx_daily_2070_2100)
                data_2070_2100 = data_2070_2100.join(data)
                # fill NaNs at 29th February
                data_2070_2100.loc[:, "TA"] = data_2070_2100["TA"].interpolate()
                data_2070_2100.loc[:, "RS"] = data_2070_2100["RS"].interpolate()
                data_2070_2100.loc[:, "PET"] = _calc_pet_with_makkink(
                    data_2070_2100.loc[:, "RS"].values, data_2070_2100.loc[:, "TA"].values, elevation
                )
                dict_bc_meteo_daily[station_id][cm]["2016-2021"] = data_2016_2021
                dict_bc_meteo_daily[station_id][cm]["2030-2060"] = data_2030_2060
                dict_bc_meteo_daily[station_id][cm]["2070-2100"] = data_2070_2100

    dict_bc_meteo_daily[station_id]["observed"] = {}
    file = base_path / "input" / f"{station_label1[station_id]}" / "observed" / "2016-2021" / "PREC.txt"
    data_prec = pd.read_csv(file, sep="\t")
    data_prec.index = idx_10mins_2016_2021
    file = base_path / "input" / f"{station_label1[station_id]}" / "observed" / "2016-2021" / "TA.txt"
    data_ta = pd.read_csv(file, sep="\t")
    file = base_path / "input" / f"{station_label1[station_id]}" / "observed" / "2016-2021" / "PET.txt"
    data_pet = pd.read_csv(file, sep="\t")
    data_2016_2021 = pd.DataFrame(index=idx_daily_2016_2021, columns=["PREC", "TA", "PET"])
    data_2016_2021.loc[:, "PREC"] = data_prec.loc[:, "PREC"].resample("1D").sum().values
    data_2016_2021.loc[:, "TA"] = data_ta.loc[:, "TA"].values
    data_2016_2021.loc[:, "PET"] = data_pet.loc[:, "PET"].values
    dict_bc_meteo_daily[station_id]["observed"]["2016-2021"] = data_2016_2021

# --- projected annual air temperature, potential evapotranspiration and precipitation -----------
dict_meteo_ann = {}
for station_id in station_ids:
    if station_id == 1443:
        elevation = 236
    elif station_id == 2787:
        elevation = 340
    elif station_id == 4189:
        elevation = 541
    dict_meteo_ann[station_id] = {}
    for cm in cms:
        dict_meteo_ann[station_id][cm] = {}
        for period in periods:
            file = file = (
                base_path
                / "climate_projections"
                / "data"
                / "daily"
                / f"BC-Neu-pcmgthr-MBCn_{cm}_DWD_{station_id}_{period}.csv"
            )
            data = pd.read_csv(file, sep=",", index_col=0)
            data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
            data.columns = ["PREC", "TA", "TAD", "RS"]
            data.loc[:, "PET"] = _calc_pet_with_makkink(data.loc[:, "RS"].values, data.loc[:, "TA"].values, elevation)
            data_ann = (
                data.loc[:, "PREC"]
                .resample("1Y")
                .sum()
                .to_frame()
                .join(data.loc[:, "TA"].resample("1Y").mean().to_frame())
                .join(data.loc[:, "PET"].resample("1Y").sum().to_frame())
            )
            dict_meteo_ann[station_id][cm][period] = data_ann
            if period == "hist":
                data_ann_1985_2015 = pd.DataFrame(index=idx_annually_1985_2015)
                data_ann_1985_2015 = data_ann_1985_2015.join(data_ann)
                dict_meteo_ann[station_id][cm]["1985-2015"] = data_ann_1985_2015

            elif period == "future":
                data_ann_2030_2060 = pd.DataFrame(index=idx_annually_2030_2060)
                data_ann_2030_2060 = data_ann_2030_2060.join(data_ann)
                data_ann_2070_2100 = pd.DataFrame(index=idx_annually_2070_2100)
                data_ann_2070_2100 = data_ann_2070_2100.join(data_ann)
                dict_meteo_ann[station_id][cm]["2030-2060"] = data_ann_2030_2060
                dict_meteo_ann[station_id][cm]["2070-2100"] = data_ann_2070_2100

# --- projected seasonal air temperature, potential evapotranspiration and precipitation ---------------
dict_meteo_seas = {}
for station_id in station_ids:
    if station_id == 1443:
        elevation = 236
    elif station_id == 2787:
        elevation = 340
    elif station_id == 4189:
        elevation = 541
    dict_meteo_seas[station_id] = {}
    for cm in cms:
        dict_meteo_seas[station_id][cm] = {}
        for period in periods:
            file = file = (
                base_path
                / "climate_projections"
                / "data"
                / "daily"
                / f"BC-Neu-pcmgthr-MBCn_{cm}_DWD_{station_id}_{period}.csv"
            )
            data = pd.read_csv(file, sep=",", index_col=0)
            data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
            data.columns = ["PREC", "TA", "TAD", "RS"]
            data.loc[:, "PET"] = _calc_pet_with_makkink(data.loc[:, "RS"].values, data.loc[:, "TA"].values, elevation)
            data_seas = (
                data.loc[:, "PREC"]
                .groupby(pd.Grouper(freq="QS-DEC"))
                .sum()
                .to_frame()
                .join(data.loc[:, "TA"].groupby(pd.Grouper(freq="QS-DEC")).mean().to_frame())
                .join(data.loc[:, "PET"].groupby(pd.Grouper(freq="QS-DEC")).mean().to_frame())
            )

            dict_meteo_seas[station_id][cm][period] = data_seas
            if period == "hist":
                data_seas_1985_2015 = pd.DataFrame(index=idx_seasonally_1985_2015)
                data_seas_1985_2015 = data_seas_1985_2015.join(data_seas)
                dict_meteo_seas[station_id][cm]["1985-2015"] = data_seas_1985_2015

            elif period == "future":
                data_seas_2030_2060 = pd.DataFrame(index=idx_seasonally_2030_2060)
                data_seas_2030_2060 = data_seas_2030_2060.join(data_seas)
                data_seas_2070_2100 = pd.DataFrame(index=idx_seasonally_2070_2100)
                data_seas_2070_2100 = data_seas_2070_2100.join(data_seas)
                dict_meteo_seas[station_id][cm]["2030-2060"] = data_seas_2030_2060
                dict_meteo_seas[station_id][cm]["2070-2100"] = data_seas_2070_2100

# --- observed annual air temperature, potential evapotranspiration and precipitation --------------
for station_id in station_ids:
    for cm in cms:
        file = file = (
            base_path
            / "climate_projections"
            / "data"
            / "daily"
            / f"BC-Neu-pcmgthr-MBCn_{cm}_DWD_{station_id}_future.csv"
        )
        data = pd.read_csv(file, sep=",", index_col=0)
        data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
        data.columns = ["PREC", "TA", "TAD", "RS"]
        data.loc[:, "PET"] = _calc_pet_with_makkink(data.loc[:, "RS"].values, data.loc[:, "TA"].values, elevation)
        data_2016_2021 = pd.DataFrame(index=idx_daily_2016_2021)
        data_2016_2021 = data_2016_2021.join(data)
        data_ann = (
            data_2016_2021.loc[:, "PREC"]
            .resample("1Y")
            .sum()
            .to_frame()
            .join(data_2016_2021.loc[:, "TA"].resample("1Y").mean().to_frame())
            .join(data_2016_2021.loc[:, "PET"].resample("1Y").sum().to_frame())
        )
        dict_meteo_ann[station_id][cm]["2016-2021"] = data_ann
    data_2016_2021 = dict_bc_meteo_daily[station_id]["observed"]["2016-2021"]
    data_ann = (
        data_2016_2021.loc[:, "PREC"]
        .resample("1Y")
        .sum()
        .to_frame()
        .join(data_2016_2021.loc[:, "TA"].resample("1Y").mean().to_frame())
    )
    dict_meteo_ann[station_id]["observed"] = {}
    dict_meteo_ann[station_id]["observed"]["2016-2021"] = data_ann

# --- compare projected annual air temperature and precipitation ------------------------
for station_id in station_ids:
    fig, axs = plt.subplots(1, 1, figsize=(5, 3))
    for period in periods:
        for cm in cms:
            data = dict_meteo_ann[station_id][cm][period]
            prec_avg = data.loc[:, "PREC"].mean()
            prec_std = data.loc[:, "PREC"].std()
            prec_ipr = np.percentile(data.loc[:, "PREC"], 90) - np.percentile(data.loc[:, "PREC"], 10)
            ta_avg = data.loc[:, "TA"].mean()
            ta_std = data.loc[:, "TA"].std()
            ta_ipr = np.percentile(data.loc[:, "TA"], 90) - np.percentile(data.loc[:, "TA"], 10)
            axs.errorbar(
                ta_avg,
                prec_avg,
                xerr=ta_std,
                yerr=prec_std,
                fmt="o",
                label=label[cm],
                color=color[f"{cm}_{period}"],
                ms=2.5,
            )
    axs.set_ylabel("Mean annual precipitation [mm]")
    axs.set_xlabel("Mean annual air temperature [°C]")
    axs.set_title(f"{station_label[station_id]} (station ID: {station_id})")
    lines, labels = axs.get_legend_handles_labels()
    fig.legend(
        lines[:7],
        labels[:7],
        loc="upper right",
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(1.01, 0.9),
        title="1985-2015",
    )
    fig.legend(
        lines[7:],
        labels[7:],
        loc="upper right",
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(1.01, 0.55),
        title="2016-2100",
    )
    fig.subplots_adjust(right=0.68)
    file = base_path_figs / f"projected_annual_prec_and_ta_{station_label[station_id]}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

for station_id in station_ids:
    fig, axs = plt.subplots(1, 1, figsize=(5, 3))
    data = dict_meteo_ann[station_id]["observed"]["2016-2021"]
    prec_avg = data.loc[:, "PREC"].mean()
    ta_avg = data.loc[:, "TA"].mean()
    axs.scatter(ta_avg, prec_avg, label="observed", color="blue", s=5)
    for cm in cms:
        data = dict_meteo_ann[station_id][cm]["2016-2021"]
        prec_avg = data.loc[:, "PREC"].mean()
        ta_avg = data.loc[:, "TA"].mean()
        axs.scatter(ta_avg, prec_avg, label=label[cm], color=color[f"{cm}_future"], s=4)
    axs.set_ylabel("Mean annual precipitation [mm]")
    axs.set_xlabel("Mean annual air temperature [°C]")
    axs.set_title(f"{station_label[station_id]} (station ID: {station_id})")
    lines, labels = axs.get_legend_handles_labels()
    fig.legend(
        lines, labels, loc="upper right", fontsize=6, frameon=False, bbox_to_anchor=(1.01, 0.9), title="2016-2021"
    )
    fig.subplots_adjust(right=0.68)
    file = base_path_figs / f"annual_prec_and_ta_{station_label[station_id]}_obs_and_proj.png"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

# --- add minimum TA and maximum TA --------------------------------
stations = ["freiburg", "weingarten", "ingelfingen"]
for station, station_id in zip(stations, station_ids):
    file = base_path / "dwd" / f"{station}" / "meteo.txt"
    data = pd.read_csv(file, sep=";", na_values=-999)
    data.index = pd.to_datetime(data["MESS_DATUM"], format="%Y%m%d")
    data_2016_2021 = pd.DataFrame(index=idx_daily_2016_2021)
    data_2016_2021 = data_2016_2021.join(data)
    df_TA_min_max = pd.DataFrame(index=idx_daily_2016_2021)
    if station == "freiburg":
        df_TA_min_max.loc[:, "TA_min"] = data_2016_2021[" TNK"].values
        df_TA_min_max.loc[:, "TA_max"] = data_2016_2021[" TXK"].values
    # correct for altitude effect since values are obtained from nearby station
    elif station == "weingarten":
        df_TA_min_max.loc[:, "TA_min"] = data_2016_2021[" TNK"].values + ((440 - 340) / 100) * 0.65
        df_TA_min_max.loc[:, "TA_max"] = data_2016_2021[" TXK"].values + ((440 - 340) / 100) * 0.65
    elif station == "ingelfingen":
        df_TA_min_max.loc[:, "TA_min"] = data_2016_2021[" TNK"].values + ((385 - 541) / 100) * 0.65
        df_TA_min_max.loc[:, "TA_max"] = data_2016_2021[" TXK"].values + ((385 - 541) / 100) * 0.65
    meteo_2016_2021 = dict_bc_meteo_daily[station_id]["observed"]["2016-2021"]
    meteo_2016_2021 = meteo_2016_2021.join(df_TA_min_max).astype("float64")
    dict_bc_meteo_daily[station_id]["observed"]["2016-2021"] = meteo_2016_2021

for station_id in station_ids:
    for cm in ["MPI-M-MPI-ESM-LR_RCA4", "CCCma-CanESM2_CCLM4-8-17"]:
        # uncorrected daily minimum and maximum air temperature
        file = (
            base_path
            / "climate_projections"
            / "data"
            / "daily"
            / f"tmin-max-daily_FullTs_{cm}_station-DWD_{station_id}.csv"
        )
        data = pd.read_csv(file, sep=",", index_col=2)
        data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
        data = data.loc["1985":"2100", :]
        cond = data["Var"].values == "tasmax"
        data_ta_max = data.loc[cond, "Center"]
        cond = data["Var"].values == "tasmin"
        data_ta_min = data.loc[cond, "Center"]
        data1 = pd.DataFrame(index=data_ta_min.index, columns=["TA_min", "TA_max"])
        data1.loc[:, "TA_min"] = data_ta_min.values
        data1.loc[:, "TA_max"] = data_ta_max.values
        data = pd.DataFrame(index=idx_daily_1985_2100)
        data = data.join(data1)
        # uncorrected daily average air temperature
        file = base_path / "climate_projections" / "data" / "daily" / f"pcmgthr_{cm}_DWD_{station_id}_hist.csv"
        data_hist = pd.read_csv(file, sep=",", index_col=0)
        data_hist.index = pd.to_datetime(data_hist.index, format="%Y-%m-%d")
        data_hist.columns = ["PREC", "TA", "TAD", "RS"]
        file = base_path / "climate_projections" / "data" / "daily" / f"pcmgthr_{cm}_DWD_{station_id}_future.csv"
        data_future = pd.read_csv(file, sep=",", index_col=0)
        data_future.index = pd.to_datetime(data_future.index, format="%Y-%m-%d")
        data_future.columns = ["PREC", "TA", "TAD", "RS"]
        data_hist_future = pd.concat([data_hist, data_future])
        data_1985_2100 = pd.DataFrame(index=idx_daily_1985_2100)
        data_1985_2100 = data_1985_2100.join(data_hist_future)
        # fill NaNs at 29th February
        data_1985_2100.loc[:, "TA"] = data_1985_2100["TA"].interpolate()
        data_1985_2100.loc[:, "RS"] = data_1985_2100["RS"].interpolate()
        data1 = data1.join(data_1985_2100.loc[:, "TA"].to_frame())
        data = pd.DataFrame(index=idx_daily_1985_2100)
        data = data.join(data1)
        # bias correction daily minimum and maximum air temperature
        # with difference between bias-corrected daily average air temperature and uncorrected air temperature
        data.loc[:, "scale"] = (
            dict_bc_meteo_daily[station_id][cm]["1985-2100"].loc[:, "TA"] - data_1985_2100.loc[:, "TA"]
        )
        data.loc[:, "TA_min"] = data.loc[:, "TA_min"] + data.loc[:, "scale"]
        data.loc[:, "TA_max"] = data.loc[:, "TA_max"] + data.loc[:, "scale"]
        data = data.loc[:, ["TA_min", "TA_max"]]
        data = data.astype("float64")

        data_1985_2015 = dict_bc_meteo_daily[station_id][cm]["1985-2015"]
        data_1985_2015 = data_1985_2015.join(data)
        # fill NaNs at 29th February
        data_1985_2015.loc[:, "TA_min"] = data_1985_2015["TA_min"].interpolate()
        data_1985_2015.loc[:, "TA_max"] = data_1985_2015["TA_max"].interpolate()
        dict_bc_meteo_daily[station_id][cm]["1985-2015"] = data_1985_2015
        data_2016_2021 = dict_bc_meteo_daily[station_id][cm]["2016-2021"]
        data_2016_2021 = data_2016_2021.join(data)
        # fill NaNs at 29th February
        data_2016_2021.loc[:, "TA_min"] = data_2016_2021["TA_min"].interpolate()
        data_2016_2021.loc[:, "TA_max"] = data_2016_2021["TA_max"].interpolate()
        data_2030_2060 = dict_bc_meteo_daily[station_id][cm]["2030-2060"]
        data_2030_2060 = data_2030_2060.join(data)
        # fill NaNs at 29th February
        data_2030_2060.loc[:, "TA_min"] = data_2030_2060["TA_min"].interpolate()
        data_2030_2060.loc[:, "TA_max"] = data_2030_2060["TA_max"].interpolate()
        data_2070_2100 = dict_bc_meteo_daily[station_id][cm]["2070-2100"]
        data_2070_2100 = data_2070_2100.join(data)
        # fill NaNs at 29th February
        data_2070_2100.loc[:, "TA_min"] = data_2070_2100["TA_min"].interpolate()
        data_2070_2100.loc[:, "TA_max"] = data_2070_2100["TA_max"].interpolate()
        dict_bc_meteo_daily[station_id][cm]["2016-2021"] = data_2016_2021
        dict_bc_meteo_daily[station_id][cm]["2030-2060"] = data_2030_2060
        dict_bc_meteo_daily[station_id][cm]["2070-2100"] = data_2070_2100
        dict_bc_meteo_daily[station_id][cm]["1985-2100"] = data_1985_2100

# --- downscale uncorrected precipitation to 10 minutes ------------------------
dict_precip_10mins = {}
cm = "MPI-M-MPI-ESM-LR_RCA4"
for station_id in station_ids:
    dict_precip_10mins[station_id] = {}
    dict_precip_10mins[station_id][cm] = {}
    file = (
        base_path
        / "climate_projections"
        / "data"
        / "subdaily"
        / f"pr_subdaily_FullTs_{cm}_station-DWD_{station_id}_new.csv"
    )
    data = pd.read_csv(file, sep=",", index_col=0)

    file = base_path / "climate_projections" / "data" / "subdaily" / "datetime_MPI-RCA1hr.txt"
    data_idx = pd.read_csv(file, sep=";")
    data.index = pd.to_datetime(data_idx.iloc[:, 0].astype(str).values, format="%Y-%m-%d %H:%M")
    data = data.loc["1985":"2100", :]
    data_hourly = pd.DataFrame(index=idx_hourly_1985_2100)
    data_hourly.loc[:, "PREC_hourly"] = data.loc["1985":"2099", "Center"].values
    # resample to daily precipitation
    data_daily = data_hourly.resample("1D").sum()
    data_daily.columns = ["PREC_daily"]

    data_10mins = pd.DataFrame(index=idx_10mins_1985_2100)
    data_10mins = data_10mins.join([data_daily, data_hourly])
    data_10mins = data_10mins.ffill()
    # donwnscale hourly precipitation by linear interpolation
    data_10mins.loc[:, "PREC"] = data_10mins.loc[:, "PREC_hourly"] / 6
    # scaling factor derived from uncorrected precipitation for downscaling of daily bias-corrected precipitation
    data_10mins.loc[:, "scale"] = data_10mins.loc[:, "PREC"] / data_10mins.loc[:, "PREC_daily"]
    data_10mins = data_10mins.fillna(0)

    data_1985_2015 = pd.DataFrame(index=idx_10mins_1985_2015)
    data_1985_2015 = data_1985_2015.join(data_10mins)
    dict_precip_10mins[station_id][cm]["1985-2015"] = data_1985_2015
    data_2016_2021 = pd.DataFrame(index=idx_10mins_2016_2021)
    data_2016_2021 = data_2016_2021.join(data_10mins)
    data_2030_2060 = pd.DataFrame(index=idx_10mins_2030_2060)
    data_2030_2060 = data_2030_2060.join(data_10mins)
    data_2070_2100 = pd.DataFrame(index=idx_10mins_2070_2100)
    data_2070_2100 = data_2070_2100.join(data_10mins)
    dict_precip_10mins[station_id][cm]["2016-2021"] = data_2016_2021
    dict_precip_10mins[station_id][cm]["2030-2060"] = data_2030_2060
    dict_precip_10mins[station_id][cm]["2070-2100"] = data_2070_2100
    dict_precip_10mins[station_id][cm]["1985-2100"] = data_10mins

cm = "CCCma-CanESM2_CCLM4-8-17"
for station_id in station_ids:
    dict_precip_10mins[station_id][cm] = {}
    file = (
        base_path
        / "climate_projections"
        / "data"
        / "subdaily"
        / f"pr_subdaily_FullTs_{cm}_station-DWD_{station_id}_new.csv"
    )
    data = pd.read_csv(file, sep=",", index_col=0)

    file = base_path / "climate_projections" / "data" / "subdaily" / "datetime_CANESM-CLM3hr.txt"
    data_idx = pd.read_csv(file, sep=";")
    data.index = pd.to_datetime(data_idx.iloc[:, 0].astype(str).values, format="%Y-%m-%d %H:%M")
    data = data.loc["1985":"2100", :]
    data_3hourly = pd.DataFrame(index=idx_3hourly_1985_2100c)
    data_3hourly = data_3hourly.join(data.loc[:, "Center"].to_frame())
    data_3hourly.index = idx_3hourly_1985_2100
    data_3hourly.columns = ["PREC_3hourly"]
    data_3hourly.loc[:, "PREC_3hourly"] = data_3hourly.loc[:, "PREC_3hourly"].values
    # fill 29th February in leap years
    data_3hourly = data_3hourly.fillna(0)
    # resample to daily precipitation
    data_daily = data_3hourly.resample("1D").sum()
    data_daily.columns = ["PREC_daily"]

    data_10mins = pd.DataFrame(index=idx_10mins_1985_2100)
    data_10mins = data_10mins.join([data_daily, data_3hourly])
    data_10mins = data_10mins.ffill()
    # donwnscale 3-hourly precipitation by linear interpolation
    data_10mins.loc[:, "PREC"] = data_10mins.loc[:, "PREC_3hourly"] / 18
    # scaling factor derived from uncorrected precipitation for downscaling of daily bias-corrected precipitation
    data_10mins.loc[:, "scale"] = data_10mins.loc[:, "PREC"] / data_10mins.loc[:, "PREC_daily"]
    data_10mins = data_10mins.fillna(0)

    data_1985_2015 = pd.DataFrame(index=idx_10mins_1985_2015)
    data_1985_2015 = data_1985_2015.join(data_10mins)
    dict_precip_10mins[station_id][cm]["1985-2015"] = data_1985_2015
    data_2016_2021 = pd.DataFrame(index=idx_10mins_2016_2021)
    data_2016_2021 = data_2016_2021.join(data_10mins)
    data_2030_2060 = pd.DataFrame(index=idx_10mins_2030_2060)
    data_2030_2060 = data_2030_2060.join(data_10mins)
    data_2070_2100 = pd.DataFrame(index=idx_10mins_2070_2100)
    data_2070_2100 = data_2070_2100.join(data_10mins)
    dict_precip_10mins[station_id][cm]["2016-2021"] = data_2016_2021
    dict_precip_10mins[station_id][cm]["2030-2060"] = data_2030_2060
    dict_precip_10mins[station_id][cm]["2070-2100"] = data_2070_2100
    dict_precip_10mins[station_id][cm]["1985-2100"] = data_10mins

# --- downscale bias-corrected precipitation to 10 minutes ------------------------
dict_bc_precip_10mins = {}
for station_id in station_ids:
    dict_bc_precip_10mins[station_id] = {}
    for cm in ["MPI-M-MPI-ESM-LR_RCA4", "CCCma-CanESM2_CCLM4-8-17"]:
        dict_bc_precip_10mins[station_id][cm] = {}
        data_daily = dict_bc_meteo_daily[station_id][cm]["1985-2100"].loc[:, "PREC"].to_frame()

        data_10mins = pd.DataFrame(index=idx_10mins_1985_2100)
        # donwnscale daily precipitation by subdaily scaling factors
        data_10mins = data_10mins.join(data_daily)
        data_10mins = data_10mins.ffill()
        data_10mins.loc[:, "PREC"] = (
            data_10mins.loc[:, "PREC"] * dict_precip_10mins[station_id][cm]["1985-2100"].loc[:, "scale"]
        )
        # replace numerical artefacts
        cond0 = data_10mins["PREC"] < 0.001
        data_10mins.loc[cond0, "PREC"] = 0

        data_1985_2015 = pd.DataFrame(index=idx_10mins_1985_2015)
        data_1985_2015 = data_1985_2015.join(data_10mins)
        dict_bc_precip_10mins[station_id][cm]["1985-2015"] = data_1985_2015
        data_2016_2021 = pd.DataFrame(index=idx_10mins_2016_2021)
        data_2016_2021 = data_2016_2021.join(data_10mins)
        data_2030_2060 = pd.DataFrame(index=idx_10mins_2030_2060)
        data_2030_2060 = data_2030_2060.join(data_10mins)
        data_2070_2100 = pd.DataFrame(index=idx_10mins_2070_2100)
        data_2070_2100 = data_2070_2100.join(data_10mins)
        dict_bc_precip_10mins[station_id][cm]["2016-2021"] = data_2016_2021
        dict_bc_precip_10mins[station_id][cm]["2030-2060"] = data_2030_2060
        dict_bc_precip_10mins[station_id][cm]["2070-2100"] = data_2070_2100

# --- plot time series -------------------------------------
stations = ["freiburg", "kupferzell", "altheim"]
for station, station_id in zip(stations, station_ids):
    for cm in ["MPI-M-MPI-ESM-LR_RCA4", "CCCma-CanESM2_CCLM4-8-17"]:
        for period in ["1985-2015", "2016-2021", "2030-2060", "2070-2100"]:
            data_precip = dict_bc_precip_10mins[station_id][cm][period]
            data_precip_uc = dict_precip_10mins[station_id][cm][period]
            data_meteo = dict_bc_meteo_daily[station_id][cm][period]
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
            file = base_path_figs / f"precip_scale_{station}_{cm}_{period}.png"
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
            file = base_path_figs / f"precip_{station}_{cm}_{period}.png"
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
            file = base_path_figs / f"precip_{station}_{cm}_{period}_cumulated.png"
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
            file = base_path_figs / f"ta_{station}_{cm}_{period}.png"
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
            file = base_path_figs / f"ta_min_max_{station}_{cm}_{period}.png"
            fig.savefig(file, dpi=300)
            plt.close(fig=fig)

# --- write input data to .txt -------------------------------------
stations = ["freiburg", "kupferzell", "altheim"]
for station, station_id in zip(stations, station_ids):
    for cm in ["MPI-M-MPI-ESM-LR_RCA4", "CCCma-CanESM2_CCLM4-8-17"]:
        for period in ["1985-2015", "2016-2021", "2030-2060", "2070-2100"]:
            data_precip = dict_bc_precip_10mins[station_id][cm][period]
            data_meteo = dict_bc_meteo_daily[station_id][cm][period]
            data_ta = data_meteo.loc[:, ["TA", "TA_min", "TA_max"]]
            data_rs = data_meteo.loc[:, ["RS"]]

            path_dir = base_path / "input" / station / cm / period
            if not os.path.exists(path_dir):
                os.mkdir(path_dir)

            idx_10mins = pd.date_range(start=str(data_precip.index[0]), end=str(data_precip.index[-1]), freq="10T")
            idx_daily = pd.date_range(start=str(data_ta.index[0]), end=str(data_ta.index[-1]), freq="d")
            df_PREC = pd.DataFrame(index=idx_10mins, columns=["YYYY", "MM", "DD", "hh", "mm", "PREC"])
            df_PREC["YYYY"] = data_precip.index.year.values
            df_PREC["MM"] = data_precip.index.month.values
            df_PREC["DD"] = data_precip.index.day.values
            df_PREC["hh"] = data_precip.index.hour.values
            df_PREC["mm"] = data_precip.index.minute.values
            df_PREC["PREC"] = data_precip["PREC"].values
            path_txt = path_dir / "PREC.txt"
            df_PREC.to_csv(path_txt, header=True, index=False, sep="\t")
            nas = np.sum(np.isnan(data_precip["PREC"].values))
            print(f"{station}-{cm}-{period}-PREC: {nas}")

            df_TA = pd.DataFrame(index=idx_daily, columns=["YYYY", "MM", "DD", "hh", "mm"])
            df_TA["YYYY"] = data_ta.index.year.values
            df_TA["MM"] = data_ta.index.month.values
            df_TA["DD"] = data_ta.index.day.values
            df_TA["hh"] = data_ta.index.hour.values
            df_TA["mm"] = data_ta.index.minute.values
            df_TA["TA"] = data_ta["TA"].values
            df_TA["TA_min"] = data_ta["TA_min"].values
            df_TA["TA_max"] = data_ta["TA_max"].values
            path_txt = path_dir / "TA.txt"
            df_TA.to_csv(path_txt, header=True, index=False, sep="\t")
            nas = np.sum(np.isnan(data_ta["TA"].values))
            print(f"{station}-{cm}-{period}-TA: {nas} NaN values")
            nas = np.sum(np.isnan(data_ta["TA_min"].values))
            print(f"{station}-{cm}-{period}-TA_min: {nas} NaN values")
            nas = np.sum(np.isnan(data_ta["TA_max"].values))
            print(f"{station}-{cm}-{period}-TA_max: {nas} NaN values")

            df_RS = pd.DataFrame(index=idx_daily, columns=["YYYY", "MM", "DD", "hh", "mm"])
            df_RS["YYYY"] = data_rs.index.year.values
            df_RS["MM"] = data_rs.index.month.values
            df_RS["DD"] = data_rs.index.day.values
            df_RS["hh"] = data_rs.index.hour.values
            df_RS["mm"] = data_rs.index.minute.values
            df_RS["RS"] = data_rs["RS"].values * 0.0864  # convert watt (i.e. J/s) to MJ/day
            path_txt = path_dir / "RS.txt"
            df_RS.to_csv(path_txt, header=True, index=False, sep="\t")
            nas = np.sum(np.isnan(data_rs["RS"].values))
            print(f"{station}-{cm}-{period}-RS: {nas} NaN values")

stations = ["freiburg", "kupferzell", "altheim"]
for station, station_id in zip(stations, station_ids):
    data_meteo = dict_bc_meteo_daily[station_id]["observed"]["2016-2021"]
    data_ta = data_meteo.loc[:, ["TA", "TA_min", "TA_max"]]

    path_dir = base_path / "input" / station / "observed" / "2016-2021"
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)

    idx_daily = pd.date_range(start=str(data_ta.index[0]), end=str(data_ta.index[-1]), freq="d")

    df_TA = pd.DataFrame(index=idx_daily, columns=["YYYY", "MM", "DD", "hh", "mm"])
    df_TA["YYYY"] = data_ta.index.year.values
    df_TA["MM"] = data_ta.index.month.values
    df_TA["DD"] = data_ta.index.day.values
    df_TA["hh"] = data_ta.index.hour.values
    df_TA["mm"] = data_ta.index.minute.values
    df_TA["TA"] = data_ta["TA"].values
    df_TA["TA_min"] = data_ta["TA_min"].values
    df_TA["TA_max"] = data_ta["TA_max"].values
    path_txt = path_dir / "TA.txt"
    df_TA.to_csv(path_txt, header=True, index=False, sep="\t")
    nas = np.sum(np.isnan(data_ta["TA"].values))
    print(f"{station}-observed-{period}-TA: {nas} NaN values")
    nas = np.sum(np.isnan(data_ta["TA_min"].values))
    print(f"{station}-observed-{period}-TA_min: {nas} NaN values")
    nas = np.sum(np.isnan(data_ta["TA_max"].values))
    print(f"{station}-observed-{period}-TA_max: {nas} NaN values")

    data_precip = data_meteo.loc[:, "PREC"].to_frame()
    nas = np.sum(np.isnan(data_precip["PREC"].values))
    print(f"{station}-observed-{period}-PREC: {nas} NaN values")

    data_pet = data_meteo.loc[:, "PET"].to_frame()
    nas = np.sum(np.isnan(data_pet["PET"].values))
    print(f"{station}-observed-{period}-PET: {nas} NaN values")

# calculate total deltas
dict_deltas_climate = {}
for station_id in station_ids:
    dict_deltas_climate[station_label1[station_id]] = {}
    for cm in cms:
        dict_deltas_climate[station_label1[station_id]][cm] = {}
        data_ref = dict_meteo_ann[station_id][cm]["1985-2015"]
        data_nf = dict_meteo_ann[station_id][cm]["2030-2060"]
        data_ff = dict_meteo_ann[station_id][cm]["2070-2100"]
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

# calculate seasonal deltas
dict_deltas_climate_seas = {}
for station_id in station_ids:
    dict_deltas_climate_seas[station_label1[station_id]] = {}
    for cm in cms:
        dict_deltas_climate_seas[station_label1[station_id]][cm] = {}
        data_ref = dict_meteo_seas[station_id][cm]["1985-2015"]
        data_nf = dict_meteo_seas[station_id][cm]["2030-2060"]
        data_ff = dict_meteo_seas[station_id][cm]["2070-2100"]
        df_prec = pd.DataFrame(
            index=["winter", "spring", "summer", "autumn"], columns=["dAvg_nf", "dIPR_nf", "dAvg_ff", "dIPR_ff"]
        )
        df_ta = pd.DataFrame(
            index=["winter", "spring", "summer", "autumn"], columns=["dAvg_nf", "dIPR_nf", "dAvg_ff", "dIPR_ff"]
        )
        for seas, seas_id in zip(["winter", "spring", "summer", "autumn"], [12, 3, 6, 9]):
            prec_avg_ref = data_ref.loc[data_ref.index.month == seas_id, "PREC"].mean()
            prec_ipr_ref = np.nanpercentile(
                data_ref.loc[data_ref.index.month == seas_id, "PREC"], 90
            ) - np.nanpercentile(data_ref.loc[data_ref.index.month == seas_id, "PREC"], 10)
            ta_avg_ref = data_ref.loc[data_ref.index.month == seas_id, "TA"].mean()
            ta_ipr_ref = np.nanpercentile(data_ref.loc[data_ref.index.month == seas_id, "TA"], 90) - np.nanpercentile(
                data_ref.loc[data_ref.index.month == seas_id, "TA"], 10
            )
            pet_avg_ref = data_ref.loc[data_ref.index.month == seas_id, "PET"].mean()
            pet_ipr_ref = np.nanpercentile(data_ref.loc[data_ref.index.month == seas_id, "PET"], 90) - np.nanpercentile(
                data_ref.loc[data_ref.index.month == seas_id, "PET"], 10
            )

            prec_avg_nf = data_nf.loc[data_nf.index.month == seas_id, "PREC"].mean()
            prec_ipr_nf = np.nanpercentile(data_nf.loc[data_nf.index.month == seas_id, "PREC"], 90) - np.nanpercentile(
                data_nf.loc[data_nf.index.month == seas_id, "PREC"], 10
            )
            ta_avg_nf = data_nf.loc[data_nf.index.month == seas_id, "TA"].mean()
            ta_ipr_nf = np.nanpercentile(data_nf.loc[data_nf.index.month == seas_id, "TA"], 90) - np.nanpercentile(
                data_nf.loc[data_nf.index.month == seas_id, "TA"], 10
            )
            pet_avg_nf = data_nf.loc[data_nf.index.month == seas_id, "PET"].mean()
            pet_ipr_nf = np.nanpercentile(data_nf.loc[data_nf.index.month == seas_id, "PET"], 90) - np.nanpercentile(
                data_nf.loc[data_nf.index.month == seas_id, "PET"], 10
            )

            prec_avg_ff = data_ff.loc[data_ff.index.month == seas_id, "PREC"].mean()
            prec_ipr_ff = np.nanpercentile(data_ff.loc[data_ff.index.month == seas_id, "PREC"], 90) - np.nanpercentile(
                data_ff.loc[data_ff.index.month == seas_id, "PREC"], 10
            )
            ta_avg_ff = data_ff.loc[data_ff.index.month == seas_id, "TA"].mean()
            ta_ipr_ff = np.nanpercentile(data_ff.loc[data_ff.index.month == seas_id, "TA"], 90) - np.nanpercentile(
                data_ff.loc[data_ff.index.month == seas_id, "TA"], 10
            )
            pet_avg_ff = data_ff.loc[data_ff.index.month == seas_id, "PET"].mean()
            pet_ipr_ff = np.nanpercentile(data_ff.loc[data_ff.index.month == seas_id, "PET"], 90) - np.nanpercentile(
                data_ff.loc[data_ff.index.month == seas_id, "PET"], 10
            )

            df_prec.loc[seas, "dAvg_nf"] = (prec_avg_nf - prec_avg_ref) / prec_avg_ref
            df_prec.loc[seas, "dAvg_ff"] = (prec_avg_ff - prec_avg_ref) / prec_avg_ref
            df_prec.loc[seas, "dIPR_nf"] = (prec_ipr_nf - prec_ipr_ref) / prec_ipr_ref
            df_prec.loc[seas, "dIPR_ff"] = (prec_ipr_ff - prec_ipr_ref) / prec_ipr_ref
            df_ta.loc[seas, "dAvg_nf"] = (ta_avg_nf - ta_avg_ref) / ta_avg_ref
            df_ta.loc[seas, "dAvg_ff"] = (ta_avg_ff - ta_avg_ref) / ta_avg_ref
            df_ta.loc[seas, "dIPR_nf"] = (ta_ipr_nf - ta_ipr_ref) / ta_ipr_ref
            df_ta.loc[seas, "dIPR_ff"] = (ta_ipr_ff - ta_ipr_ref) / ta_ipr_ref
            df_pet.loc[seas, "dAvg_nf"] = (pet_avg_nf - pet_avg_ref) / pet_avg_ref
            df_pet.loc[seas, "dAvg_ff"] = (pet_avg_ff - pet_avg_ref) / pet_avg_ref
            df_pet.loc[seas, "dIPR_nf"] = (pet_ipr_nf - pet_ipr_ref) / pet_ipr_ref
            df_pet.loc[seas, "dIPR_ff"] = (pet_ipr_ff - pet_ipr_ref) / pet_ipr_ref

        dict_deltas_climate_seas[station_label1[station_id]][cm]["prec"] = {}
        dict_deltas_climate_seas[station_label1[station_id]][cm]["prec"] = df_prec
        dict_deltas_climate_seas[station_label1[station_id]][cm]["ta"] = {}
        dict_deltas_climate_seas[station_label1[station_id]][cm]["ta"] = df_ta
        dict_deltas_climate_seas[station_label1[station_id]][cm]["pet"] = {}
        dict_deltas_climate_seas[station_label1[station_id]][cm]["pet"] = df_pet

# Store data (serialize)
file = base_path / "figures" / "delta_changes_climate.pkl"
with open(file, "wb") as handle:
    pickle.dump(dict_deltas_climate, handle, protocol=pickle.HIGHEST_PROTOCOL)

# plot delta changes of precipitation and temperature
fig, axs = plt.subplots(1, 3, figsize=(6, 1.8), sharex=True, sharey=True)
for i, station_id in enumerate(station_ids):
    for cm in cms:
        dta_avg_nf = dict_deltas_climate[station_label1[station_id]][cm]["ta"].loc[0, "dAvg_nf"] * 100
        dprec_avg_nf = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dAvg_nf"] * 100
        axs[i].scatter(dta_avg_nf, dprec_avg_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
    for cm in cms:
        dta_avg_ff = dict_deltas_climate[station_label1[station_id]][cm]["ta"].loc[0, "dAvg_ff"] * 100
        dprec_avg_ff = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dAvg_ff"] * 100
        axs[i].scatter(dta_avg_ff, dprec_avg_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
    axs[i].set_ylabel("")
    axs[i].set_xlabel(r"$\overline{\Delta}$ TA [%]")
axs[0].set_ylabel(r"$\overline{\Delta}$ PREC [%]")
axs[0].text(
    0.9, 0.9, "(a)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0].transAxes
)
axs[1].text(
    0.9, 0.9, "(b)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1].transAxes
)
axs[2].text(
    0.9, 0.9, "(c)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[2].transAxes
)
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
file = base_path_figs / f"projected_annual_dprec_and_dta_avg.png"
fig.savefig(file, dpi=300)
file = base_path_figs / f"projected_annual_dprec_and_dta_avg.pdf"
fig.savefig(file, dpi=300)
plt.close(fig=fig)

fig, axs = plt.subplots(2, 3, figsize=(6, 4), sharex="row", sharey="row")
for i, station_id in enumerate(station_ids):
    for cm in cms:
        dta_avg_nf = dict_deltas_climate[station_label1[station_id]][cm]["ta"].loc[0, "dAvg_nf"] * 100
        dprec_avg_nf = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dAvg_nf"] * 100
        axs[0, i].scatter(dta_avg_nf, dprec_avg_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
        dta_ipr_nf = dict_deltas_climate[station_label1[station_id]][cm]["ta"].loc[0, "dIPR_nf"] * 100
        dprec_ipr_nf = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dIPR_nf"] * 100
        axs[1, i].scatter(dta_ipr_nf, dprec_ipr_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
    for cm in cms:
        dta_avg_ff = dict_deltas_climate[station_label1[station_id]][cm]["ta"].loc[0, "dAvg_ff"] * 100
        dprec_avg_ff = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dAvg_ff"] * 100
        axs[0, i].scatter(dta_avg_ff, dprec_avg_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
        dta_ipr_ff = dict_deltas_climate[station_label1[station_id]][cm]["ta"].loc[0, "dIPR_ff"] * 100
        dprec_ipr_ff = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dIPR_ff"] * 100
        axs[1, i].scatter(dta_ipr_ff, dprec_ipr_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
    axs[0, i].set_ylabel("")
    axs[0, i].set_xlabel(r"$\overline{\Delta}$ TA [%]")
    axs[1, i].set_xlabel(r"$\Delta$$IPR$ TA [%]")
axs[0, 0].set_ylabel(r"$\overline{\Delta}$ PREC [%]")
axs[1, 0].set_ylabel(r"$\Delta$$IPR$ PREC [%]")
axs[0, 0].text(
    0.9, 0.9, "(a)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 0].transAxes
)
axs[0, 1].text(
    0.9, 0.9, "(b)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 1].transAxes
)
axs[0, 2].text(
    0.9, 0.9, "(c)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 2].transAxes
)
axs[1, 0].text(
    0.9, 0.9, "(d)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1, 0].transAxes
)
axs[1, 1].text(
    0.9, 0.9, "(e)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1, 1].transAxes
)
axs[1, 2].text(
    0.9, 0.9, "(f)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1, 2].transAxes
)
lines, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(
    lines[:7],
    labels[:7],
    loc="upper right",
    fontsize=6,
    frameon=False,
    bbox_to_anchor=(1.0, 0.9),
    title="2030-2060",
)
fig.legend(
    lines[7:],
    labels[7:],
    loc="upper right",
    fontsize=6,
    frameon=False,
    bbox_to_anchor=(1.0, 0.64),
    title="2070-2100",
)
fig.subplots_adjust(left=0.08, right=0.72, bottom=0.15, wspace=0.1, hspace=0.35)
file = base_path_figs / f"projected_annual_dprec_and_dta.png"
fig.savefig(file, dpi=300)
file = base_path_figs / f"projected_annual_dprec_and_dta.pdf"
fig.savefig(file, dpi=300)
plt.close(fig=fig)

# plot delta changes of precipitation and potential evapotranspiration
fig, axs = plt.subplots(2, 3, figsize=(6, 4), sharex="row", sharey="row")
for i, station_id in enumerate(station_ids):
    for cm in cms:
        dpet_avg_nf = dict_deltas_climate[station_label1[station_id]][cm]["pet"].loc[0, "dAvg_nf"] * 100
        dprec_avg_nf = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dAvg_nf"] * 100
        axs[0, i].scatter(dpet_avg_nf, dprec_avg_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
        dpet_ipr_nf = dict_deltas_climate[station_label1[station_id]][cm]["pet"].loc[0, "dIPR_nf"] * 100
        dprec_ipr_nf = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dIPR_nf"] * 100
        axs[1, i].scatter(dpet_ipr_nf, dprec_ipr_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
    for cm in cms:
        dpet_avg_ff = dict_deltas_climate[station_label1[station_id]][cm]["pet"].loc[0, "dAvg_ff"] * 100
        dprec_avg_ff = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dAvg_ff"] * 100
        axs[0, i].scatter(dpet_avg_ff, dprec_avg_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
        dpet_ipr_ff = dict_deltas_climate[station_label1[station_id]][cm]["pet"].loc[0, "dIPR_ff"] * 100
        dprec_ipr_ff = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dIPR_ff"] * 100
        axs[1, i].scatter(dpet_ipr_ff, dprec_ipr_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
    axs[0, i].set_ylabel("")
    axs[0, i].set_xlabel(r"$\overline{\Delta}$ PET [%]")
    axs[1, i].set_xlabel(r"$\Delta$$IPR$ PET [%]")
axs[0, 0].set_ylabel(r"$\overline{\Delta}$ PREC [%]")
axs[1, 0].set_ylabel(r"$\Delta$$IPR$ PREC [%]")
axs[0, 0].text(
    0.9, 0.9, "(a)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 0].transAxes
)
axs[0, 1].text(
    0.9, 0.9, "(b)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 1].transAxes
)
axs[0, 2].text(
    0.9, 0.9, "(c)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 2].transAxes
)
axs[1, 0].text(
    0.9, 0.9, "(d)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1, 0].transAxes
)
axs[1, 1].text(
    0.9, 0.9, "(e)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1, 1].transAxes
)
axs[1, 2].text(
    0.9, 0.9, "(f)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1, 2].transAxes
)
lines, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(
    lines[:7],
    labels[:7],
    loc="upper right",
    fontsize=6,
    frameon=False,
    bbox_to_anchor=(1.0, 0.9),
    title="2030-2060",
)
fig.legend(
    lines[7:],
    labels[7:],
    loc="upper right",
    fontsize=6,
    frameon=False,
    bbox_to_anchor=(1.0, 0.64),
    title="2070-2100",
)
fig.subplots_adjust(left=0.08, right=0.72, bottom=0.15, wspace=0.1, hspace=0.35)
file = base_path_figs / f"projected_annual_dprec_and_dpet.png"
fig.savefig(file, dpi=300)
file = base_path_figs / f"projected_annual_dprec_and_dpet.pdf"
fig.savefig(file, dpi=300)
plt.close(fig=fig)

for j, seas in enumerate(["winter", "spring", "summer", "autumn"]):
    fig, axs = plt.subplots(2, 3, figsize=(6, 4), sharex=True, sharey=True)
    for i, station_id in enumerate(station_ids):
        for cm in cms:
            dpet_avg_nf = dict_deltas_climate_seas[station_label1[station_id]][cm]["pet"].loc[seas, "dAvg_nf"] * 100
            dprec_avg_nf = dict_deltas_climate_seas[station_label1[station_id]][cm]["prec"].loc[seas, "dAvg_nf"] * 100
            axs[0, i].scatter(dpet_avg_nf, dprec_avg_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
            dpet_ipr_nf = dict_deltas_climate[station_label1[station_id]][cm]["pet"].loc[0, "dIPR_nf"] * 100
            dprec_ipr_nf = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dIPR_nf"] * 100
            axs[1, i].scatter(dpet_ipr_nf, dprec_ipr_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
        for cm in cms:
            dpet_avg_ff = dict_deltas_climate_seas[station_label1[station_id]][cm]["pet"].loc[seas, "dAvg_ff"] * 100
            dprec_avg_ff = dict_deltas_climate_seas[station_label1[station_id]][cm]["prec"].loc[seas, "dAvg_ff"] * 100
            axs[0, i].scatter(dpet_avg_ff, dprec_avg_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
            dpet_ipr_ff = dict_deltas_climate[station_label1[station_id]][cm]["pet"].loc[0, "dIPR_ff"] * 100
            dprec_ipr_ff = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dIPR_ff"] * 100
            axs[1, i].scatter(dpet_ipr_ff, dprec_ipr_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
        axs[0, i].set_ylabel("")
        axs[1, i].set_ylabel("")
        axs[0, i].set_xlabel(r"$\overline{\Delta}$ PET [%]")
        axs[1, i].set_xlabel(r"$\Delta$$IPR$ PET [%]")
    axs[0, 0].set_ylabel(r"$\overline{\Delta}$ PREC [%]")
    axs[1, 0].set_ylabel(r"$\Delta$$IPR$ PREC [%]")
    axs[0, 0].text(
        0.9,
        0.9,
        "(a)",
        fontsize=9,
        verticalalignment="center",
        horizontalalignment="center",
        transform=axs[0, 0].transAxes,
    )
    axs[0, 1].text(
        0.9,
        0.9,
        "(b)",
        fontsize=9,
        verticalalignment="center",
        horizontalalignment="center",
        transform=axs[0, 1].transAxes,
    )
    axs[0, 2].text(
        0.9,
        0.9,
        "(c)",
        fontsize=9,
        verticalalignment="center",
        horizontalalignment="center",
        transform=axs[0, 2].transAxes,
    )
    axs[1, 0].text(
        0.9,
        0.9,
        "(d)",
        fontsize=9,
        verticalalignment="center",
        horizontalalignment="center",
        transform=axs[1, 0].transAxes,
    )
    axs[1, 1].text(
        0.9,
        0.9,
        "(e)",
        fontsize=9,
        verticalalignment="center",
        horizontalalignment="center",
        transform=axs[1, 1].transAxes,
    )
    axs[1, 2].text(
        0.9,
        0.9,
        "(f)",
        fontsize=9,
        verticalalignment="center",
        horizontalalignment="center",
        transform=axs[1, 2].transAxes,
    )
    lines, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        lines[:7],
        labels[:7],
        loc="upper right",
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(1.0, 0.9),
        title="2030-2060",
    )
    fig.legend(
        lines[7:],
        labels[7:],
        loc="upper right",
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(1.0, 0.64),
        title="2070-2100",
    )
    fig.subplots_adjust(left=0.08, right=0.72, bottom=0.15, wspace=0.1, hspace=0.25)
    file = base_path_figs / f"projected_{seas}_dprec_and_dpet.png"
    fig.savefig(file, dpi=300)
    file = base_path_figs / f"projected_{seas}_dprec_and_dpet.pdf"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

fig, axs = plt.subplots(5, 3, figsize=(6, 6), sharex=True, sharey=True)
for i, station_id in enumerate(station_ids):
    for cm in cms:
        dpet_avg_nf = dict_deltas_climate[station_label1[station_id]][cm]["pet"].loc[0, "dAvg_nf"] * 100
        dprec_avg_nf = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dAvg_nf"] * 100
        axs[0, i].scatter(dpet_avg_nf, dprec_avg_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
    for cm in cms:
        dpet_avg_ff = dict_deltas_climate[station_label1[station_id]][cm]["pet"].loc[0, "dAvg_ff"] * 100
        dprec_avg_ff = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dAvg_ff"] * 100
        axs[0, i].scatter(dpet_avg_ff, dprec_avg_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
    axs[0, i].set_ylabel("")
axs[0, 0].set_ylabel(r"$\overline{\Delta}$ PREC [%]")
axs[0, 0].text(
    0.9, 0.9, "(a)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 0].transAxes
)
axs[0, 1].text(
    0.9, 0.9, "(b)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 1].transAxes
)
axs[0, 2].text(
    0.9, 0.9, "(c)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 2].transAxes
)
lines, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(
    lines[:7],
    labels[:7],
    loc="upper right",
    fontsize=6,
    frameon=False,
    bbox_to_anchor=(1.0, 0.88),
    title="2030-2060",
)
fig.legend(
    lines[7:],
    labels[7:],
    loc="upper right",
    fontsize=6,
    frameon=False,
    bbox_to_anchor=(1.0, 0.7),
    title="2070-2100",
)

for i, station_id in enumerate(station_ids):
    for j, seas in enumerate(["winter", "spring", "summer", "autumn"]):
        j = j + 1
        for cm in cms:
            dpet_avg_nf = dict_deltas_climate_seas[station_label1[station_id]][cm]["pet"].loc[seas, "dAvg_nf"] * 100
            dprec_avg_nf = dict_deltas_climate_seas[station_label1[station_id]][cm]["prec"].loc[seas, "dAvg_nf"] * 100
            axs[j, i].scatter(dpet_avg_nf, dprec_avg_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
        for cm in cms:
            dpet_avg_ff = dict_deltas_climate_seas[station_label1[station_id]][cm]["pet"].loc[seas, "dAvg_ff"] * 100
            dprec_avg_ff = dict_deltas_climate_seas[station_label1[station_id]][cm]["prec"].loc[seas, "dAvg_ff"] * 100
            axs[j, i].scatter(dpet_avg_ff, dprec_avg_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
        axs[j, 0].set_ylabel(r"$\overline{\Delta}$ PREC [%]")
    axs[-1, i].set_ylabel(r"$\overline{\Delta}$ PET [%]")
axs[1, 0].text(
    0.9, 0.9, "(d)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1, 0].transAxes
)
axs[1, 1].text(
    0.9, 0.9, "(e)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1, 1].transAxes
)
axs[1, 2].text(
    0.9, 0.9, "(f)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1, 2].transAxes
)
axs[2, 0].text(
    0.9, 0.9, "(g)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[2, 0].transAxes
)
axs[2, 1].text(
    0.9, 0.9, "(h)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[2, 1].transAxes
)
axs[2, 2].text(
    0.9, 0.9, "(i)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[2, 2].transAxes
)
axs[3, 0].text(
    0.9, 0.9, "(j)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[3, 0].transAxes
)
axs[3, 1].text(
    0.9, 0.9, "(k)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[3, 1].transAxes
)
axs[3, 2].text(
    0.9, 0.9, "(l)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[3, 2].transAxes
)
axs[4, 0].text(
    0.9, 0.9, "(m)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[4, 0].transAxes
)
axs[4, 1].text(
    0.9, 0.9, "(n)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[4, 1].transAxes
)
axs[4, 2].text(
    0.9, 0.9, "(o)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[4, 2].transAxes
)

fig.subplots_adjust(left=0.08, right=0.72, bottom=0.2, wspace=0.1, hspace=0.1)
file = base_path_figs / f"projected_annual_seasonal_dprec_and_dpet_avg.png"
fig.savefig(file, dpi=300)
file = base_path_figs / f"projected_annual_seasonal_dprec_and_dpet_avg.pdf"
fig.savefig(file, dpi=300)
plt.close(fig=fig)

fig, axs = plt.subplots(3, 3, figsize=(6, 5), sharex=True, sharey=True)
for i, station_id in enumerate(station_ids):
    for cm in cms:
        dpet_avg_nf = dict_deltas_climate[station_label1[station_id]][cm]["pet"].loc[0, "dAvg_nf"] * 100
        dprec_avg_nf = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dAvg_nf"] * 100
        axs[0, i].scatter(dta_avg_nf, dprec_avg_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
    for cm in cms:
        dpet_avg_ff = dict_deltas_climate[station_label1[station_id]][cm]["pet"].loc[0, "dAvg_ff"] * 100
        dprec_avg_ff = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dAvg_ff"] * 100
        axs[0, i].scatter(dpet_avg_ff, dprec_avg_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
    axs[0, i].set_ylabel("")
axs[0, 0].set_ylabel(r"$\overline{\Delta}$ PREC [%]")
axs[0, 0].text(
    0.9, 0.9, "(a)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 0].transAxes
)
axs[0, 1].text(
    0.9, 0.9, "(b)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 1].transAxes
)
axs[0, 2].text(
    0.9, 0.9, "(c)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 2].transAxes
)
lines, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(
    lines[:7],
    labels[:7],
    loc="upper right",
    fontsize=6,
    frameon=False,
    bbox_to_anchor=(1.0, 0.88),
    title="2030-2060",
)
fig.legend(
    lines[7:],
    labels[7:],
    loc="upper right",
    fontsize=6,
    frameon=False,
    bbox_to_anchor=(1.0, 0.68),
    title="2070-2100",
)

for i, station_id in enumerate(station_ids):
    for j, seas in enumerate(["winter", "summer"]):
        j = j + 1
        for cm in cms:
            dpet_avg_nf = dict_deltas_climate_seas[station_label1[station_id]][cm]["pet"].loc[seas, "dAvg_nf"] * 100
            dprec_avg_nf = dict_deltas_climate_seas[station_label1[station_id]][cm]["prec"].loc[seas, "dAvg_nf"] * 100
            axs[j, i].scatter(dpet_avg_nf, dprec_avg_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
        for cm in cms:
            dpet_avg_ff = dict_deltas_climate_seas[station_label1[station_id]][cm]["pet"].loc[seas, "dAvg_ff"] * 100
            dprec_avg_ff = dict_deltas_climate_seas[station_label1[station_id]][cm]["prec"].loc[seas, "dAvg_ff"] * 100
            axs[j, i].scatter(dpet_avg_ff, dprec_avg_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
        axs[j, 0].set_ylabel(r"$\overline{\Delta}$ PREC [%]")
    axs[-1, i].set_xlabel(r"$\overline{\Delta}$ PET [%]")
axs[1, 0].text(
    0.9, 0.9, "(d)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1, 0].transAxes
)
axs[1, 1].text(
    0.9, 0.9, "(e)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1, 1].transAxes
)
axs[1, 2].text(
    0.9, 0.9, "(f)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1, 2].transAxes
)
axs[2, 0].text(
    0.9, 0.9, "(g)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[2, 0].transAxes
)
axs[2, 1].text(
    0.9, 0.9, "(h)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[2, 1].transAxes
)
axs[2, 2].text(
    0.9, 0.9, "(i)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[2, 2].transAxes
)

fig.subplots_adjust(left=0.08, right=0.72, bottom=0.2, wspace=0.1, hspace=0.1)
file = base_path_figs / f"projected_annual_winter_summer_dprec_and_dpet_avg.png"
fig.savefig(file, dpi=300)
file = base_path_figs / f"projected_annual_winter_summer_dprec_and_dpet_avg.pdf"
fig.savefig(file, dpi=300)
plt.close(fig=fig)

fig, axs = plt.subplots(5, 3, figsize=(6, 6), sharex=True, sharey=True)
for i, station_id in enumerate(station_ids):
    for cm in cms:
        dpet_ipr_nf = dict_deltas_climate[station_label1[station_id]][cm]["pet"].loc[0, "dIPR_nf"] * 100
        dprec_ipr_nf = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dIPR_nf"] * 100
        axs[0, i].scatter(dpet_ipr_nf, dprec_ipr_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
    for cm in cms:
        dpet_ipr_ff = dict_deltas_climate[station_label1[station_id]][cm]["pet"].loc[0, "dIPR_ff"] * 100
        dprec_ipr_ff = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dIPR_ff"] * 100
        axs[0, i].scatter(dpet_ipr_ff, dprec_ipr_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
    axs[0, i].set_ylabel("")
axs[0, 0].set_ylabel(r"$\Delta$$IPR$ PREC [%]")
axs[0, 0].text(
    0.9, 0.9, "(a)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 0].transAxes
)
axs[0, 1].text(
    0.9, 0.9, "(b)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 1].transAxes
)
axs[0, 2].text(
    0.9, 0.9, "(c)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 2].transAxes
)
lines, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(
    lines[:7],
    labels[:7],
    loc="upper right",
    fontsize=6,
    frameon=False,
    bbox_to_anchor=(1.0, 0.88),
    title="2030-2060",
)
fig.legend(
    lines[7:],
    labels[7:],
    loc="upper right",
    fontsize=6,
    frameon=False,
    bbox_to_anchor=(1.0, 0.7),
    title="2070-2100",
)

for i, station_id in enumerate(station_ids):
    for j, seas in enumerate(["winter", "spring", "summer", "autumn"]):
        j = j + 1
        for cm in cms:
            dpet_ipr_nf = dict_deltas_climate_seas[station_label1[station_id]][cm]["pet"].loc[seas, "dIPR_nf"] * 100
            dprec_ipr_nf = dict_deltas_climate_seas[station_label1[station_id]][cm]["prec"].loc[seas, "dIPR_nf"] * 100
            axs[j, i].scatter(dpet_ipr_nf, dprec_ipr_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
        for cm in cms:
            dpet_ipr_ff = dict_deltas_climate_seas[station_label1[station_id]][cm]["pet"].loc[seas, "dIPR_ff"] * 100
            dprec_ipr_ff = dict_deltas_climate_seas[station_label1[station_id]][cm]["prec"].loc[seas, "dIPR_ff"] * 100
            axs[j, i].scatter(dpet_ipr_ff, dprec_ipr_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
        axs[j, 0].set_ylabel(r"$\Delta$$IPR$ PREC [%]")
    axs[-1, i].set_xlabel(r"$\Delta$$IPR$ PET [%]")
axs[1, 0].text(
    0.9, 0.9, "(d)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1, 0].transAxes
)
axs[1, 1].text(
    0.9, 0.9, "(e)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1, 1].transAxes
)
axs[1, 2].text(
    0.9, 0.9, "(f)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1, 2].transAxes
)
axs[2, 0].text(
    0.9, 0.9, "(g)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[2, 0].transAxes
)
axs[2, 1].text(
    0.9, 0.9, "(h)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[2, 1].transAxes
)
axs[2, 2].text(
    0.9, 0.9, "(i)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[2, 2].transAxes
)
axs[3, 0].text(
    0.9, 0.9, "(j)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[3, 0].transAxes
)
axs[3, 1].text(
    0.9, 0.9, "(k)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[3, 1].transAxes
)
axs[3, 2].text(
    0.9, 0.9, "(l)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[3, 2].transAxes
)
axs[4, 0].text(
    0.9, 0.9, "(m)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[4, 0].transAxes
)
axs[4, 1].text(
    0.9, 0.9, "(n)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[4, 1].transAxes
)
axs[4, 2].text(
    0.9, 0.9, "(o)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[4, 2].transAxes
)

fig.subplots_adjust(left=0.08, right=0.72, bottom=0.2, wspace=0.1, hspace=0.1)
file = base_path_figs / f"projected_annual_seasonal_dprec_and_dpet_ipr.png"
fig.savefig(file, dpi=300)
file = base_path_figs / f"projected_annual_seasonal_dprec_and_dpet_ipr.pdf"
fig.savefig(file, dpi=300)
plt.close(fig=fig)

fig, axs = plt.subplots(3, 3, figsize=(6, 5), sharex=True, sharey=True)
for i, station_id in enumerate(station_ids):
    for cm in cms:
        dpet_ipr_nf = dict_deltas_climate[station_label1[station_id]][cm]["pet"].loc[0, "dIPR_nf"] * 100
        dprec_ipr_nf = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dIPR_nf"] * 100
        axs[0, i].scatter(dpet_ipr_nf, dprec_ipr_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
    for cm in cms:
        dpet_ipr_ff = dict_deltas_climate[station_label1[station_id]][cm]["pet"].loc[0, "dIPR_ff"] * 100
        dprec_ipr_ff = dict_deltas_climate[station_label1[station_id]][cm]["prec"].loc[0, "dIPR_ff"] * 100
        axs[0, i].scatter(dpet_ipr_ff, dprec_ipr_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
axs[0, 0].set_ylabel(r"$\Delta$$IPR$ PREC [%]")
axs[0, 0].text(
    0.9, 0.9, "(a)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 0].transAxes
)
axs[0, 1].text(
    0.9, 0.9, "(b)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 1].transAxes
)
axs[0, 2].text(
    0.9, 0.9, "(c)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[0, 2].transAxes
)
lines, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(
    lines[:7],
    labels[:7],
    loc="upper right",
    fontsize=6,
    frameon=False,
    bbox_to_anchor=(1.0, 0.88),
    title="2030-2060",
)
fig.legend(
    lines[7:],
    labels[7:],
    loc="upper right",
    fontsize=6,
    frameon=False,
    bbox_to_anchor=(1.0, 0.68),
    title="2070-2100",
)

for i, station_id in enumerate(station_ids):
    for j, seas in enumerate(["winter", "summer"]):
        j = j + 1
        for cm in cms:
            dpet_ipr_nf = dict_deltas_climate_seas[station_label1[station_id]][cm]["pet"].loc[seas, "dIPR_nf"] * 100
            dprec_ipr_nf = dict_deltas_climate_seas[station_label1[station_id]][cm]["prec"].loc[seas, "dIPR_nf"] * 100
            axs[j, i].scatter(dpet_ipr_nf, dprec_ipr_nf, label=label[cm], color=color[f"{cm}_future"], s=4, marker="^")
        for cm in cms:
            dpet_ipr_ff = dict_deltas_climate_seas[station_label1[station_id]][cm]["pet"].loc[seas, "dIPR_ff"] * 100
            dprec_ipr_ff = dict_deltas_climate_seas[station_label1[station_id]][cm]["prec"].loc[seas, "dIPR_ff"] * 100
            axs[j, i].scatter(dpet_ipr_ff, dprec_ipr_ff, label=label[cm], color=color[f"{cm}_future"], s=8, marker="*")
        axs[j, 0].set_ylabel(r"$\Delta$$IPR$ PREC [%]")
    axs[-1, i].set_xlabel(r"$\Delta$$IPR$ PET [%]")
axs[1, 0].text(
    0.9, 0.9, "(d)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1, 0].transAxes
)
axs[1, 1].text(
    0.9, 0.9, "(e)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1, 1].transAxes
)
axs[1, 2].text(
    0.9, 0.9, "(f)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[1, 2].transAxes
)
axs[2, 0].text(
    0.9, 0.9, "(g)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[2, 0].transAxes
)
axs[2, 1].text(
    0.9, 0.9, "(h)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[2, 1].transAxes
)
axs[2, 2].text(
    0.9, 0.9, "(i)", fontsize=9, verticalalignment="center", horizontalalignment="center", transform=axs[2, 2].transAxes
)

fig.subplots_adjust(left=0.08, right=0.72, bottom=0.15, wspace=0.1, hspace=0.1)
file = base_path_figs / f"projected_annual_winter_summer_dprec_and_dpet_ipr.png"
fig.savefig(file, dpi=300)
file = base_path_figs / f"projected_annual_winter_summer_dprec_and_dpet_ipr.pdf"
fig.savefig(file, dpi=300)
plt.close(fig=fig)
