from pathlib import Path
import xarray as xr
from cftime import num2date
import pandas as pd
import geopandas as gpd
import numpy as onp
import matplotlib as mpl
import seaborn as sns
import click
import warnings
warnings.filterwarnings('ignore')

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.titlesize"] = 8
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["legend.fontsize"] = 8
mpl.rcParams["legend.title_fontsize"] = 9
sns.set_style("ticks")
sns.plotting_context(
    "paper",
    font_scale=1,
    rc={
        "font.size": 8.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.title_fontsize": 9.0,
    },
)

_dict_ffid = {"winter-wheat_clover": "1_0",
              "winter-wheat_silage-corn": "2_0",
              "summer-wheat_winter-wheat": "3_0",
              "summer-wheat_clover_winter-wheat": "4_0",
              "winter-wheat_clover_silage-corn": "5_0",
              "winter-wheat_sugar-beet_silage-corn": "6_0",
              "summer-wheat_winter-wheat_silage-corn": "7_0",
              "summer-wheat_winter-wheat_winter-rape": "8_0",
              "winter-wheat_winter-rape": "9_0",
              "winter-wheat_soybean_winter-rape": "10_0",
              "sugar-beet_winter-wheat_winter-barley": "11_0", 
              "grain-corn_winter-wheat_winter-rape": "12_0", 
              "grain-corn_winter-wheat_winter-barley": "13_0",
              "grain-corn_winter-wheat_clover": "14_0",
              "miscanthus": "15_0",
              "bare-grass": "16_0",
              "winter-wheat_silage-corn_yellow-mustard": "2_1",
              "summer-wheat_winter-wheat_yellow-mustard": "3_1",
              "winter-wheat_sugar-beet_silage-corn_yellow-mustard": "6_1",
              "summer-wheat_winter-wheat_silage-corn_yellow-mustard": "7_1",
              "summer-wheat_winter-wheat_winter-rape_yellow-mustard": "8_1",
              "sugar-beet_winter-wheat_winter-barley_yellow-mustard": "11_1", 
              "grain-corn_winter-wheat_winter-rape_yellow-mustard": "12_1", 
              "grain-corn_winter-wheat_winter-barley_yellow-mustard": "13_1", 
}

_dict_var_names = {"q_hof": "QSUR",
                   "ground_cover": "GC",
                   "M_q_ss": "MPERC",
                   "C_q_ss": "CPERC",
                   "q_ss": "PERC",
}

_dict_fert = {"low": 1,
              "medium": 2,
              "high": 3,
}

_dict_crop_id = {"winter-wheat": 115,
                 "clover": 425,
                 "silage-corn": 411,
                 "summer-wheat": 116,
                 "sugar-beet": 603,
                 "winter-rape": 311,
                 "soybean": 330,
                 "grain-corn": 171,
                 "winter-barley": 131,
                 "grass": 591,
                 "miscanthus": 852,
                }

_dict_lu_id = {"winter-wheat": [557],
               "clover": [583, 584, 585],
               "silage-corn": [539],
               "summer-wheat": [543],
               "sugar-beet": [563],
               "winter-rape": [559],
               "soybean": [541],
               "grain-corn": [525],
               "winter-barley": [556],
               "miscanthus": [589, 590, 591],
               "grass": [571, 573, 574],
               "bare": [599],
                }

_dict_crop_periods = {"winter-wheat_clover": {"winter-wheat": [557], "clover": [583, 584, 585]},
                      "winter-wheat_silage-corn": {"winter-wheat": [557], "silage-corn": [539]},
                      "summer-wheat_winter-wheat": {"winter-wheat": [557], "summer-wheat": [543]},
                      "summer-wheat_clover_winter-wheat": {"winter-wheat": [557], "summer-wheat": [543], "clover": [583, 584, 585]},
                      "winter-wheat_clover_silage-corn": {"winter-wheat": [557], "silage-corn": [539], "clover": [583, 584, 585]},
                      "winter-wheat_sugar-beet_silage-corn": {"winter-wheat": [557], "silage-corn": [539], "sugar-beet": [563]},
                      "summer-wheat_winter-wheat_silage-corn": {"winter-wheat": [557], "summer-wheat": [543], "silage-corn": [539]},
                      "summer-wheat_winter-wheat_winter-rape": {"winter-wheat": [557], "summer-wheat": [543], "winter-rape": [559]},
                      "winter-wheat_winter-rape": {"winter-wheat": [557], "winter-rape": [559]},
                      "winter-wheat_soybean_winter-rape": {"winter-wheat": [557], "winter-rape": [559], "soybean": [541]},
                      "sugar-beet_winter-wheat_winter-barley": {"winter-wheat": [557], "winter-barley": [556], "sugar-beet": [563]}, 
                      "grain-corn_winter-wheat_winter-rape": {"grain-corn": [525], "winter-wheat": [557], "winter-rape": [559]},
                      "grain-corn_winter-wheat_winter-barley": {"grain-corn": [525], "winter-wheat": [557], "winter-barley": [556]},
                      "grain-corn_winter-wheat_clover": {"grain-corn": [525], "winter-wheat": [557], "clover": [583, 584, 585]},
                      "miscanthus": {"miscanthus": [589, 590, 591]},
                      "bare-grass": {"grass": [599, 571, 573, 574]},
                      "winter-wheat_silage-corn_yellow-mustard": {"winter-wheat": [557], "silage-corn": [539]},
                      "summer-wheat_winter-wheat_yellow-mustard": {"winter-wheat": [557], "summer-wheat": [543]},
                      "winter-wheat_sugar-beet_silage-corn_yellow-mustard": {"winter-wheat": [557], "silage-corn": [539], "sugar-beet": [563]},
                      "summer-wheat_winter-wheat_silage-corn_yellow-mustard": {"winter-wheat": [557], "summer-wheat": [543], "silage-corn": [539]},
                      "summer-wheat_winter-wheat_winter-rape_yellow-mustard": {"winter-wheat": [557], "summer-wheat": [543], "winter-rape": [559]},
                      "sugar-beet_winter-wheat_winter-barley_yellow-mustard": {"winter-wheat": [557], "winter-barley": [556], "sugar-beet": [563]},  
                      "grain-corn_winter-wheat_winter-rape_yellow-mustard": {"grain-corn": [525], "winter-wheat": [557], "winter-rape": [559]}, 
                      "grain-corn_winter-wheat_winter-barley_yellow-mustard": {"grain-corn": [525], "winter-wheat": [557], "winter-barley": [556]}, 
}

crop_rotation_scenarios = ["winter-wheat_clover",
                           "winter-wheat_silage-corn",
                           "summer-wheat_winter-wheat",
                           "summer-wheat_clover_winter-wheat",
                           "winter-wheat_clover_silage-corn",
                           "winter-wheat_sugar-beet_silage-corn",
                           "summer-wheat_winter-wheat_silage-corn",
                           "summer-wheat_winter-wheat_winter-rape",
                           "winter-wheat_winter-rape",
                           "winter-wheat_soybean_winter-rape",
                           "sugar-beet_winter-wheat_winter-barley", 
                           "grain-corn_winter-wheat_winter-rape", 
                           "grain-corn_winter-wheat_winter-barley",
                           "grain-corn_winter-wheat_clover",
                           "winter-wheat_silage-corn_yellow-mustard",
                           "summer-wheat_winter-wheat_yellow-mustard",
                           "winter-wheat_sugar-beet_silage-corn_yellow-mustard",
                           "summer-wheat_winter-wheat_silage-corn_yellow-mustard",
                           "summer-wheat_winter-wheat_winter-rape_yellow-mustard",
                           "sugar-beet_winter-wheat_winter-barley_yellow-mustard", 
                           "grain-corn_winter-wheat_winter-rape_yellow-mustard", 
                           "grain-corn_winter-wheat_winter-barley_yellow-mustard",
                           "miscanthus",
                           "bare-grass"]

fertilization_intensities = ["low", "medium", "high"]


def nanmeanweighted(y, w, axis=None):
    w1 = w / onp.nansum(w, axis=axis)
    w2 = onp.where(onp.isnan(w), 0, w1)
    w3 = onp.where(onp.isnan(y), 0, w2)
    y1 = onp.where(onp.isnan(y), 0, y)
    wavg = onp.sum(y1 * w3, axis=axis) / onp.sum(w3, axis=axis)

    return wavg

@click.option("-l", "--location", type=click.Choice(["freiburg", "lahr", "muellheim", 
                                                     "stockach", "gottmadingen", "weingarten",
                                                     "eppingen-elsenz", "bruchsal-heidelsheim", "bretten",
                                                     "ehingen-kirchen", "merklingen", "hayingen",
                                                     "kupferzell", "oehringen", "vellberg-kleinaltdorf"]), 
                                                     default="muellheim", help="Location of the meteorological station.")
@click.option("-td", "--tmp-dir", type=str, default=Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "output")
@click.command("main")
def main(location, tmp_dir):
    base_path = Path(__file__).parent
    # directory of results
    base_path_output = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "output"
    
    # load linkage between BK50 and cropland clusters
    file = base_path_output / "link_shp_clust_acker.h5"
    df_link_bk50_cluster_cropland = pd.read_hdf(file)

    # load model parameters
    csv_file = base_path / "parameters.csv"
    df_params = pd.read_csv(csv_file, sep=";", skiprows=1)
    cond = (df_params["CLUST_flag"] == 2)
    df_params = df_params.loc[cond, :]
    clust_ids = pd.unique(df_params["CLUST_ID"].values).tolist()
    df_params = pd.read_csv(csv_file, sep=";", skiprows=1)

    # load nitrogen loads and concentrations
    dict_nitrate = {}
    dict_nitrate[location] = {}
    for crop_rotation_scenario in crop_rotation_scenarios:
        dict_nitrate[location][crop_rotation_scenario] = {}
        for fertilization_intensity in fertilization_intensities:
            dict_nitrate[location][crop_rotation_scenario][fertilization_intensity] = {}
            output_nitrate_file = (
                base_path_output
                / "svat_crop_nitrate"
                / f"SVATCROPNITRATE_{location}_{crop_rotation_scenario}_{fertilization_intensity}_Nfert.nc"
            )
            ds_nitrate = xr.open_dataset(output_nitrate_file, engine="h5netcdf")
            # assign date
            days = ds_nitrate["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
            date = num2date(
                days,
                units=f"days since {ds_nitrate['Time'].attrs['time_origin']}",
                calendar="standard",
                only_use_cftime_datetimes=False,
            )
            ds_nitrate = ds_nitrate.assign_coords(Time=("Time", date))
            dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] = ds_nitrate

    # load simulated fluxes and states
    dict_fluxes_states = {}
    dict_fluxes_states[location] = {}
    for crop_rotation_scenario in crop_rotation_scenarios:
        dict_fluxes_states[location][crop_rotation_scenario] = {}
        output_hm_file = (
            base_path_output
            / "svat_crop"
            / f"SVATCROP_{location}_{crop_rotation_scenario}.nc"
        )
        ds_fluxes_states = xr.open_dataset(output_hm_file, engine="h5netcdf")
        # assign date
        days = ds_fluxes_states["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        date = num2date(
            days,
            units=f"days since {ds_fluxes_states['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_fluxes_states = ds_fluxes_states.assign_coords(Time=("Time", date))
        dict_fluxes_states[location][crop_rotation_scenario] = ds_fluxes_states

    # aggregate nitrate leaching, surface runoff and percolation to annual average values
    file = base_path_output / "BK50_NBiomasseBW_for_assignment.gpkg"
    gdf1 = gpd.read_file(file)
    vars_sim = ["q_hof", "q_ss", "M_q_ss", "C_q_ss"]
    ll_df = []
    ll_df_crop_periods = []
    for crop_rotation_scenario in crop_rotation_scenarios:
        click.echo(f"{crop_rotation_scenario}")
        for var_sim in vars_sim:
            click.echo(f"{var_sim}")
            if var_sim in ["M_q_ss", "C_q_ss"]:
                for fertilization_intensity in fertilization_intensities:
                    click.echo(f"{fertilization_intensity}")
                    mask = (gdf1['stationsna'] == location)
                    gdf = gdf1.loc[mask, :]
                    gdf['FFID'] = None
                    gdf['FFID'] = _dict_ffid[crop_rotation_scenario]
                    gdf['CID'] = None
                    gdf['CID'] = int(0)
                    gdf['region'] = None
                    if location in ["freiburg", "lahr", "muellheim"]:
                        gdf['region'] = "upper rhine valley"
                    elif location in ["stockach", "gottmadingen", "weingarten"]:
                        gdf['region'] = "lake constance"
                    elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                        gdf['region'] = "kraichgau"
                    elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                        gdf['region'] = "alb-danube"
                    elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                        gdf['region'] = "hohenlohe"
                    gdf[f'{_dict_var_names[var_sim]}_N{_dict_fert[fertilization_intensity]}'] = None  # initialize field, float, two decimals
                    gdf[f'{_dict_var_names[var_sim]}_N{_dict_fert[fertilization_intensity]}'] = gdf[f'{_dict_var_names[var_sim]}_N{_dict_fert[fertilization_intensity]}'].astype('float64')
                    gdf[f'{_dict_var_names[var_sim]}_N{_dict_fert[fertilization_intensity]}'] = gdf[f'{_dict_var_names[var_sim]}_N{_dict_fert[fertilization_intensity]}'].round(decimals=2)
                    ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert']
                    if var_sim == "M_q_ss":
                        sim_vals = ds[var_sim].isel(y=0).values[:, 1:]
                        df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T)
                        # calculate annual sum
                        df_ann = df.resample("YE").sum() * 0.01  # convert from mg/m2 to kg/ha
                        # calculate average
                        df_avg = df_ann.mean(axis=0).to_frame()
                    elif var_sim == "C_q_ss":
                        ds = dict_fluxes_states[location][crop_rotation_scenario]
                        sim_vals1 = ds["q_ss"].isel(y=0).values
                        ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
                        sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:] * 4.427  # convert nitrate-nitrogen to nitrate
                        sim_vals = onp.where(sim_vals1 > 0.01, (sim_vals2/sim_vals1) * (sim_vals1/onp.sum(sim_vals1, axis=-1)[:, onp.newaxis]), onp.nan)  # weighted average
                        cond1 = (df_params["CLUST_flag"] == 2)
                        df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T).loc[:, cond1]
                        # calculate annual mean
                        df_avg = df.sum(axis=0).to_frame()
                        df_avg.loc[:, "location"] = location
                    # assign aggregated values to polygons
                    for clust_id in clust_ids:
                        cond1 = (df_params["CLUST_ID"] == clust_id) & (df_params["CLUST_flag"] == 2)
                        val = df_avg.loc[cond1, :].values[0][0]
                        cond = (df_link_bk50_cluster_cropland["CLUST_ID"] == clust_id)
                        shp_ids = df_link_bk50_cluster_cropland.loc[cond, :].index.tolist()
                        cond2 = gdf["SHP_ID"].isin(shp_ids)
                        if cond2.any():
                            gdf.loc[cond2, f'{_dict_var_names[var_sim]}_N{_dict_fert[fertilization_intensity]}'] = val
                    gdf = gdf.copy()
                    gdf = gdf.to_crs("EPSG:25832")
                    ll_df.append(gdf)
            elif var_sim in ["q_ss", "q_hof"]:
                mask = (gdf1['stationsna'] == location)
                gdf = gdf1.loc[mask, :]
                gdf['FFID'] = None
                gdf['FFID'] = _dict_ffid[crop_rotation_scenario]
                gdf['CID'] = None
                gdf['CID'] = int(0)
                gdf['region'] = None
                if location in ["freiburg", "lahr", "muellheim"]:
                    gdf['region'] = "upper rhine valley"
                elif location in ["stockach", "gottmadingen", "weingarten"]:
                    gdf['region'] = "lake constance"
                elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                    gdf['region'] = "kraichgau"
                elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                    gdf['region'] = "alb-danube"
                elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                    gdf['region'] = "hohenlohe"
                gdf[f'{_dict_var_names[var_sim]}'] = None  # initialize field, float, two decimals
                gdf[f'{_dict_var_names[var_sim]}'] = gdf[f'{_dict_var_names[var_sim]}'].astype('float64')
                gdf[f'{_dict_var_names[var_sim]}'] = gdf[f'{_dict_var_names[var_sim]}'].round(decimals=2)
                ds = dict_fluxes_states[location][crop_rotation_scenario]
                sim_vals = ds[var_sim].isel(y=0).values[:, 1:]
                df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T)
                # calculate annual sum
                df_ann = df.resample("YE").sum()
                # calculate average
                df_avg = df_ann.mean(axis=0).to_frame()

                # assign aggregated values to polygons
                for clust_id in clust_ids:
                    cond1 = (df_params["CLUST_ID"] == clust_id) & (df_params["CLUST_flag"] == 2)
                    val = df_avg.loc[cond1, :].values[0][0]
                    cond = (df_link_bk50_cluster_cropland["CLUST_ID"] == clust_id)
                    shp_ids = df_link_bk50_cluster_cropland.loc[cond, :].index.tolist()
                    cond2 = gdf["SHP_ID"].isin(shp_ids)
                    if cond2.any():
                        gdf.loc[cond2, f'{_dict_var_names[var_sim]}'] = val
                gdf = gdf.copy()
                gdf = gdf.to_crs("EPSG:25832")
                ll_df.append(gdf)

            # extract the values for the single crop periods within the crop rotation
            crop_ids = _dict_crop_periods[crop_rotation_scenario]
            click.echo(f"{crop_ids}")
            for crop_id in crop_ids:
                ds = dict_fluxes_states[location][crop_rotation_scenario]
                lu_id = ds["lu_id"].isel(x=0, y=0).values
                df_lu_id = pd.DataFrame(index=ds["Time"].values, data=lu_id)
                df_lu_id.columns = ["lu_id"]
                cond = onp.isin(df_lu_id.values, onp.array([586, 587, 599])).flatten()
                df_lu_id.loc[cond, "lu_id"] = onp.nan
                df_lu_id = df_lu_id.bfill()
                df_lu_id.loc[:, 'mask'] = onp.isin(df_lu_id.loc[:, "lu_id"].values, _dict_lu_id[crop_id]).flatten()
                df_lu_id['new'] = df_lu_id['mask'].gt(df_lu_id['mask'].shift(fill_value=False))
                df_lu_id['crop_period'] = (df_lu_id.new + 0).cumsum() * df_lu_id['mask']
                df_lu_id['day'] = 1
                df_lu_id.replace(onp.nan, 0, inplace=True)
                df_lu_id['crop_period'] = df_lu_id['crop_period'].replace(0, onp.nan)
                df_lu_id = df_lu_id.drop(columns=['mask', 'new'])
                df_lu_id = df_lu_id.astype({"lu_id": int, "crop_period": float, "day": float})
                cond_crop = onp.isin(df_lu_id.loc[:, "lu_id"].values, _dict_crop_periods[crop_rotation_scenario][crop_id]).flatten()
                if var_sim in ["M_q_ss", "C_q_ss"]:
                    for fertilization_intensity in fertilization_intensities:
                        click.echo(f"{crop_id}: {var_sim} {fertilization_intensity}")
                        mask = (gdf1['stationsna'] == location)
                        gdf = gdf1.loc[mask, :]
                        gdf['FFID'] = None
                        gdf['FFID'] = f'{_dict_ffid[crop_rotation_scenario]}'
                        gdf['CID'] = None
                        gdf['CID'] = int(f'{_dict_crop_id[crop_id]}')
                        gdf['region'] = None
                        if location in ["freiburg", "lahr", "muellheim"]:
                            gdf['region'] = "upper rhine valley"
                        elif location in ["stockach", "gottmadingen", "weingarten"]:
                            gdf['region'] = "lake constance"
                        elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                            gdf['region'] = "kraichgau"
                        elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                            gdf['region'] = "alb-danube"
                        elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                            gdf['region'] = "hohenlohe"
                        gdf[f'{_dict_var_names[var_sim]}_N{_dict_fert[fertilization_intensity]}'] = None  # initialize field, float, two decimals
                        gdf[f'{_dict_var_names[var_sim]}_N{_dict_fert[fertilization_intensity]}'] = gdf[f'{_dict_var_names[var_sim]}_N{_dict_fert[fertilization_intensity]}'].astype('float64')
                        gdf[f'{_dict_var_names[var_sim]}_N{_dict_fert[fertilization_intensity]}'] = gdf[f'{_dict_var_names[var_sim]}_N{_dict_fert[fertilization_intensity]}'].round(decimals=2)
                        ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert']
                        if var_sim == "M_q_ss":
                            sim_vals = ds[var_sim].isel(y=0).values[:, 1:]
                            df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T)
                            df.loc[:, 'crop_period'] = df_lu_id.loc[:, 'crop_period'].values
                            df = df.loc[cond_crop, :]
                            # calculate sum per crop period
                            weight = 365/df_lu_id.groupby("crop_period").sum()["day"].values
                            weight = onp.where(weight > 1, 1, weight)
                            df_cp = df.groupby("crop_period").sum() * weight[:, onp.newaxis] * 0.01  # convert from mg/m2 to kg/ha                            
                            # calculate average
                            df_avg = df_cp.mean(axis=0).to_frame()
                        elif var_sim == "C_q_ss":
                            ds = dict_fluxes_states[location][crop_rotation_scenario]
                            sim_vals1 = ds["q_ss"].isel(y=0).values[:, cond_crop]
                            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
                            sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:][:, cond_crop] * 4.427  # convert nitrate-nitrogen to nitrate
                            sim_vals = onp.where(sim_vals1 > 0.01, (sim_vals2/sim_vals1) * (sim_vals1/onp.sum(sim_vals1, axis=-1)[:, onp.newaxis]), onp.nan)  # weighted average
                            cond1 = (df_params["CLUST_flag"] == 2)
                            df = pd.DataFrame(data=sim_vals.T).loc[:, cond1]
                            # calculate annual mean
                            df_avg = df.sum(axis=0).to_frame()
                        # assign aggregated values to polygons
                        for clust_id in clust_ids:
                            cond1 = (df_params["CLUST_ID"] == clust_id) & (df_params["CLUST_flag"] == 2)
                            val = df_avg.loc[cond1, :].values[0][0]
                            cond = (df_link_bk50_cluster_cropland["CLUST_ID"] == clust_id)
                            shp_ids = df_link_bk50_cluster_cropland.loc[cond, :].index.tolist()
                            cond2 = gdf["SHP_ID"].isin(shp_ids)
                            if cond2.any():
                                gdf.loc[cond2, f'{_dict_var_names[var_sim]}_N{_dict_fert[fertilization_intensity]}'] = val
                        gdf = gdf.copy()
                        gdf = gdf.to_crs("EPSG:25832")
                        ll_df.append(gdf)
                elif var_sim in ["q_ss", "q_hof"]:
                    click.echo(f"{crop_id}: {var_sim}")
                    mask = (gdf1['stationsna'] == location)
                    gdf = gdf1.loc[mask, :]
                    gdf['FFID'] = None
                    gdf['FFID'] = f'{_dict_ffid[crop_rotation_scenario]}'
                    gdf['CID'] = None
                    gdf['CID'] = int(f'{_dict_crop_id[crop_id]}')
                    gdf['region'] = None
                    if location in ["freiburg", "lahr", "muellheim"]:
                        gdf['region'] = "upper rhine valley"
                    elif location in ["stockach", "gottmadingen", "weingarten"]:
                        gdf['region'] = "lake constance"
                    elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                        gdf['region'] = "kraichgau"
                    elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                        gdf['region'] = "alb-danube"
                    elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                        gdf['region'] = "hohenlohe"
                    gdf[f'{_dict_var_names[var_sim]}'] = None  # initialize field, float, two decimals
                    gdf[f'{_dict_var_names[var_sim]}'] = gdf[f'{_dict_var_names[var_sim]}'].astype('float64')
                    gdf[f'{_dict_var_names[var_sim]}'] = gdf[f'{_dict_var_names[var_sim]}'].round(decimals=2)
                    ds = dict_fluxes_states[location][crop_rotation_scenario]
                    sim_vals = ds[var_sim].isel(y=0).values[:, 1:]
                    df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T)
                    df.loc[:, 'crop_period'] = df_lu_id.loc[:, 'crop_period'].values[1:]
                    df = df.loc[cond_crop[1:], :]
                    # calculate sum per crop period
                    weight = 365/df_lu_id.groupby("crop_period").sum()["day"].values
                    weight = onp.where(weight > 1, 1, weight)
                    df_cp = df.groupby("crop_period").sum() * weight[:, onp.newaxis]                          
                    # calculate average
                    df_avg = df_cp.mean(axis=0).to_frame()
                    if var_sim == "q_hof":
                        df_crop_periods = pd.DataFrame(index=[f"{crop_rotation_scenario}_{crop_id}"], columns=["crop_rotation", "crop", "period_in_days"])
                        df_crop_periods.iloc[0, 0] = crop_rotation_scenario
                        df_crop_periods.iloc[0, 1] = crop_id
                        df_crop_periods.iloc[0, 2] = df_lu_id.groupby("crop_period").sum().mean()["day"]
                        ll_df_crop_periods.append(df_crop_periods)
                    # assign aggregated values to polygons
                    for clust_id in clust_ids:
                        cond1 = (df_params["CLUST_ID"] == clust_id) & (df_params["CLUST_flag"] == 2)
                        val = df_avg.loc[cond1, :].values[0][0]
                        cond = (df_link_bk50_cluster_cropland["CLUST_ID"] == clust_id)
                        shp_ids = df_link_bk50_cluster_cropland.loc[cond, :].index.tolist()
                        cond2 = gdf["SHP_ID"].isin(shp_ids)
                        if cond2.any():
                            gdf.loc[cond2, f'{_dict_var_names[var_sim]}'] = val
                    gdf = gdf.copy()
                    gdf = gdf.to_crs("EPSG:25832")
                    ll_df.append(gdf)

        click.echo(f"Finalized {crop_rotation_scenario}")

    gdf = pd.concat(ll_df, axis=0)
    gdf = gdf.to_crs("EPSG:25832")
    file = Path(tmp_dir) / f"nitrate_leaching_{location}.gpkg"
    gdf.to_file(file, driver="GPKG")


if __name__ == "__main__":
    main()