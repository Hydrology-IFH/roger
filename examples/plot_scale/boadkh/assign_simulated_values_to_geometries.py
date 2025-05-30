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
                   "prec": "PRECIP",
                   "Nfert": "NFERT",
}

_dict_fert = {"low": 1,
              "medium": 2,
              "high": 3,
}

_dict_crop_id = {"winter-wheat": 115,
                 "clover": 422,
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


_dict_clust_id = {51: 210,
                  52: 211,
                  53: 212,
                  54: 213,
                  55: 214,
                  56: 215,
                  57: 216,
                  58: 217,
                  59: 218,
                  510: 219,
                  511: 220,
                  512: 221,
                  513: 222,
                  514: 223,
                  515: 224,
                  516: 225,
                  517: 226,
                  518: 227,
                  519: 228,
                  520: 229,
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

_dict_locations = {"freiburg": 111,
                   "lahr": 112,
                   "muellheim": 113,
                   "stockach": 114,
                   "gottmadingen": 115,
                   "weingarten": 116,
                   "eppingen-elsenz": 117,
                   "bruchsal-heidelsheim": 118,
                   "bretten": 119,
                   "ehingen-kirchen": 120,
                   "merklingen": 121,
                   "hayingen": 122,
                   "kupferzell": 123,
                   "oehringen": 124,
                   "vellberg-kleinaltdorf": 125,
}

locations = ["freiburg", "lahr", "muellheim", 
             "stockach", "gottmadingen", "weingarten",
             "eppingen-elsenz", "bruchsal-heidelsheim", "bretten",
             "ehingen-kirchen", "merklingen", "hayingen",
             "kupferzell", "oehringen", "vellberg-kleinaltdorf"]

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
                           "miscanthus",
                           "bare-grass",
                           "winter-wheat_silage-corn_yellow-mustard",
                           "summer-wheat_winter-wheat_yellow-mustard",
                           "winter-wheat_sugar-beet_silage-corn_yellow-mustard",
                           "summer-wheat_winter-wheat_silage-corn_yellow-mustard",
                           "summer-wheat_winter-wheat_winter-rape_yellow-mustard",
                           "sugar-beet_winter-wheat_winter-barley_yellow-mustard", 
                           "grain-corn_winter-wheat_winter-rape_yellow-mustard", 
                           "grain-corn_winter-wheat_winter-barley_yellow-mustard"]

fertilization_intensities = ["low", "medium", "high"]

def nanmeanweighted(y, w, axis=None):
    w1 = w / onp.nansum(w, axis=axis)
    w2 = onp.where(onp.isnan(w), 0, w1)
    w3 = onp.where(onp.isnan(y), 0, w2)
    y1 = onp.where(onp.isnan(y), 0, y)
    wavg = onp.sum(y1 * w3, axis=axis) / onp.sum(w3, axis=axis)

    return wavg

@click.option("-td", "--tmp-dir", type=str, default=Path(__file__).parent / "output")
@click.command("main")
def main(tmp_dir):
    base_path = Path(__file__).parent
    # directory of results
    base_path_output = Path(__file__).parent / "output"
    # base_path_output = Path(__file__).parent / "output"

    # load linkage between BK50 and cropland clusters
    file = Path(__file__).parent / "output" / "link_cluster_geometries_cropland.h5"
    df_link_bk50_cluster_cropland = pd.read_hdf(file)

    # load model parameters
    csv_file = base_path / "parameters.csv"
    df_params = pd.read_csv(csv_file, sep=";", skiprows=1)
    cond = (df_params["CLUST_flag"] == 1)
    df_params = df_params.loc[cond, :]
    clust_ids = pd.unique(df_params["CLUST_ID"].values).tolist()
    df_values = df_params.loc[:, ["CLUST_ID", "SHP_ID"]]
    df_values["ID"] = 0
    for clust_id in clust_ids:
        cond = (df_values["CLUST_ID"] == clust_id)
        df = df_values.loc[cond, :]
        ll_ids = df["CLUST_ID"].values[0].split("-")
        id1 = "".join(ll_ids)
        df_values.loc[cond, "ID"] = id1
    df_values_ = df_values.copy()
    df_params = pd.read_csv(csv_file, sep=";", skiprows=1)

    # write geometries
    file = base_path_output / "BK50_NBiomasseBW_for_assignment.gpkg"
    gdf1 = gpd.read_file(file)
    ll_dfs = []
    for location in locations:
        mask = (gdf1['stationsna'] == location)
        gdf = gdf1.loc[mask, :]
        gdf['OID'] = None
        gdf['MID'] = None
        gdf['MID'] = _dict_locations[location]
        gdf['SID'] = None
        for clust_id in clust_ids:
            cond = (df_link_bk50_cluster_cropland["CLUST_ID"] == clust_id)
            shp_ids = df_link_bk50_cluster_cropland.loc[cond, :].index.tolist()
            cond2 = gdf["SHP_ID"].isin(shp_ids)
            if cond2.any():
                id1 = int(df_values.loc[df_values["CLUST_ID"] == clust_id, "ID"].values[0])
                gdf.loc[cond2, 'SID'] = int(f"{_dict_clust_id[id1]}")
                gdf.loc[cond2, 'OID'] = int(f"{_dict_locations[location]}{_dict_clust_id[id1]}")
        gdf = gdf.copy()
        gdf = gdf.to_crs("EPSG:25832")
        ll_dfs.append(gdf)
    
    gdf = pd.concat(ll_dfs, axis=0)
    gdf = gdf.drop(columns=["SHP_ID", "area"])
    gdf = gdf[["OID", "MID", "SID", "stationsna", "agr_region", "geometry"]]
    gdf = gdf.to_crs("EPSG:25832")
    file = Path(tmp_dir) / "nitrate_leaching_geometries.gpkg"
    gdf.to_file(file, driver="GPKG")
    file = Path(tmp_dir) / "nitrate_leaching_geometries.shp"
    gdf.to_file(file)

    # load nitrogen loads and concentrations
    dict_nitrate = {}
    for location in locations:
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
    for location in locations:
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
    vars_sim = ["q_hof", "q_ss", "M_q_ss", "C_q_ss", "prec", "Nfert"]
    ll_df = []
    for location in locations:
        click.echo(f"{location}")
        for crop_rotation_scenario in crop_rotation_scenarios:
            click.echo(f"{crop_rotation_scenario}")
            df_values = df_values_.copy()
            df_values['stationsna'] = location
            df_values['agr_region'] = None
            if location in ["freiburg", "lahr", "muellheim"]:
                df_values['agr_region'] = "upper rhine valley"
            elif location in ["stockach", "gottmadingen", "weingarten"]:
                df_values['agr_region'] = "lake constance"
            elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                df_values['agr_region'] = "kraichgau"
            elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                df_values['agr_region'] = "alb-danube"
            elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                df_values['agr_region'] = "hohenlohe"
            df_values['OID'] = None
            df_values['MID'] = None
            df_values['MID'] = _dict_locations[location]
            df_values['SID'] = None
            df_values['FFID'] = None
            df_values['FFID'] = _dict_ffid[crop_rotation_scenario]
            df_values['CID'] = None
            df_values['CID'] = 400
            for var_sim in vars_sim:
                click.echo(f"{var_sim}")
                if var_sim in ["M_q_ss", "Nfert", "C_q_ss"]:
                    for fertilization_intensity in fertilization_intensities:
                        click.echo(f"{fertilization_intensity}")
                        df_values[f'{_dict_var_names[var_sim]}_N{_dict_fert[fertilization_intensity]}'] = None
                        ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert']
                        if var_sim == "M_q_ss":
                            sim_vals = ds[var_sim].isel(y=0).values[:, 1:-1]
                            df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T)
                            # calculate annual sum
                            df_ann = df.resample("YE").sum() * 0.01  # convert from mg/m2 to kg/ha
                            # calculate average
                            df_avg = df_ann.mean(axis=0).to_frame()
                        elif var_sim == "Nfert":
                            sim_vals = ds[var_sim].isel(y=0).values[:, 1:-1]
                            df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T)
                            # calculate annual sum
                            df_ann = df.resample("YE").sum() * 0.01  # convert from mg/m2 to kg/ha
                            # calculate average
                            df_avg = df_ann.mean(axis=0).to_frame()
                        elif var_sim == "C_q_ss":
                            ds = dict_fluxes_states[location][crop_rotation_scenario]
                            sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
                            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
                            sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
                            sim_vals = onp.where(sim_vals1 > 0.01, (sim_vals2/sim_vals1) * (sim_vals1/onp.sum(sim_vals1, axis=-1)[:, onp.newaxis]), onp.nan)  # weighted average
                            cond1 = (df_params["CLUST_flag"] == 1)
                            df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
                            # calculate annual mean
                            df_avg = df.sum(axis=0).to_frame()
                            df_avg.loc[:, "location"] = location
                        # assign aggregated values to polygons
                        for clust_id in clust_ids:
                            cond1 = (df_params["CLUST_ID"] == clust_id) & (df_params["CLUST_flag"] == 1)
                            val = df_avg.loc[cond1, :].values[0][0]
                            cond = (df_link_bk50_cluster_cropland["CLUST_ID"] == clust_id)
                            if cond.any():
                                cond2 = (df_values["CLUST_ID"] == clust_id)
                                id1 = int(df_values.loc[df_values["CLUST_ID"] == clust_id, "ID"].values[0])
                                df_values.loc[cond2, 'SID'] = int(f"{id1}")
                                df_values.loc[cond2, 'OID'] = int(f"{_dict_locations[location]}{_dict_clust_id[id1]}")
                                cond3 = (df_values['OID'] == int(f"{_dict_locations[location]}{_dict_clust_id[id1]}"))
                                df_values.loc[cond3, f'{_dict_var_names[var_sim]}_N{_dict_fert[fertilization_intensity]}'] = onp.round(val, 2)
                elif var_sim in ["q_ss", "q_hof", "prec"]:
                    df_values[f'{_dict_var_names[var_sim]}'] = None  # initialize field, float, two decimals
                    ds = dict_fluxes_states[location][crop_rotation_scenario]
                    sim_vals = ds[var_sim].isel(y=0).values[:, 1:]
                    df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T)
                    # calculate annual sum
                    df_ann = df.resample("YE").sum()
                    # calculate average
                    df_avg = df_ann.mean(axis=0).to_frame()

                    # assign aggregated values to polygons
                    for clust_id in clust_ids:
                        cond1 = (df_params["CLUST_ID"] == clust_id) & (df_params["CLUST_flag"] == 1)
                        val = df_avg.loc[cond1, :].values[0][0]
                        cond = (df_link_bk50_cluster_cropland["CLUST_ID"] == clust_id)
                        if cond.any():
                            cond2 = (df_values["CLUST_ID"] == clust_id)
                            id1 = int(df_values.loc[df_values["CLUST_ID"] == clust_id, "ID"].values[0])
                            df_values.loc[cond2, 'SID'] = int(f"{id1}")
                            df_values.loc[cond2, 'OID'] = int(f"{_dict_locations[location]}{_dict_clust_id[id1]}")
                            cond3 = (df_values['OID'] == int(f"{_dict_locations[location]}{_dict_clust_id[id1]}"))
                            df_values.loc[cond3, f'{_dict_var_names[var_sim]}'] = onp.round(val, 2)
                df_values1 = df_values.copy()
            df_values1 = df_values1.dropna()
            ll_df.append(df_values1)

            # extract the values for the single crop periods within the crop rotation
            crop_ids = _dict_crop_periods[crop_rotation_scenario]
            click.echo(f"{crop_ids}")
            for crop_id in crop_ids:
                df_values = df_values_.copy()
                df_values['stationsna'] = location
                df_values['agr_region'] = None
                if location in ["freiburg", "lahr", "muellheim"]:
                    df_values['agr_region'] = "upper rhine valley"
                elif location in ["stockach", "gottmadingen", "weingarten"]:
                    df_values['agr_region'] = "lake constance"
                elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                    df_values['agr_region'] = "kraichgau"
                elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                    df_values['agr_region'] = "alb-danube"
                elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                    df_values['agr_region'] = "hohenlohe"
                df_values['OID'] = None
                df_values['MID'] = None
                df_values['MID'] = _dict_locations[location]
                df_values['SID'] = None
                df_values['FFID'] = None
                df_values['FFID'] = _dict_ffid[crop_rotation_scenario]
                df_values['CID'] = None
                df_values['CID'] = int(f'{_dict_crop_id[crop_id]}')

                ds = dict_fluxes_states[location][crop_rotation_scenario]
                lu_id = ds["lu_id"].isel(x=0, y=0).values[1:]
                df_lu_id = pd.DataFrame(index=ds["Time"].values[1:], data=lu_id)
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
                for var_sim in vars_sim:
                    if var_sim in ["M_q_ss", "C_q_ss", "Nfert"]:
                        for fertilization_intensity in fertilization_intensities:
                            click.echo(f"{crop_id}: {var_sim} {fertilization_intensity}")
                            df_values[f'{_dict_var_names[var_sim]}_N{_dict_fert[fertilization_intensity]}'] = None  # initialize field, float, two decimals
                            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert']
                            if var_sim == "M_q_ss":
                                sim_vals = ds[var_sim].isel(y=0).values[:, 1:-1]
                                df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T)
                                df.loc[:, 'crop_period'] = df_lu_id.loc[:, 'crop_period'].values
                                df = df.loc[cond_crop, :]
                                # calculate sum per crop period
                                weight = 365/df_lu_id.groupby("crop_period").sum()["day"].values
                                weight = onp.where(weight > 1, 1, weight)
                                df_cp = df.groupby("crop_period").sum() * weight[:, onp.newaxis] * 0.01  # convert from mg/m2 to kg/ha                            
                                # calculate average
                                df_avg = df_cp.mean(axis=0).to_frame()
                            elif var_sim == "Nfert":
                                sim_vals = ds[var_sim].isel(y=0).values[:, 1:-1]
                                df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T)
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
                                sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:][:, cond_crop]
                                ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
                                sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1][:, cond_crop] * 4.427  # convert nitrate-nitrogen to nitrate
                                sim_vals = onp.where(sim_vals1 > 0.01, (sim_vals2/sim_vals1) * (sim_vals1/onp.sum(sim_vals1, axis=-1)[:, onp.newaxis]), onp.nan)  # weighted average
                                cond1 = (df_params["CLUST_flag"] == 1)
                                df = pd.DataFrame(data=sim_vals.T).loc[:, cond1]
                                # calculate annual mean
                                df_avg = df.sum(axis=0).to_frame()
                            # assign aggregated values to polygons
                            for clust_id in clust_ids:
                                cond1 = (df_params["CLUST_ID"] == clust_id) & (df_params["CLUST_flag"] == 1)
                                val = df_avg.loc[cond1, :].values[0][0]
                                cond = (df_link_bk50_cluster_cropland["CLUST_ID"] == clust_id)
                                if cond.any():
                                    cond2 = (df_values["CLUST_ID"] == clust_id)
                                    id1 = int(df_values.loc[df_values["CLUST_ID"] == clust_id, "ID"].values[0])
                                    df_values.loc[cond2, 'SID'] = int(f"{id1}")
                                    df_values.loc[cond2, 'OID'] = int(f"{_dict_locations[location]}{_dict_clust_id[id1]}")
                                    cond3 = (df_values['OID'] == int(f"{_dict_locations[location]}{_dict_clust_id[id1]}"))
                                    df_values.loc[cond3, f'{_dict_var_names[var_sim]}_N{_dict_fert[fertilization_intensity]}'] = onp.round(val, 2)
                    elif var_sim in ["q_ss", "q_hof", "prec"]:
                        click.echo(f"{crop_id}: {var_sim}")
                        df_values[f'{_dict_var_names[var_sim]}'] = None  # initialize field, float, two decimals
                        ds = dict_fluxes_states[location][crop_rotation_scenario]
                        sim_vals = ds[var_sim].isel(y=0).values[:, 1:]
                        df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T)
                        df.loc[:, 'crop_period'] = df_lu_id.loc[:, 'crop_period'].values
                        df = df.loc[cond_crop, :]
                        # calculate sum per crop period
                        weight = 365/df_lu_id.groupby("crop_period").sum()["day"].values
                        weight = onp.where(weight > 1, 1, weight)
                        df_cp = df.groupby("crop_period").sum() * weight[:, onp.newaxis]                          
                        # calculate average
                        df_avg = df_cp.mean(axis=0).to_frame()
                        # assign aggregated values to polygons
                        for clust_id in clust_ids:
                            cond1 = (df_params["CLUST_ID"] == clust_id) & (df_params["CLUST_flag"] == 1)
                            val = df_avg.loc[cond1, :].values[0][0]
                            cond = (df_link_bk50_cluster_cropland["CLUST_ID"] == clust_id)
                            if cond.any():
                                cond2 = (df_values["CLUST_ID"] == clust_id)
                                id1 = int(df_values.loc[df_values["CLUST_ID"] == clust_id, "ID"].values[0])
                                df_values.loc[cond2, 'SID'] = int(f"{id1}")
                                df_values.loc[cond2, 'OID'] = int(f"{_dict_locations[location]}{_dict_clust_id[id1]}")
                                cond3 = (df_values['OID'] == int(f"{_dict_locations[location]}{_dict_clust_id[id1]}"))
                                df_values.loc[cond3, f'{_dict_var_names[var_sim]}'] = onp.round(val, 2)
                    df_values1 = df_values.copy()
                df_values1 = df_values1.dropna()
                ll_df.append(df_values1)

            click.echo(f"Finalized {location}-{crop_rotation_scenario}")
        click.echo(f"Finalized {location}")

    df = pd.concat(ll_df, axis=0)
    df = df.drop(columns=["SHP_ID", "ID"])
    df = df[["OID", "MID", "SID", "agr_region", "stationsna", "FFID", "CID", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3", "PRECIP", "NFERT"]]

    df = df.fillna(-9999)
    file = Path(tmp_dir) / "nitrate_leaching_values.csv"
    df.to_csv(file, sep=";", index=False, header=True)

if __name__ == "__main__":
    main()