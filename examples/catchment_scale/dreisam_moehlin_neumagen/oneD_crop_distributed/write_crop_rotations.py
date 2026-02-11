from pathlib import Path
import xarray as xr
import geoxarray
import numpy as onp
import pandas as pd
import datetime
import roger.lookuptables as lut

summer_crops = lut.SUMMER_CROPS.tolist()
winter_crops = lut.WINTER_CROPS.tolist()

base_path = Path(__file__).parent

# load the netcdf file containing the parameters of RoGeR
params_file = base_path / "input" / "parameters_roger_25m__.nc"
ds_params = xr.open_dataset(params_file)
xcoords = ds_params.x.values
ycoords = ds_params.y.values

# load the netcdf file containing crop rotations from 2018-2022 encoded as lu_id
file = base_path / "input" / "crops_2018-2022.nc"
ds_cr_2018_2022 = xr.open_dataset(file)
spatial_ref = ds_cr_2018_2022.spatial_ref
lu_ids_2018_2022 = ds_cr_2018_2022['Nutzcode'].values
cond = onp.isnan(lu_ids_2018_2022)
lu_ids_2018_2022[cond] = -9999  # set nan to
lu_ids_2018_2022 = lu_ids_2018_2022.astype(onp.int16)

years_2018_2022 = onp.arange(2018, 2023)
years_2013_2023 = onp.arange(2013, 2024)
years_2000_2024 = onp.arange(2000, 2025)

# extend annual crop types to years 2013-2023 and 2000-2023
lu_ids_2013_2023 = onp.zeros((len(years_2013_2023), lu_ids_2018_2022.shape[1], lu_ids_2018_2022.shape[2]), dtype=onp.int16)
lu_ids_2013_2023[:5, :, :] = lu_ids_2018_2022
lu_ids_2013_2023[5:10, :, :] = lu_ids_2018_2022
lu_ids_2013_2023[-1, :, :] = lu_ids_2018_2022[0, :, :]

lu_ids_2000_2024 = onp.zeros((len(years_2000_2024), lu_ids_2018_2022.shape[1], lu_ids_2018_2022.shape[2]), dtype=onp.int16)
lu_ids_2000_2024[0:3, :, :] = lu_ids_2018_2022[2:5, :, :]
lu_ids_2000_2024[3:8, :, :] = lu_ids_2018_2022
lu_ids_2000_2024[8:13, :, :] = lu_ids_2018_2022
lu_ids_2000_2024[13:18, :, :] = lu_ids_2018_2022
lu_ids_2000_2024[18:23, :, :] = lu_ids_2018_2022
lu_ids_2000_2024[23:25, :, :] = lu_ids_2018_2022[0:2, :, :]

# create xarray dataset
attrs = dict(
        date_created=datetime.datetime.today().isoformat(),
        title="lu_id of RoGeR in the Dreisam-Moehlin-Neumagen catchment for the years 2013-2023",
        institution="University of Freiburg, Chair of Hydrology",
    )
coords = {
        "lon": ("lon", xcoords),  # x
        "lat": ("lat", ycoords),  # y
        "Year": ("Year", years_2013_2023),
    }
data_vars=dict(
        crop_type=(["Year", "lat", "lon"], lu_ids_2013_2023),
    )

ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
ds["crop_type"].attrs["units"] = ""
ds["crop_type"].attrs["long_name"] = "Crop type encoded as lu_id from RoGeR"
# create spatial reference
ds = ds.geo.write_crs("EPSG:25832")
ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
file = base_path / "input" / "crops_2013-2023.nc"
comp = dict(zlib=True, complevel=1)  # compress data to save storage
encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)

# create xarray dataset
attrs = dict(
        date_created=datetime.datetime.today().isoformat(),
        title="lu_id of RoGeR in the Dreisam-Moehlin-Neumagen catchment for the years 2000-2023",
        institution="University of Freiburg, Chair of Hydrology",
    )
coords = {
        "lon": ("lon", xcoords),  # x
        "lat": ("lat", ycoords),  # y
        "Year": ("Year", years_2000_2024),
    }
data_vars=dict(
        crop_type=(["Year", "lat", "lon"], lu_ids_2000_2024),
    )

ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
ds["crop_type"].attrs["units"] = ""
ds["crop_type"].attrs["long_name"] = "Crop type encoded as lu_id from RoGeR"
# create spatial reference
ds = ds.geo.write_crs("EPSG:25832")
ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
file = base_path / "input" / "crops_2000-2023.nc"
comp = dict(zlib=True, complevel=1)  # compress data to save storage
encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)

# create csv file for crop rotations
years_header = []
for year in [2018, 2019, 2020, 2021, 2022]:
    years_header.append(f"{year}")
df_crops = pd.DataFrame(columns=years_header)
for t, year in enumerate([2018, 2019, 2020, 2021, 2022]):
    lu_ids_year = lu_ids_2018_2022[t, :, :].flatten()
    df_crops.loc[:, f"{year}"] = lu_ids_year
_df_crops = df_crops.copy()
# count occurrences of values in each column
crop_counts = df_crops.apply(pd.Series.value_counts).fillna(0)
crop_counts.loc[:, 'total'] = crop_counts.sum(axis=1)

file = base_path / "input" / "crops_2018_2022_summary.csv"
crop_counts.to_csv(file, sep=";", index=True)

_df_crop_rotations_2018_2022 = df_crops.copy()
# count occurence of identical crop rotations and calculate area in ha and area share
_df_crop_rotations_2018_2022.loc[:, "count"] = _df_crop_rotations_2018_2022.groupby(years_header).cumcount() + 1
_df_crop_rotations_2018_2022.loc[:, "area_ha"] = _df_crop_rotations_2018_2022["count"] * 0.0625  # 25m x 25m = 625m2 = 0.0625ha
_df_crop_rotations_2018_2022 = _df_crop_rotations_2018_2022.drop_duplicates(keep='last')
_df_crop_rotations_2018_2022.loc[:, "area_share"] = _df_crop_rotations_2018_2022["count"] / _df_crop_rotations_2018_2022["count"].sum()
file = base_path / "input" / "available_crop_rotations_2018_2022.csv"
_df_crop_rotations_2018_2022.to_csv(file, sep=";", index=False)

cond0 = onp.isin(_df_crops.loc[:, "2019":].values, [8, 81, 82]).all(axis=1)
cond1 = _df_crops["2018"] == 8
df_crops.loc[cond0 & cond1, "2018"] = 573
cond3 = _df_crops["2018"] == 81
df_crops.loc[cond0 & cond3, "2018"] = 596
cond5 = _df_crops["2018"] == 82
df_crops.loc[cond0 & cond5, "2018"] = 594

# convert lu_id=8 (grassland) to lu_id=571 (grassland without fertilisation)
for year1, year2 in zip([2018, 2019, 2020, 2021], [2019, 2020, 2021, 2022]):
    cond1 = (_df_crops[f"{year1}"] != 8) & onp.isin(_df_crops[f"{year1}"], summer_crops)
    cond2 = _df_crops[f"{year2}"] == 8
    df_crops.loc[cond1 & cond2, f"{year2}"] = 571
    cond3 = (_df_crops[f"{year1}"] == 8) & onp.isin(_df_crops[f"{year2}"], onp.arange(500, 600).tolist())
    df_crops.loc[cond3, f"{year1}"] = 571

# convert lu_id=8 (grassland) to lu_id=572 (grassland without fertilisation)
for year1, year2 in zip([2018, 2019, 2020, 2021], [2019, 2020, 2021, 2022]):
    cond1 = (_df_crops[f"{year1}"] != 8) & onp.isin(_df_crops[f"{year1}"], winter_crops)
    cond2 = _df_crops[f"{year2}"] == 8
    df_crops.loc[cond1 & cond2, f"{year2}"] = 572

for year1, year2 in zip([2018, 2019, 2020, 2021], [2019, 2020, 2021, 2022]):
    # convert lu_id=8 (grassland) to lu_id=573 (grassland with fertilisation)
    cond1 = _df_crops[f"{year1}"] == 8
    cond2 = _df_crops[f"{year2}"] == 8
    df_crops.loc[cond1 & cond2, f"{year2}"] = 573
    # convert lu_id=81 (grassland) to lu_id=592 (extensive grassland)
    cond3 = _df_crops[f"{year1}"] == 81
    cond4 = _df_crops[f"{year2}"] == 81
    df_crops.loc[cond3 & cond4, f"{year1}"] = 592
    # convert lu_id=82 (grassland) to lu_id=565 (intensive grassland)
    cond5 = _df_crops[f"{year1}"] == 82
    cond6 = _df_crops[f"{year2}"] == 82
    df_crops.loc[cond5 & cond6, f"{year1}"] = 565

# convert lu_id=81 (grassland) to lu_id=592 (extensive grassland)
for year1, year2 in zip([2018, 2019, 2020, 2021], [2019, 2020, 2021, 2022]):
    cond1 = (_df_crops[f"{year1}"] != 81) & onp.isin(_df_crops[f"{year1}"], summer_crops)
    cond2 = _df_crops[f"{year2}"] == 81
    df_crops.loc[cond1 & cond2, f"{year2}"] = 592
    cond3 = (_df_crops[f"{year1}"] == 81) & onp.isin(_df_crops[f"{year2}"], onp.arange(500, 600).tolist())
    df_crops.loc[cond3, f"{year1}"] = 592

# convert lu_id=81 (grassland) to lu_id=593 (extensive grassland)
for year1, year2 in zip([2018, 2019, 2020, 2021], [2019, 2020, 2021, 2022]):
    cond1 = (_df_crops[f"{year1}"] != 81) & onp.isin(_df_crops[f"{year1}"], winter_crops)
    cond2 = _df_crops[f"{year2}"] == 81
    df_crops.loc[cond1 & cond2, f"{year2}"] = 593

# convert lu_id=81 (grassland) to lu_id=594 (extensive grassland)
for year1, year2 in zip([2018, 2019, 2020, 2021], [2019, 2020, 2021, 2022]):
    cond1 = _df_crops[f"{year1}"] == 81
    cond2 = _df_crops[f"{year2}"] == 81
    df_crops.loc[cond1 & cond2, f"{year2}"] = 594

# convert lu_id=82 (grassland) to lu_id=565 (intensive grassland)
for year1, year2 in zip([2018, 2019, 2020, 2021], [2019, 2020, 2021, 2022]):
    cond1 = (_df_crops[f"{year1}"] != 82) & onp.isin(_df_crops[f"{year1}"], summer_crops)
    cond2 = _df_crops[f"{year2}"] == 82
    df_crops.loc[cond1 & cond2, f"{year2}"] = 565
    cond3 = (_df_crops[f"{year1}"] == 82) & onp.isin(_df_crops[f"{year2}"], onp.arange(500, 600).tolist())
    df_crops.loc[cond3, f"{year1}"] = 565

# convert lu_id=82 (grassland) to lu_id=566 (intensive grassland)
for year1, year2 in zip([2018, 2019, 2020, 2021], [2019, 2020, 2021, 2022]):
    cond1 = (_df_crops[f"{year1}"] != 82) & onp.isin(_df_crops[f"{year1}"], winter_crops)
    cond2 = _df_crops[f"{year2}"] == 82
    df_crops.loc[cond1 & cond2, f"{year2}"] = 566

# convert lu_id=82 (grassland) to lu_id=596 (intensive grassland)
for year1, year2 in zip([2018, 2019, 2020, 2021], [2019, 2020, 2021, 2022]):
    cond1 = _df_crops[f"{year1}"] == 82
    cond2 = _df_crops[f"{year2}"] == 82
    df_crops.loc[cond1 & cond2, f"{year2}"] = 596

# convert lu_id=589 (misscanthus) to lu_id=591 (misscanthus continued)
for year1, year2 in zip([2018, 2019, 2020, 2021], [2019, 2020, 2021, 2022]):
    cond1 = _df_crops[f"{year1}"] == 589
    cond2 = _df_crops[f"{year2}"] == 589
    df_crops.loc[cond1 & cond2, f"{year2}"] = 591

# set lu_id=6 (vineyard) and lu_id=7 (orchard)
cond = (_df_crops == 6).any(axis=1)
df_crops.loc[cond, :] = 6
cond = (_df_crops == 7).any(axis=1)
df_crops.loc[cond, :] = 7

# write crops of the years 2018 to 2022 to csv
df_crops.index = df_crops.index + 1
file = base_path / "input" / "crops_2018_2022.csv"
df_crops.to_csv(file, sep=";", index=True)
df_crops.index = df_crops.index - 1

# write crop rotations required by RoGeR
unit_header = [""]
years_header = ["No"]
for year in [2018, 2019, 2020, 2021, 2022]:
    unit_header.append("[year_season]")
    unit_header.append("[year_season]")
    years_header.append(f"{year}_summer")
    years_header.append(f"{year}_winter")
df_crop_rotations = pd.DataFrame(index=range(len(df_crops)), columns=years_header)
for year1, year2 in zip([2018, 2019, 2020, 2021], [2019, 2020, 2021, 2022]):
    cond_summer_crops = onp.isin(df_crops[f"{year1}"].values, summer_crops)
    # define crop rotations of summer crops
    df_crop_rotations.loc[cond_summer_crops, f"{year1}_summer"] = df_crops.loc[cond_summer_crops, f"{year1}"]
    # define crop rotations of winter crops
    cond_winter_crops = onp.isin(df_crops[f"{year1}"].values, winter_crops)
    cond_summer1_crops = onp.isin(df_crops[f"{year2}"].values, summer_crops)
    cond_winter1_crops = onp.isin(df_crops[f"{year2}"].values, winter_crops)
    # move winter crops to previous year's winter slot since RoGeR assumes that winter crops are sown in autumn of the previous year
    df_crop_rotations.loc[cond_winter1_crops, f"{year1}_winter"] = df_crops.loc[cond_winter1_crops, f"{year2}"]
    df_crop_rotations.loc[cond_winter1_crops, f"{year2}_summer"] = 599
    df_crop_rotations.loc[cond_winter1_crops, f"{year2}_winter"] = 599
    # define crop rotations for grassland (no fertilisation)
    cond_571 = df_crops[f"{year1}"] == 571
    df_crop_rotations.loc[cond_571, f"{year1}_summer"] = 571  # sowing period
    df_crop_rotations.loc[cond_571, f"{year1}_winter"] = 574  # continuation period
    cond_572 = df_crops[f"{year1}"] == 572
    df_crop_rotations.loc[cond_572, f"{year1}_winter"] = 572  # sowing period
    df_crop_rotations.loc[cond_572, f"{year2}_summer"] = 573  # continuation period
    # define crop rotations for extensive grassland
    cond_592 = df_crops[f"{year1}"] == 592
    df_crop_rotations.loc[cond_592, f"{year1}_summer"] = 592  # sowing period
    df_crop_rotations.loc[cond_592, f"{year1}_winter"] = 595 # continuation period
    cond_593 = df_crops[f"{year1}"] == 593
    df_crop_rotations.loc[cond_593, f"{year1}_winter"] = 593  # sowing period
    df_crop_rotations.loc[cond_593, f"{year2}_summer"] = 594  # continuation period
    # define crop rotations for intensive grassland
    cond_565 = df_crops[f"{year1}"] == 565
    df_crop_rotations.loc[cond_565, f"{year1}_summer"] = 565  # sowing period
    df_crop_rotations.loc[cond_565, f"{year1}_winter"] = 597  # continuation period
    cond_566 = df_crops[f"{year1}"] == 566
    df_crop_rotations.loc[cond_566, f"{year1}_winter"] = 566  # sowing period
    df_crop_rotations.loc[cond_566, f"{year2}_summer"] = 596  # continuation period
    cond_591 = df_crops[f"{year1}"] == 591
    df_crop_rotations.loc[cond_591, f"{year2}_summer"] = 591  # sowing period
    df_crop_rotations.loc[cond_591, f"{year1}_winter"] = 590  # continuation period
for year1, year2 in zip([2018, 2019, 2020, 2021], [2019, 2020, 2021, 2022]):
    # define crop rotations for intensive grassland continued over multiple years
    cond_596 = (df_crop_rotations.loc[:, f"{year1}_summer"] == 596) & (df_crop_rotations.loc[:, f"{year2}_summer"] == 596)
    df_crop_rotations.loc[cond_596, f"{year1}_winter"] = 597
    # define crop rotations for extensive grassland continued over multiple years
    cond_594 = (df_crop_rotations.loc[:, f"{year1}_summer"] == 594) & (df_crop_rotations.loc[:, f"{year2}_summer"] == 594)
    df_crop_rotations.loc[cond_594, f"{year1}_winter"] = 595

# set grassland of year 2022
cond_596 = (df_crops["2022"] == 596)
df_crop_rotations.loc[cond_596, "2022_summer"] = 596
cond_594 = (df_crops["2022"] == 594)
df_crop_rotations.loc[cond_594, "2022_summer"] = 594
# set summer crops of year 2022
cond_summer_crops_2022 = onp.isin(df_crops["2022"].values, summer_crops)
df_crop_rotations.loc[cond_summer_crops_2022, "2022_summer"] = df_crops.loc[cond_summer_crops_2022, "2022"]

df_crop_rotations.loc[:, "No"] = df_crop_rotations.index + 1
for col in df_crop_rotations.columns[1:]:
    df_crop_rotations.loc[:, col] = df_crop_rotations.loc[:, col].astype(float)
# if any value in row is finite replace nan with 599
cond = onp.isfinite(df_crop_rotations.loc[:, years_header[1:]].astype(float)).any(axis=1)
df_crop_rotations.loc[cond, years_header[1:]] = df_crop_rotations.loc[cond, years_header[1:]].astype(float).fillna(599).infer_objects(copy=False)
# if all values in row are NaN replace nan with 598
cond = onp.isnan(df_crop_rotations.loc[:, years_header[1:]].astype(float)).all(axis=1)
df_crop_rotations.loc[cond, years_header[1:]] = df_crop_rotations.loc[cond, years_header[1:]].astype(float).fillna(598).infer_objects(copy=False)

for year1, year2 in zip([2018, 2019, 2020, 2021], [2019, 2020, 2021, 2022]):
    cond_596_ = (df_crop_rotations.loc[:, f"{year1}_summer"] == 596) & (df_crop_rotations.loc[:, f"{year2}_summer"] == 599)
    df_crop_rotations.loc[cond_596_, f"{year1}_winter"] = 597
    cond_594_ = (df_crop_rotations.loc[:, f"{year1}_summer"] == 594) & (df_crop_rotations.loc[:, f"{year2}_summer"] == 599)
    df_crop_rotations.loc[cond_594_, f"{year1}_winter"] = 595
    cond_596_ = (df_crop_rotations.loc[:, f"{year2}_summer"] == 596) & (df_crop_rotations.loc[:, f"{year2}_winter"] == 599)
    df_crop_rotations.loc[cond_596_, f"{year2}_winter"] = 597
    cond_594_ = (df_crop_rotations.loc[:, f"{year2}_summer"] == 594) & (df_crop_rotations.loc[:, f"{year2}_winter"] == 599)
    df_crop_rotations.loc[cond_594_, f"{year2}_winter"] = 595

# set winter crops of year 2022
cond_winter_crops_2018 = onp.isin(df_crops["2018"].values, winter_crops)
cond_bare_2018 = (df_crop_rotations.loc[:, "2018_summer"].values == 599) & (df_crop_rotations.loc[:, "2018_winter"].values == 599)
_cond_winter_crops_2022 = onp.isin(df_crop_rotations.loc[:, "2022_summer"].values, summer_crops)
df_crop_rotations.loc[cond_winter_crops_2018 & cond_bare_2018 & _cond_winter_crops_2022, "2022_winter"] = df_crops.loc[cond_winter_crops_2018 & cond_bare_2018 & _cond_winter_crops_2022, "2018"]

cond_565_596 = (df_crops["2018"] == 565) & (df_crops["2022"] == 596)
df_crop_rotations.loc[cond_565_596, "2018_summer"] = 565
cond_592_594 = (df_crops["2018"] == 592) & (df_crops["2022"] == 594)
df_crop_rotations.loc[cond_592_594, "2022_summer"] = 594

for year in [2018, 2019, 2020, 2021, 2022]:
    cond_599 = onp.isin(df_crop_rotations.loc[:, f"{year1}_winter"].values, summer_crops)
    df_crop_rotations.loc[cond_599, f"{year}_winter"] = 599
    cond_599 = onp.isin(df_crop_rotations.loc[:, f"{year1}_summer"].values, winter_crops)
    df_crop_rotations.loc[cond_599, f"{year}_summer"] = 599

file = base_path / "input" / "crop_rotations_2018_2022.csv"
df_crop_rotations.to_csv(file, sep=";", index=False)

df_crop_rotations.fillna(-9999, inplace=True)
for col in df_crop_rotations.columns[1:]:
    df_crop_rotations.loc[:, col] = df_crop_rotations.loc[:, col].astype(int)
for i, year in enumerate([2018, 2019, 2020, 2021, 2022]):
    df_crop_rotations.loc[:, f"{year}_summer"] = df_crop_rotations.loc[:, f"{year}_summer"].astype(onp.int16)
    df_crop_rotations.loc[:, f"{year}_winter"] = df_crop_rotations.loc[:, f"{year}_winter"].astype(onp.int16)

summer_crops_2018_2022 = onp.zeros(lu_ids_2018_2022.shape, dtype=onp.int16)
winter_crops_2018_2022 = onp.zeros(lu_ids_2018_2022.shape, dtype=onp.int16)
for i, year in enumerate([2018, 2019, 2020, 2021, 2022]):
    summer_crops_2018_2022[i, :, :] = df_crop_rotations.loc[:, f"{year}_summer"].values.reshape(lu_ids_2018_2022.shape[1], lu_ids_2018_2022.shape[2])
    winter_crops_2018_2022[i, :, :] = df_crop_rotations.loc[:, f"{year}_winter"].values.reshape(lu_ids_2018_2022.shape[1], lu_ids_2018_2022.shape[2])

for i, year in enumerate([2018, 2019, 2020, 2021, 2022]):
    crops_year = df_crops.loc[:, f"{year}"].values.reshape(lu_ids_2018_2022.shape[1], lu_ids_2018_2022.shape[2])
    print(f"Year: {year}", summer_crops_2018_2022[i, 629, 265], winter_crops_2018_2022[i, 629, 265], crops_year[629, 265])

# extend crop rotations to years 2013-2023
summer_crops_2013_2023 = onp.zeros((len(years_2013_2023), lu_ids_2018_2022.shape[1], lu_ids_2018_2022.shape[2]), dtype=onp.int16)
summer_crops_2013_2023[:5, :, :] = summer_crops_2018_2022
summer_crops_2013_2023[5:10, :, :] = summer_crops_2018_2022
summer_crops_2013_2023[-1, :, :] = summer_crops_2018_2022[0, :, :]

winter_crops_2013_2023 = onp.zeros((len(years_2013_2023), lu_ids_2018_2022.shape[1], lu_ids_2018_2022.shape[2]), dtype=onp.int16)
winter_crops_2013_2023[:5, :, :] = winter_crops_2018_2022
winter_crops_2013_2023[5:10, :, :] = winter_crops_2018_2022
winter_crops_2013_2023[-1, :, :] = winter_crops_2018_2022[0, :, :]

for i, year in enumerate(years_2013_2023):
    print(f"Year: {year}({i})", summer_crops_2013_2023[i, 629, 265], winter_crops_2013_2023[i, 629, 265])

# extend crop rotations to years 2000-2024
summer_crops_2000_2024 = onp.zeros((len(years_2000_2024), lu_ids_2018_2022.shape[1], lu_ids_2018_2022.shape[2]), dtype=onp.int16)
summer_crops_2000_2024[0:3, :, :] = summer_crops_2018_2022[2:5, :, :]
summer_crops_2000_2024[3:8, :, :] = summer_crops_2018_2022
summer_crops_2000_2024[8:13, :, :] = summer_crops_2018_2022
summer_crops_2000_2024[13:18, :, :] = summer_crops_2018_2022
summer_crops_2000_2024[18:23, :, :] = summer_crops_2018_2022
summer_crops_2000_2024[23:25, :, :] = summer_crops_2018_2022[0:2, :, :]

winter_crops_2000_2024 = onp.zeros((len(years_2000_2024), lu_ids_2018_2022.shape[1], lu_ids_2018_2022.shape[2]), dtype=onp.int16)
winter_crops_2000_2024[0:3, :, :] = winter_crops_2018_2022[2:5, :, :]
winter_crops_2000_2024[3:8, :, :] = winter_crops_2018_2022
winter_crops_2000_2024[8:13, :, :] = winter_crops_2018_2022
winter_crops_2000_2024[13:18, :, :] = winter_crops_2018_2022
winter_crops_2000_2024[18:23, :, :] = winter_crops_2018_2022
winter_crops_2000_2024[23:25, :, :] = winter_crops_2018_2022[0:2, :, :]

for i, year in enumerate(years_2000_2024):
    print(f"Year: {year}({i})", summer_crops_2000_2024[i, 629, 265], winter_crops_2000_2024[i, 629, 265])

# write crop rotations to netcdf files
attrs = dict(
        date_created=datetime.datetime.today().isoformat(),
        title="lu_id of RoGeR in the Dreisam-Moehlin-Neumagen catchment for the years 2018-2022",
        institution="University of Freiburg, Chair of Hydrology",
    )
coords = {
        "lon": ("lon", xcoords),  # x
        "lat": ("lat", ycoords),  # y
        "Year": ("Year", years_2018_2022),
    }
data_vars=dict(
        summer_crops=(["Year", "lat", "lon"], summer_crops_2018_2022),
        winter_crops=(["Year", "lat", "lon"], winter_crops_2018_2022),
    )

ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
ds["summer_crops"].attrs["units"] = ""
ds["summer_crops"].attrs["long_name"] = "Summer crops encoded as lu_id from RoGeR"
ds["winter_crops"].attrs["units"] = ""
ds["winter_crops"].attrs["long_name"] = "Winter crops encoded as lu_id from RoGeR"
# create spatial reference
ds = ds.geo.write_crs("EPSG:25832")
ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
file = base_path / "input" / "crop_rotations_2018-2022.nc"
comp = dict(zlib=True, complevel=1)  # compress data to save storage
encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)


attrs = dict(
        date_created=datetime.datetime.today().isoformat(),
        title="lu_id of RoGeR in the Dreisam-Moehlin-Neumagen catchment for the years 2013-2023",
        institution="University of Freiburg, Chair of Hydrology",
    )
coords = {
        "lon": ("lon", xcoords),  # x
        "lat": ("lat", ycoords),  # y
        "Year": ("Year", years_2013_2023),
    }
data_vars=dict(
        summer_crops=(["Year", "lat", "lon"], summer_crops_2013_2023),
        winter_crops=(["Year", "lat", "lon"], winter_crops_2013_2023),
    )

ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
ds["summer_crops"].attrs["units"] = ""
ds["summer_crops"].attrs["long_name"] = "Summer crops encoded as lu_id from RoGeR"
ds["winter_crops"].attrs["units"] = ""
ds["winter_crops"].attrs["long_name"] = "Winter crops encoded as lu_id from RoGeR"
# create spatial reference
ds = ds.geo.write_crs("EPSG:25832")
ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
file = base_path / "input" / "crop_rotations_2013-2023.nc"
comp = dict(zlib=True, complevel=1)  # compress data to save storage
encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)

attrs = dict(
        date_created=datetime.datetime.today().isoformat(),
        title="lu_id of RoGeR in the Dreisam-Moehlin-Neumagen catchment for the years 2000-2023",
        institution="University of Freiburg, Chair of Hydrology",
    )
coords = {
        "lon": ("lon", xcoords),  # x
        "lat": ("lat", ycoords),  # y
        "Year": ("Year", years_2000_2024),
    }
data_vars=dict(
        summer_crops=(["Year", "lat", "lon"], summer_crops_2000_2024),
        winter_crops=(["Year", "lat", "lon"], winter_crops_2000_2024),
    )

ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
ds["summer_crops"].attrs["units"] = ""
ds["summer_crops"].attrs["long_name"] = "Summer crops encoded as lu_id from RoGeR"
ds["winter_crops"].attrs["units"] = ""
ds["winter_crops"].attrs["long_name"] = "Winter crops encoded as lu_id from RoGeR"
# create spatial reference
ds = ds.geo.write_crs("EPSG:25832")
ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
file = base_path / "input" / "crop_rotations_2000-2024.nc"
comp = dict(zlib=True, complevel=1)  # compress data to save storage
encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)

# create csv files for crop rotations
unit_header = [""]
years_header = ["No"]
for year in years_2013_2023:
    unit_header.append("[year_season]")
    unit_header.append("[year_season]")
    years_header.append(f"{year}_summer")
    years_header.append(f"{year}_winter")
df_crop_rotations_2013_2023 = pd.DataFrame(columns=years_header)
for i, year in enumerate(years_2013_2023):
    df_crop_rotations_2013_2023.loc[:, f"{year}_summer"] = summer_crops_2013_2023[i, :, :].flatten()
    df_crop_rotations_2013_2023.loc[:, f"{year}_winter"] = winter_crops_2013_2023[i, :, :].flatten()
df_crop_rotations_2013_2023.loc[:, "No"] = onp.arange(1, df_crop_rotations_2013_2023.shape[0]+1)
df_crop_rotations_2013_2023.columns = [unit_header, years_header]
file = base_path / "input" / "crop_rotations_2013_2023.csv"
df_crop_rotations_2013_2023.to_csv(file, sep=";", index=False)

unit_header = [""]
years_header = ["No"]
for year in years_2000_2024:
    unit_header.append("[year_season]")
    unit_header.append("[year_season]")
    years_header.append(f"{year}_summer")
    years_header.append(f"{year}_winter")
df_crop_rotations_2000_2024 = pd.DataFrame(columns=years_header)
for i, year in enumerate(years_2000_2024):
    df_crop_rotations_2000_2024.loc[:, f"{year}_summer"] = summer_crops_2000_2024[i, :, :].flatten()
    df_crop_rotations_2000_2024.loc[:, f"{year}_winter"] = winter_crops_2000_2024[i, :, :].flatten()
df_crop_rotations_2000_2024.loc[:, "No"] = onp.arange(1, df_crop_rotations_2000_2024.shape[0]+1)
df_crop_rotations_2000_2024.columns = [unit_header, years_header]
file = base_path / "input" / "crop_rotations_2000_2024.csv"
df_crop_rotations_2000_2024.to_csv(file, sep=";", index=False)