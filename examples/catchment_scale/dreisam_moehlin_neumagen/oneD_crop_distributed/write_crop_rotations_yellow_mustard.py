from pathlib import Path
import xarray as xr
import geoxarray
import numpy as onp
import pandas as pd
import datetime
import roger.lookuptables as lut
from matplotlib_map_utils.core.north_arrow import north_arrow
from matplotlib_map_utils.core.scale_bar import scale_bar

summer_crops = lut.SUMMER_CROPS.tolist()
winter_crops = lut.WINTER_CROPS.tolist()

base_path = Path(__file__).parent

years_2018_2022 = onp.arange(2018, 2023)
years_2013_2023 = onp.arange(2013, 2024)
years_2000_2024 = onp.arange(2000, 2025)

# get x and y coordinates from RoGeR parameter file
params_file = base_path / "input" / "parameters_roger_25m__.nc"
ds_params = xr.open_dataset(params_file)
xcoords = ds_params.x.values
ycoords = ds_params.y.values

# load the netcdf file containing crops encoded as lu_id from 2018-2022
file = base_path / "input" / "crops_2018-2022.nc"
ds_cr_2018_2022 = xr.open_dataset(file)
spatial_ref = ds_cr_2018_2022.spatial_ref
lu_ids_2018_2022 = ds_cr_2018_2022['Nutzcode'].values
lu_ids_2018_2022 = ds_cr_2018_2022['Nutzcode'].values
cond = onp.isnan(lu_ids_2018_2022)
lu_ids_2018_2022[cond] = -9999  # set nan to
lu_ids_2018_2022 = lu_ids_2018_2022.astype(onp.int16)

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

# read crop rotations from csv file
file = base_path / "input" / "crop_rotations_2018_2022.csv"
df_crop_rotations = pd.read_csv(file, sep=";", header=0)

_summer_crops = summer_crops.copy()
_summer_crops.remove(525)
_summer_crops.remove(539)

df_crop_rotations_yellow_mustard = df_crop_rotations.copy()
# insert yellow mustard before summer crops
for year1, year2 in zip([2018, 2019, 2020, 2021], [2019, 2020, 2021, 2022]):
    # define crop rotations of winter crops
    cond_summer_crop_year1 = onp.isin(df_crop_rotations_yellow_mustard.loc[:, f"{year1}_summer"], _summer_crops + [599])
    cond_winter_crop_year1 = (df_crop_rotations_yellow_mustard.loc[:, f"{year1}_winter"] == 599)
    cond_summer_crop_year2 = onp.isin(df_crop_rotations_yellow_mustard.loc[:, f"{year2}_summer"], summer_crops + [599])
    cond_yellow_mustard = cond_summer_crop_year1 & cond_winter_crop_year1 & cond_summer_crop_year2
    df_crop_rotations_yellow_mustard.loc[cond_yellow_mustard, f"{year1}_winter"] = 587

    cond_summer_crop_year1 = onp.isin(df_crop_rotations_yellow_mustard.loc[:, f"{year1}_summer"], [525, 539, 599])
    cond_winter_crop_year1 = (df_crop_rotations_yellow_mustard.loc[:, f"{year1}_winter"] == 599)
    cond_summer_crop_year2 = onp.isin(df_crop_rotations_yellow_mustard.loc[:, f"{year2}_summer"], summer_crops + [599])
    cond_yellow_mustard = cond_summer_crop_year1 & cond_winter_crop_year1 & cond_summer_crop_year2
    df_crop_rotations_yellow_mustard.loc[cond_yellow_mustard, f"{year1}_winter"] = 587

    cond_winter_crop_year1 = onp.isin(df_crop_rotations_yellow_mustard.loc[:, f"{year1}_winter"], [556, 557])
    cond_summer_crop_year2 = (df_crop_rotations_yellow_mustard.loc[:, f"{year2}_summer"] == 599)
    cond_winter_crop_year2 = (df_crop_rotations_yellow_mustard.loc[:, f"{year2}_winter"] == 599)
    cond_yellow_mustard = cond_winter_crop_year1 & cond_summer_crop_year2 & cond_winter_crop_year2
    df_crop_rotations_yellow_mustard.loc[cond_yellow_mustard, f"{year2}_winter"] = 586

cond_summer_crop_year1 = onp.isin(df_crop_rotations_yellow_mustard.loc[:, "2022_summer"], _summer_crops + [599])
cond_winter_crop_year1 = (df_crop_rotations_yellow_mustard.loc[:, "2022_winter"] == 599)
cond_summer_crop_year2 = onp.isin(df_crop_rotations_yellow_mustard.loc[:, "2018_summer"], summer_crops + [599])
cond_yellow_mustard = cond_summer_crop_year1 & cond_winter_crop_year1 & cond_summer_crop_year2
df_crop_rotations_yellow_mustard.loc[cond_yellow_mustard, "2022_winter"] = 587

cond_summer_crop_year1 = onp.isin(df_crop_rotations_yellow_mustard.loc[:, "2022_summer"], [525, 539, 599])
cond_winter_crop_year1 = (df_crop_rotations_yellow_mustard.loc[:, "2022_winter"] == 599)
cond_summer_crop_year2 = onp.isin(df_crop_rotations_yellow_mustard.loc[:, "2018_summer"], summer_crops + [599])
cond_yellow_mustard = cond_summer_crop_year1 & cond_winter_crop_year1 & cond_summer_crop_year2
df_crop_rotations_yellow_mustard.loc[cond_yellow_mustard, "2022_winter"] = 587

file = base_path / "input" / "crop_rotations_2018_2022_yellow_mustard.csv"
df_crop_rotations_yellow_mustard.to_csv(file, sep=";", index=False)

summer_crops_2018_2022_yellow_mustard = onp.zeros(lu_ids_2018_2022.shape, dtype=onp.int16)
winter_crops_2018_2022_yellow_mustard = onp.zeros(lu_ids_2018_2022.shape, dtype=onp.int16)
for i, year in enumerate([2018, 2019, 2020, 2021, 2022]):
    summer_crops_2018_2022_yellow_mustard[i, :, :] = df_crop_rotations_yellow_mustard.loc[:, f"{year}_summer"].values.reshape(lu_ids_2018_2022.shape[1], lu_ids_2018_2022.shape[2])
    winter_crops_2018_2022_yellow_mustard[i, :, :] = df_crop_rotations_yellow_mustard.loc[:, f"{year}_winter"].values.reshape(lu_ids_2018_2022.shape[1], lu_ids_2018_2022.shape[2])

for i, year in enumerate([2018, 2019, 2020, 2021, 2022]):
    print(f"Year: {year}", summer_crops_2018_2022_yellow_mustard[i, 629, 265], winter_crops_2018_2022_yellow_mustard[i, 629, 265])

# extend crop rotations to years 2013-2023
summer_crops_2013_2023_yellow_mustard = onp.zeros((len(years_2013_2023), lu_ids_2018_2022.shape[1], lu_ids_2018_2022.shape[2]), dtype=onp.int16)
summer_crops_2013_2023_yellow_mustard[:5, :, :] = summer_crops_2018_2022_yellow_mustard 
summer_crops_2013_2023_yellow_mustard[5:10, :, :] = summer_crops_2018_2022_yellow_mustard 
summer_crops_2013_2023_yellow_mustard[-1, :, :] = summer_crops_2018_2022_yellow_mustard[0, :, :]

winter_crops_2013_2023_yellow_mustard  = onp.zeros((len(years_2013_2023), lu_ids_2018_2022.shape[1], lu_ids_2018_2022.shape[2]), dtype=onp.int16)
winter_crops_2013_2023_yellow_mustard[:5, :, :] = winter_crops_2018_2022_yellow_mustard 
winter_crops_2013_2023_yellow_mustard[5:10, :, :] = winter_crops_2018_2022_yellow_mustard 
winter_crops_2013_2023_yellow_mustard[-1, :, :] = winter_crops_2018_2022_yellow_mustard[0, :, :]

for i, year in enumerate(years_2013_2023):
    print(f"Year: {year}({i})", summer_crops_2013_2023_yellow_mustard[i, 629, 265], winter_crops_2013_2023_yellow_mustard[i, 629, 265])

# extend crop rotations to years 2000-2024
summer_crops_2000_2024_yellow_mustard = onp.zeros((len(years_2000_2024), lu_ids_2018_2022.shape[1], lu_ids_2018_2022.shape[2]), dtype=onp.int16)
summer_crops_2000_2024_yellow_mustard[0:3, :, :] = summer_crops_2018_2022_yellow_mustard[2:5, :, :] 
summer_crops_2000_2024_yellow_mustard[3:8, :, :] = summer_crops_2018_2022_yellow_mustard 
summer_crops_2000_2024_yellow_mustard[8:13, :, :] = summer_crops_2018_2022_yellow_mustard 
summer_crops_2000_2024_yellow_mustard[13:18, :, :] = summer_crops_2018_2022_yellow_mustard 
summer_crops_2000_2024_yellow_mustard[18:23, :, :] = summer_crops_2018_2022_yellow_mustard 
summer_crops_2000_2024_yellow_mustard[23:25, :, :] = summer_crops_2018_2022_yellow_mustard[0:2, :, :]

winter_crops_2000_2024_yellow_mustard = onp.zeros((len(years_2000_2024), lu_ids_2018_2022.shape[1], lu_ids_2018_2022.shape[2]), dtype=onp.int16)
winter_crops_2000_2024_yellow_mustard[0:3, :, :] = winter_crops_2018_2022_yellow_mustard[2:5, :, :] 
winter_crops_2000_2024_yellow_mustard[3:8, :, :] = winter_crops_2018_2022_yellow_mustard 
winter_crops_2000_2024_yellow_mustard[8:13, :, :] = winter_crops_2018_2022_yellow_mustard 
winter_crops_2000_2024_yellow_mustard[13:18, :, :] = winter_crops_2018_2022_yellow_mustard 
winter_crops_2000_2024_yellow_mustard[18:23, :, :] = winter_crops_2018_2022_yellow_mustard 
winter_crops_2000_2024_yellow_mustard[23:25, :, :] = winter_crops_2018_2022_yellow_mustard[0:2, :, :]

for i, year in enumerate(years_2000_2024):
    print(f"Year: {year}({i})", summer_crops_2000_2024_yellow_mustard[i, 629, 265], winter_crops_2000_2024_yellow_mustard[i, 629, 265])

# write crop rotations to netcdf files
attrs = dict(
        date_created=datetime.datetime.today().isoformat(),
        title="lu_id of RoGeR in the Dreisam-Moehlin-Neumagen catchment for the years 2013-2023",
        institution="University of Freiburg, Chair of Hydrology",
    )
coords = {
        "lon": ("lon", xcoords),  # x
        "lat": ("lat", ycoords),  # y
        "Year": ("Year", years_2018_2022),
    }
data_vars=dict(
        summer_crops=(["Year", "lat", "lon"], summer_crops_2018_2022_yellow_mustard),
        winter_crops=(["Year", "lat", "lon"], winter_crops_2018_2022_yellow_mustard),
    )

ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
ds["summer_crops"].attrs["units"] = ""
ds["summer_crops"].attrs["long_name"] = "Summer crops encoded as lu_id from RoGeR"
ds["winter_crops"].attrs["units"] = ""
ds["winter_crops"].attrs["long_name"] = "Winter crops encoded as lu_id from RoGeR"
# create spatial reference
ds = ds.geo.write_crs("EPSG:25832")
ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
file = base_path / "input" / "crop_rotations_2013-2023_yellow_mustard.nc"
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
        summer_crops=(["Year", "lat", "lon"], summer_crops_2013_2023_yellow_mustard),
        winter_crops=(["Year", "lat", "lon"], winter_crops_2013_2023_yellow_mustard),
    )

ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
ds["summer_crops"].attrs["units"] = ""
ds["summer_crops"].attrs["long_name"] = "Summer crops encoded as lu_id from RoGeR"
ds["winter_crops"].attrs["units"] = ""
ds["winter_crops"].attrs["long_name"] = "Winter crops encoded as lu_id from RoGeR"
# create spatial reference
ds = ds.geo.write_crs("EPSG:25832")
ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
file = base_path / "input" / "crop_rotations_2013-2023_yellow_mustard.nc"
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
        summer_crops=(["Year", "lat", "lon"], summer_crops_2000_2024_yellow_mustard),
        winter_crops=(["Year", "lat", "lon"], winter_crops_2000_2024_yellow_mustard),
    )

ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
ds["summer_crops"].attrs["units"] = ""
ds["summer_crops"].attrs["long_name"] = "Summer crops encoded as lu_id from RoGeR"
ds["winter_crops"].attrs["units"] = ""
ds["winter_crops"].attrs["long_name"] = "Winter crops encoded as lu_id from RoGeR"
# create spatial reference
ds = ds.geo.write_crs("EPSG:25832")
ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
file = base_path / "input" / "crop_rotations_2000-2024_yellow_mustard.nc"
comp = dict(zlib=True, complevel=1)  # compress data to save storage
encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)

unit_header = [""]
years_header = ["No"]
for year in years_2000_2024:
    unit_header.append("[year_season]")
    unit_header.append("[year_season]")
    years_header.append(f"{year}_summer")
    years_header.append(f"{year}_winter")
df_crop_rotations_2000_2024_yellow_mustard = pd.DataFrame(columns=years_header)
for i, year in enumerate(years_2000_2024):
    df_crop_rotations_2000_2024_yellow_mustard.loc[:, f"{year}_summer"] = summer_crops_2000_2024_yellow_mustard[i, :, :].flatten()
    df_crop_rotations_2000_2024_yellow_mustard.loc[:, f"{year}_winter"] = winter_crops_2000_2024_yellow_mustard[i, :, :].flatten()
df_crop_rotations_2000_2024_yellow_mustard.loc[:, "No"] = onp.arange(1, df_crop_rotations_2000_2024_yellow_mustard.shape[0]+1)
df_crop_rotations_2000_2024_yellow_mustard.columns = [unit_header, years_header]
file = base_path / "input" / "crop_rotations_2000_2024_yellow_mustard.csv"
df_crop_rotations_2000_2024_yellow_mustard.to_csv(file, sep=";", index=False)

unit_header = [""]
years_header = ["No"]
for year in years_2013_2023:
    unit_header.append("[year_season]")
    unit_header.append("[year_season]")
    years_header.append(f"{year}_summer")
    years_header.append(f"{year}_winter")
df_crop_rotations_2013_2023_yellow_mustard = pd.DataFrame(columns=years_header)
for i, year in enumerate(years_2013_2023):
    df_crop_rotations_2013_2023_yellow_mustard.loc[:, f"{year}_summer"] = summer_crops_2013_2023_yellow_mustard[i, :, :].flatten()
    df_crop_rotations_2013_2023_yellow_mustard.loc[:, f"{year}_winter"] = winter_crops_2013_2023_yellow_mustard[i, :, :].flatten()
df_crop_rotations_2013_2023_yellow_mustard.loc[:, "No"] = onp.arange(1, df_crop_rotations_2013_2023_yellow_mustard.shape[0]+1)
df_crop_rotations_2013_2023_yellow_mustard.columns = [unit_header, years_header]
file = base_path / "input" / "crop_rotations_2013_2023_yellow_mustard.csv"
df_crop_rotations_2013_2023_yellow_mustard.to_csv(file, sep=";", index=False)