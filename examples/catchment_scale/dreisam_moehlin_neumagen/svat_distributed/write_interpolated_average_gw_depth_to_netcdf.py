import os
from pathlib import Path
import xarray as xr
import geoxarray
import numpy as onp
import datetime
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent

    # load spatial reference and coordinates
    with xr.open_dataset(base_path / "parameters.nc") as ds:
        lu_id = ds["lanu"].values
        gw_depth = ds["gwfa_gew"].values
        gw_depth = onp.where(lu_id < 0, onp.nan, gw_depth/100)
        gw_depth = onp.where(gw_depth < 0, 0, gw_depth)
        spatial_ref = ds.spatial_ref
        xcoords = ds.x.values
        ycoords = ds.y.values

    # create xarray dataset
    attrs = dict(
            date_created=datetime.datetime.today().isoformat(),
            title="Interpolated average groundwater depth of the Dreisam-Moehlin-Neumagen catchment",
            institution="University of Freiburg, Chair of Hydrology",
        )
    coords = {
            "lon": ("lon", xcoords),  # x
            "lat": ("lat", ycoords),  # y
        }
    data_vars=dict(
            gw_depth=(["lat", "lon"], gw_depth),
        )

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    ds["gw_depth"].attrs["units"] = "m"
    ds["gw_depth"].attrs["long_name"] = "Interpolated average groundwater depth"
    # create spatial reference
    ds = ds.geo.write_crs("EPSG:25832")
    ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
    file = base_path / "interpolated_average_groundwater_depth.nc"
    comp = dict(zlib=True, complevel=1)  # compress data to save storage
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)
    return


if __name__ == "__main__":
    main()