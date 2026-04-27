import os
import h5netcdf
import xarray as xr
import datetime
from pathlib import Path
import numpy as onp
import pandas as pd
import shutil
import click
import tarfile
import roger


@click.option("-stm", "--stress-test-meteo", type=click.Choice(["base", "base_2000-2024", "spring-drought", "summer-drought", "spring-summer-drought", "spring-summer-wet", "long-term"]), default="base", help="Type of meteorological stress test")
@click.option("-stmm", "--stress-test-meteo-magnitude", type=click.Choice([0, 1, 2]), default=0, help="Magnitude of meteorological stress test")
@click.option("-stmd", "--stress-test-meteo-duration", type=click.Choice([0, 2, 3]), default=0, help="Duration of meteorological stress test in consecutive years")
@click.option("-irr", "--irrigation", type=click.Choice(["no-irrigation", "irrigation"]), default="no-irrigation", help="Enable irrigation")
@click.option("-ym", "--yellow-mustard", type=click.Choice(["no-yellow-mustard", "yellow-mustard"]), default="no-yellow-mustard", help="Enable catch crop using yellow mustard")
@click.option("-sc", "--soil-compaction", type=click.Choice(["no-soil-compaction", "soil-compaction"]), default="no-soil-compaction", help="Enable soil compaction")
@click.option("-gco", "--grain-corn-only", type=click.Choice(["no-grain-corn-only", "grain-corn-only"]), default="no-grain-corn-only", help="Enable grain corn monoculture (no crop rotation)")
@click.command("main")
def main(stress_test_meteo, stress_test_meteo_magnitude, stress_test_meteo_duration, irrigation, yellow_mustard, soil_compaction, grain_corn_only):
    base_path = Path(__file__).parent
    # directory of results
    # base_path_output = Path("/Volumes/LaCie/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed") / "output"
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)

    base_path_project = Path("/pfs/10/project/bw22g004/fr_rs1092/workspace-1773831854/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed") / "output"

    if grain_corn_only == "no-grain-corn-only":
        _grain_corn_only = ""
    else:
        _grain_corn_only = "_grain-corn-only"

    file = base_path / "input" / "crop_heat_stress.csv"
    df_crop_heat_stress = pd.read_csv(file, sep=";", skiprows=1, index_col=0)
    df_crop_heat_stress.index = df_crop_heat_stress.loc[:, "lu_id"]
    dict_crop_heat_stress = df_crop_heat_stress.loc[:, "ta_heat_stress"].to_frame().to_dict()["ta_heat_stress"]

    # load model parameters
    model_parameters_file = base_path / "parameters_roger.nc"
    with xr.open_dataset(model_parameters_file) as infile:
        mask = (infile["maskCatch"].values == 1)
        xcoords = infile["x"].values
        ycoords = infile["y"].values
        spatial_ref = infile.spatial_ref
        nx = len(xcoords)
        ny = len(ycoords)
        slope = infile["slope"].values / 100
        slope = onp.where(slope > 1, 1, slope)
        theta_ufc = infile["NFK"].values / 100
        theta_pwp = infile["PWP"].values / 100
        theta_ac = infile["LK"].values / 100

    output_file = base_path_output / "theta_fc.nc"
    if not os.path.exists(output_file):
        with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title=f"Field capacity of the Dreisam-Moehlin-Neumagen catchment",
                references="",
            )
            # set dimensions with a dictionary
            dict_dim = {
                "x": nx,
                "y": ny,
                "scalar": 1,
            }
            f.dimensions = dict_dim
            v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
            v.attrs["long_name"] = "x"
            v.attrs["units"] = "m"
            v[:] = xcoords
            v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
            v.attrs["long_name"] = "y"
            v.attrs["units"] = "m"
            v[:] = ycoords
            v = f.create_variable("spatial_ref", ("scalar",), dtype="i4")
            for key in spatial_ref.attrs:
                v.attrs[key] = spatial_ref.attrs[key]
            v = f.create_variable("theta_fc", ("y", "x"), float, compression="gzip", compression_opts=1)
            v[:, :] = theta_pwp + theta_ufc
            v.attrs.update(long_name="Field capacity", units="m3/m3", grid_mapping="spatial_ref")

    if stress_test_meteo == "base_2000-2024":
        time_origin = "2000-01-01 00:00:00"
    else:
        time_origin = "2013-01-01 00:00:00"
    files_to_compress = []
    files_to_compress_rci = []
    # merge model output into single file
    diag_file = str(base_path_project / f"ONEDCROP_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}.collect.nc")
    click.echo(f"Processing file: {diag_file}")
    if os.path.exists(diag_file):
        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
            time = df.variables.get("Time")[:]
            date_time = pd.date_range(start=time_origin, periods=len(time), freq="D")
            # get unique years
            years = onp.unique(date_time.year).tolist()[:-1]
            nx = len(df.variables["x"])
            ny = len(df.variables["y"])
            var_object_lu_id = df.variables.get("lu_id")
            for year in years:
                output_file = base_path_output / f"land_use_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
                files_to_compress.append(output_file)
                if not os.path.exists(output_file):
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"RoGeR land use IDs of the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                            references="",
                            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            model_structure="ONED model with free drainage and explicit crop growth dynamics",
                            roger_version=f"{roger.__version__}",
                        )
                        # set dimensions with a dictionary
                        dict_dim = {
                            "x": nx,
                            "y": ny,
                            "Time": len(onp.where(date_time.year == year)[0]),
                            "scalar": 1,
                        }
                        f.dimensions = dict_dim
                        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "x"
                        v.attrs["units"] = "m"
                        v[:] = xcoords
                        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "y"
                        v.attrs["units"] = "m"
                        v[:] = ycoords
                        v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                        v.attrs.update(time_origin=f"{year}-01-01 00:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        v = f.create_variable("spatial_ref", ("scalar",), dtype="i4")
                        for key in spatial_ref.attrs:
                            v.attrs[key] = spatial_ref.attrs[key]
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            v = f.create_variable("land_use", ("Time", "y", "x"), int, compression="gzip", compression_opts=1)
                            var_object = df.variables.get("lu_id")
                            v[:, :, :] = var_object[time_indices, :, :]
                            v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"], grid_mapping="spatial_ref")

                output_file = base_path_output / f"ta_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
                files_to_compress.append(output_file)
                if not os.path.exists(output_file):
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"Average daily air temperature of the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                            institution="University of Freiburg, Chair of Hydrology",
                            references="",
                            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            model_structure="ONED model with free drainage and explicit crop growth dynamics",
                            roger_version=f"{roger.__version__}",
                        )
                        # set dimensions with a dictionary
                        dict_dim = {
                            "x": nx,
                            "y": ny,
                            "Time": len(onp.where(date_time.year == year)[0]),
                            "scalar": 1,
                        }
                        f.dimensions = dict_dim
                        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "x"
                        v.attrs["units"] = "m"
                        v[:] = xcoords
                        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "y"
                        v.attrs["units"] = "m"
                        v[:] = ycoords
                        v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                        v.attrs.update(time_origin=f"{year}-01-01 00:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        v = f.create_variable("spatial_ref", ("scalar",), dtype="i4")
                        for key in spatial_ref.attrs:
                            v.attrs[key] = spatial_ref.attrs[key]
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            v = f.create_variable("ta", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            var_object = df.variables.get("ta")
                            v[:, :, :] = var_object[time_indices, :, :]
                            v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"], grid_mapping="spatial_ref")

                output_file = base_path_output / f"tamax_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
                files_to_compress.append(output_file)
                if not os.path.exists(output_file):
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"Maximum daily air temperature of the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                            institution="University of Freiburg, Chair of Hydrology",
                            references="",
                            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            model_structure="ONED model with free drainage and explicit crop growth dynamics",
                            roger_version=f"{roger.__version__}",
                        )
                        # set dimensions with a dictionary
                        dict_dim = {
                            "x": nx,
                            "y": ny,
                            "Time": len(onp.where(date_time.year == year)[0]),
                            "scalar": 1,
                        }
                        f.dimensions = dict_dim
                        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "x"
                        v.attrs["units"] = "m"
                        v[:] = xcoords
                        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "y"
                        v.attrs["units"] = "m"
                        v[:] = ycoords
                        v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                        v.attrs.update(time_origin=f"{year}-01-01 00:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        v = f.create_variable("spatial_ref", ("scalar",), dtype="i4")
                        for key in spatial_ref.attrs:
                            v.attrs[key] = spatial_ref.attrs[key]
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            v = f.create_variable("ta_max", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            var_object = df.variables.get("ta_max")
                            v[:, :, :] = var_object[time_indices, :, :]
                            v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"], grid_mapping="spatial_ref")


                output_file = base_path_output / f"irrigation_demand_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
                files_to_compress.append(output_file)
                if not os.path.exists(output_file):
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"RoGeR irrigation demand simulations of the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                            institution="University of Freiburg, Chair of Hydrology",
                            references="",
                            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            model_structure="ONED model with free drainage and explicit crop growth dynamics",
                            roger_version=f"{roger.__version__}",
                        )
                        # set dimensions with a dictionary
                        dict_dim = {
                            "x": nx,
                            "y": ny,
                            "Time": len(onp.where(date_time.year == year)[0]),
                            "scalar": 1,
                        }
                        f.dimensions = dict_dim
                        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "x"
                        v.attrs["units"] = "m"
                        v[:] = xcoords
                        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "y"
                        v.attrs["units"] = "m"
                        v[:] = ycoords
                        v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                        v.attrs.update(time_origin=f"{year}-01-01 00:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        v = f.create_variable("spatial_ref", ("scalar",), dtype="i4")
                        for key in spatial_ref.attrs:
                            v.attrs[key] = spatial_ref.attrs[key]
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            v = f.create_variable("irrigation_demand", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            var_object = df.variables.get("irr_demand")
                            v[:, :, :] = var_object[time_indices, :, :]
                            v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"], grid_mapping="spatial_ref")

                output_file = base_path_output / f"root_depth_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
                files_to_compress.append(output_file)
                if not os.path.exists(output_file):
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"RoGeR root depth simulations of the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                            institution="University of Freiburg, Chair of Hydrology",
                            references="",
                            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            model_structure="ONED model with free drainage and explicit crop growth dynamics",
                            roger_version=f"{roger.__version__}",
                        )
                        # set dimensions with a dictionary
                        dict_dim = {
                            "x": nx,
                            "y": ny,
                            "Time": len(onp.where(date_time.year == year)[0]),
                            "scalar": 1,
                        }
                        f.dimensions = dict_dim
                        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "x"
                        v.attrs["units"] = "m"
                        v[:] = xcoords
                        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "y"
                        v.attrs["units"] = "m"
                        v[:] = ycoords
                        v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                        v.attrs.update(time_origin=f"{year}-01-01 00:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        v = f.create_variable("spatial_ref", ("scalar",), dtype="i4")
                        for key in spatial_ref.attrs:
                            v.attrs[key] = spatial_ref.attrs[key]
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            v = f.create_variable("z_root", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            var_object = df.variables.get("z_root")
                            v[:, :, :] = var_object[time_indices, :, :]
                            v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"], grid_mapping="spatial_ref")

                    output_file = base_path_output / f"canopy_cover_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
                    files_to_compress.append(output_file)
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"RoGeR canopy cover simulations of the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                            institution="University of Freiburg, Chair of Hydrology",
                            references="",
                            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            model_structure="ONED model with free drainage and explicit crop growth dynamics",
                            roger_version=f"{roger.__version__}",
                        )
                        # set dimensions with a dictionary
                        dict_dim = {
                            "x": nx,
                            "y": ny,
                            "Time": len(onp.where(date_time.year == year)[0]),
                            "scalar": 1,    
                        }
                        f.dimensions = dict_dim
                        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "x"
                        v.attrs["units"] = "m"
                        v[:] = xcoords
                        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "y"
                        v.attrs["units"] = "m"
                        v[:] = ycoords
                        v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                        v.attrs.update(time_origin=f"{year}-01-01 00:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        v = f.create_variable("spatial_ref", ("scalar",), dtype="i4")
                        for key in spatial_ref.attrs:
                            v.attrs[key] = spatial_ref.attrs[key]
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            v = f.create_variable("canopy_cover", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            var_object = df.variables.get("ground_cover")
                            v[:, :, :] = var_object[time_indices, :, :]
                            v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"], grid_mapping="spatial_ref")

                    output_file = base_path_output / f"heat_stress_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
                    files_to_compress.append(output_file)
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"Heat stress of the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                            institution="University of Freiburg, Chair of Hydrology",
                            references="",
                            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            model_structure="ONED model with free drainage and explicit crop growth dynamics",
                            roger_version=f"{roger.__version__}",
                        )
                        # set dimensions with a dictionary
                        dict_dim = {
                            "x": nx,
                            "y": ny,
                            "Time": len(onp.where(date_time.year == year)[0]),
                            "scalar": 1,
                        }
                        f.dimensions = dict_dim
                        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "x"
                        v.attrs["units"] = "m"
                        v[:] = xcoords
                        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "y"
                        v.attrs["units"] = "m"
                        v[:] = ycoords
                        v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                        v.attrs.update(time_origin=f"{year}-01-01 00:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        v = f.create_variable("spatial_ref", ("scalar",), dtype="i4")
                        for key in spatial_ref.attrs:
                            v.attrs[key] = spatial_ref.attrs[key]
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            lu_ids_year = var_object_lu_id[time_indices, :, :]
                            ta_max_year = df.variables.get("ta_max")[time_indices, :, :]
                            _lu_ids_year = onp.unique(lu_ids_year)
                            heat_stress = onp.zeros(lu_ids_year.shape)
                            for i, lu_id in enumerate(_lu_ids_year):
                                if lu_id > 500 and lu_id < 598:
                                    ta_heat_stress = dict_crop_heat_stress[lu_id]
                                    mask_heat_stress = (lu_ids_year == lu_id) & (ta_max_year > ta_heat_stress)
                                    heat_stress = onp.where(mask_heat_stress, 1, heat_stress)
                            heat_stress = onp.where(mask[onp.newaxis, :, :], heat_stress, onp.nan)
                            v = f.create_variable("heat_stress", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            v[:, :, :] = heat_stress
                            v.attrs.update(long_name="heat stress", units="", grid_mapping="spatial_ref")


                    output_file = base_path_output / f"heat_days_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
                    files_to_compress.append(output_file)
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"Heat days of the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                            institution="University of Freiburg, Chair of Hydrology",
                            references="",
                            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            model_structure="ONED model with free drainage and explicit crop growth dynamics",
                            roger_version=f"{roger.__version__}",
                        )
                        # set dimensions with a dictionary
                        dict_dim = {
                            "x": nx,
                            "y": ny,
                            "Time": len(onp.where(date_time.year == year)[0]),
                            "scalar": 1,
                        }
                        f.dimensions = dict_dim
                        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "x"
                        v.attrs["units"] = "m"
                        v[:] = xcoords
                        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "y"
                        v.attrs["units"] = "m"
                        v[:] = ycoords
                        v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                        v.attrs.update(time_origin=f"{year}-01-01 00:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        v = f.create_variable("spatial_ref", ("scalar",), dtype="i4")
                        for key in spatial_ref.attrs:
                            v.attrs[key] = spatial_ref.attrs[key]
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            ta_max_year = df.variables.get("ta_max")[time_indices, :, :]
                            heat_days = onp.zeros(ta_max_year.shape)
                            mask_heat_days = (ta_max_year > 30)
                            heat_days = onp.where(mask_heat_days, 1, heat_days)
                            heat_days = onp.where(mask[onp.newaxis, :, :], heat_days, onp.nan)
                            v = f.create_variable("heat_day", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            v[:, :, :] = heat_days
                            v.attrs.update(long_name="heat day", units="", grid_mapping="spatial_ref")

                    output_file = base_path_output / f"drought_days_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
                    files_to_compress.append(output_file)
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"Drought days of the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                            institution="University of Freiburg, Chair of Hydrology",
                            references="",
                            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            model_structure="ONED model with free drainage and explicit crop growth dynamics",
                            roger_version=f"{roger.__version__}",
                        )
                        # set dimensions with a dictionary
                        dict_dim = {
                            "x": nx,
                            "y": ny,
                            "Time": len(onp.where(date_time.year == year)[0]),
                            "scalar": 1,
                        }
                        f.dimensions = dict_dim
                        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "x"
                        v.attrs["units"] = "m"
                        v[:] = xcoords
                        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "y"
                        v.attrs["units"] = "m"
                        v[:] = ycoords
                        v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                        v.attrs.update(time_origin=f"{year}-01-01 00:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        v = f.create_variable("spatial_ref", ("scalar",), dtype="i4")
                        for key in spatial_ref.attrs:
                            v.attrs[key] = spatial_ref.attrs[key]
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            theta_rz_year = df.variables.get("theta_rz")[time_indices, :, :]
                            drought_days = onp.zeros(theta_rz_year.shape)
                            mask_drought_days = (theta_rz_year < theta_pwp[onp.newaxis, :, :] + (0.35 * theta_ufc[onp.newaxis, :, :]))
                            drought_days = onp.where(mask_drought_days, 1, drought_days)
                            drought_days = onp.where(mask[onp.newaxis, :, :], drought_days, onp.nan)
                            v = f.create_variable("drought_day", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            v[:, :, :] = drought_days
                            v.attrs.update(long_name="drought day", units="", grid_mapping="spatial_ref")

                    output_file = base_path_output / f"root_ventilation_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
                    files_to_compress.append(output_file)
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"Root ventilation of the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                            institution="University of Freiburg, Chair of Hydrology",
                            references="",
                            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            model_structure="ONED model with free drainage and explicit crop growth dynamics",
                            roger_version=f"{roger.__version__}",
                        )
                        # set dimensions with a dictionary
                        dict_dim = {
                            "x": nx,
                            "y": ny,
                            "Time": len(onp.where(date_time.year == year)[0]),
                            "scalar": 1,
                        }
                        f.dimensions = dict_dim
                        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "x"
                        v.attrs["units"] = "m"
                        v[:] = xcoords
                        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "y"
                        v.attrs["units"] = "m"
                        v[:] = ycoords
                        v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                        v.attrs.update(time_origin=f"{year}-01-01 00:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        v = f.create_variable("spatial_ref", ("scalar",), dtype="i4")
                        for key in spatial_ref.attrs:
                            v.attrs[key] = spatial_ref.attrs[key]
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            theta_rz_year = df.variables.get("theta_rz")[time_indices, :, :]
                            root_ventilation = (theta_rz_year - (theta_ufc[onp.newaxis, :, :] + theta_pwp[onp.newaxis, :, :])) / theta_ac[onp.newaxis, :, :]
                            root_ventilation = onp.where(root_ventilation < 0, 0, root_ventilation)
                            root_ventilation = onp.where(root_ventilation > 1, 1, root_ventilation)
                            root_ventilation = onp.where(mask[onp.newaxis, :, :], (1 - root_ventilation) * 100, onp.nan)
                            v = f.create_variable("root_ventilation", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            v[:, :, :] = root_ventilation
                            v.attrs.update(long_name="root ventilation", units="%", grid_mapping="spatial_ref")

                    output_file = base_path_output / f"theta_rz_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
                    files_to_compress.append(output_file)
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"Soil water content root zone of the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                            institution="University of Freiburg, Chair of Hydrology",
                            references="",
                            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            model_structure="ONED model with free drainage and explicit crop growth dynamics",
                            roger_version=f"{roger.__version__}",
                        )
                        # set dimensions with a dictionary
                        dict_dim = {
                            "x": nx,
                            "y": ny,
                            "Time": len(onp.where(date_time.year == year)[0]),
                            "scalar": 1,
                        }
                        f.dimensions = dict_dim
                        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "x"
                        v.attrs["units"] = "m"
                        v[:] = xcoords
                        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "y"
                        v.attrs["units"] = "m"
                        v[:] = ycoords
                        v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                        v.attrs.update(time_origin=f"{year}-01-01 00:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        v = f.create_variable("spatial_ref", ("scalar",), dtype="i4")
                        for key in spatial_ref.attrs:
                            v.attrs[key] = spatial_ref.attrs[key]
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            theta_rz_year = df.variables.get("theta_rz")[time_indices, :, :]
                            v = f.create_variable("theta_rz", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            v[:, :, :] = theta_rz_year
                            v.attrs.update(long_name="soil water content of the root zone", units="-", grid_mapping="spatial_ref")


    # merge model output into single file
    diag_file = str(base_path_project / f"ONEDCROP_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}.rate.nc")
    click.echo(f"Processing file: {diag_file}")
    if os.path.exists(diag_file):
        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
            time = df.variables.get("Time")[:]
            date_time = pd.date_range(start=time_origin, periods=len(time), freq="D")
            # get unique years
            years = onp.unique(date_time.year).tolist()[:-1]
            nx = len(df.variables["x"])
            ny = len(df.variables["y"])
            for year in years:
                output_file = base_path_output / f"recharge_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
                files_to_compress.append(output_file)
                files_to_compress_rci.append(output_file)
                if not os.path.exists(output_file):
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"RoGeR recharge simulations of the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                            institution="University of Freiburg, Chair of Hydrology",
                            references="",
                            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            model_structure="ONED model with free drainage and explicit crop growth dynamics",
                            roger_version=f"{roger.__version__}",
                        )
                        # set dimensions with a dictionary
                        dict_dim = {
                            "x": nx,
                            "y": ny,
                            "Time": len(onp.where(date_time.year == year)[0]),
                            "scalar": 1,
                        }
                        f.dimensions = dict_dim
                        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "x"
                        v.attrs["units"] = "m"
                        v[:] = xcoords
                        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "y"
                        v.attrs["units"] = "m"
                        v[:] = ycoords
                        v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                        v.attrs.update(time_origin=f"{year}-01-01 00:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        v = f.create_variable("spatial_ref", ("scalar",), dtype="i4")
                        for key in spatial_ref.attrs:
                            v.attrs[key] = spatial_ref.attrs[key]
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            v = f.create_variable("recharge", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            var_object = df.variables.get("q_ss")
                            v[:, :, :] = var_object[time_indices, :, :]
                            v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"], grid_mapping="spatial_ref")

                output_file = base_path_output / f"capillary_rise_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
                files_to_compress.append(output_file)
                files_to_compress_rci.append(output_file)
                if not os.path.exists(output_file):
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"RoGeR capillary rise simulations of the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                            institution="University of Freiburg, Chair of Hydrology",
                            references="",
                            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            model_structure="ONED model with free drainage and explicit crop growth dynamics",
                            roger_version=f"{roger.__version__}",
                        )
                        # set dimensions with a dictionary
                        dict_dim = {
                            "x": nx,
                            "y": ny,
                            "Time": len(onp.where(date_time.year == year)[0]),
                            "scalar": 1,
                        }
                        f.dimensions = dict_dim
                        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "x"
                        v.attrs["units"] = "m"
                        v[:] = xcoords
                        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "y"
                        v.attrs["units"] = "m"
                        v[:] = ycoords
                        v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                        v.attrs.update(time_origin=f"{year}-01-01 00:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        v = f.create_variable("spatial_ref", ("scalar",), dtype="i4")
                        for key in spatial_ref.attrs:
                            v.attrs[key] = spatial_ref.attrs[key]
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            v = f.create_variable("capillary_rise", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            var_object = df.variables.get("cpr_ss")
                            v[:, :, :] = var_object[time_indices, :, :]
                            v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"], grid_mapping="spatial_ref")

                output_file = base_path_output / f"transpiration_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
                files_to_compress.append(output_file)
                if not os.path.exists(output_file):
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"RoGeR transpiration simulations of the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                            institution="University of Freiburg, Chair of Hydrology",
                            references="",
                            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            model_structure="ONED model with free drainage and explicit crop growth dynamics",
                            roger_version=f"{roger.__version__}",
                        )
                        # set dimensions with a dictionary
                        dict_dim = {
                            "x": nx,
                            "y": ny,
                            "Time": len(onp.where(date_time.year == year)[0]),
                            "scalar": 1,
                        }
                        f.dimensions = dict_dim
                        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "x"
                        v.attrs["units"] = "m"
                        v[:] = xcoords
                        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "y"
                        v.attrs["units"] = "m"
                        v[:] = ycoords
                        v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                        v.attrs.update(time_origin=f"{year}-01-01 00:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        v = f.create_variable("spatial_ref", ("scalar",), dtype="i4")
                        for key in spatial_ref.attrs:
                            v.attrs[key] = spatial_ref.attrs[key]
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            v = f.create_variable("transpiration", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            var_object = df.variables.get("transp")
                            v[:, :, :] = var_object[time_indices, :, :]
                            v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"], grid_mapping="spatial_ref")

                output_file = base_path_output / f"potential_transpiration_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
                files_to_compress.append(output_file)
                if not os.path.exists(output_file):
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"RoGeR transpiration simulations of the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                            institution="University of Freiburg, Chair of Hydrology",
                            references="",
                            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            model_structure="ONED model with free drainage and explicit crop growth dynamics",
                            roger_version=f"{roger.__version__}",
                        )
                        # set dimensions with a dictionary
                        dict_dim = {
                            "x": nx,
                            "y": ny,
                            "Time": len(onp.where(date_time.year == year)[0]),
                            "scalar": 1,
                        }
                        f.dimensions = dict_dim
                        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "x"
                        v.attrs["units"] = "m"
                        v[:] = xcoords
                        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "y"
                        v.attrs["units"] = "m"
                        v[:] = ycoords
                        v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                        v.attrs.update(time_origin=f"{year}-01-01 00:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        v = f.create_variable("spatial_ref", ("scalar",), dtype="i4")
                        for key in spatial_ref.attrs:
                            v.attrs[key] = spatial_ref.attrs[key]
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            v = f.create_variable("potential_transpiration", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            var_object = df.variables.get("pt")
                            v[:, :, :] = var_object[time_indices, :, :]
                            v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"], grid_mapping="spatial_ref")

                output_file = base_path_output / f"photosynthesis_index_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
                files_to_compress.append(output_file)
                if not os.path.exists(output_file):
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"RoGeR photosynthesis index of the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                            institution="University of Freiburg, Chair of Hydrology",
                            references="",
                            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            model_structure="ONED model with free drainage and explicit crop growth dynamics",
                            roger_version=f"{roger.__version__}",
                        )
                        # set dimensions with a dictionary
                        dict_dim = {
                            "x": nx,
                            "y": ny,
                            "Time": len(onp.where(date_time.year == year)[0]),
                            "scalar": 1,
                        }
                        f.dimensions = dict_dim
                        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "x"
                        v.attrs["units"] = "m"
                        v[:] = xcoords
                        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                        v.attrs["long_name"] = "y"
                        v.attrs["units"] = "m"
                        v[:] = ycoords
                        v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                        v.attrs.update(time_origin=f"{year}-01-01 00:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        v = f.create_variable("spatial_ref", ("scalar",), dtype="i4")
                        for key in spatial_ref.attrs:
                            v.attrs[key] = spatial_ref.attrs[key]
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            v = f.create_variable("photosynthesis_index", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            var_object1 = df.variables.get("transp")
                            var_object2 = df.variables.get("pt")
                            photosynthesis_index = var_object1[time_indices, :, :] / var_object2[time_indices, :, :]
                            photosynthesis_index = onp.where(photosynthesis_index > 1, 1, photosynthesis_index)
                            photosynthesis_index = onp.where(photosynthesis_index < 0, 0, photosynthesis_index)
                            v[:, :, :] = photosynthesis_index
                            v.attrs.update(long_name="Photosynthesis Index", units="-", grid_mapping="spatial_ref")

                if irrigation == "irrigation":
                    output_file = base_path_output / f"irrigation_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
                    files_to_compress.append(output_file)
                    files_to_compress_rci.append(output_file)
                    if not os.path.exists(output_file):
                        with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                            f.attrs.update(
                                date_created=datetime.datetime.today().isoformat(),
                                title=f"RoGeR irrigation supply simulations of the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                                institution="University of Freiburg, Chair of Hydrology",
                                references="",
                                comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                                model_structure="ONED model with free drainage and explicit crop growth dynamics",
                                roger_version=f"{roger.__version__}",
                            )
                            # set dimensions with a dictionary
                            dict_dim = {
                                "x": nx,
                                "y": ny,
                                "Time": len(onp.where(date_time.year == year)[0]),
                                "scalar": 1,
                            }
                            f.dimensions = dict_dim
                            v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                            v.attrs["long_name"] = "x"
                            v.attrs["units"] = "m"
                            v[:] = xcoords
                            v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                            v.attrs["long_name"] = "y"
                            v.attrs["units"] = "m"
                            v[:] = ycoords
                            v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                            v.attrs.update(time_origin=f"{year}-01-01 00:00:00", units="days")
                            v[:] = range(dict_dim["Time"])
                            v = f.create_variable("spatial_ref", ("scalar",), dtype="i4")
                            for key in spatial_ref.attrs:
                                v.attrs[key] = spatial_ref.attrs[key]
                            with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                                time_indices = onp.where(date_time.year == year)[0]
                                v = f.create_variable("irrigation", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                                var_object = df.variables.get("irrig")
                                v[:, :, :] = var_object[time_indices, :, :]
                                v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"], grid_mapping="spatial_ref")
    
    # compress files
    if files_to_compress:
        archive_name = f"ONEDCROP_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}.tar.gz"
        archive_path = base_path_output / archive_name
        with tarfile.open(archive_path, "w:gz") as tar:
            for file in files_to_compress:
                tar.add(file, arcname=file.name)
        shutil.copy(archive_path, base_path_project / archive_name)
        # os.remove(archive_path)

    if files_to_compress_rci:
        archive_name = f"ONEDCROP_rci_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}.tar.gz"
        archive_path = base_path_output / archive_name
        with tarfile.open(archive_path, "w:gz") as tar:
            for file in files_to_compress_rci:
                tar.add(file, arcname=file.name)
        shutil.copy(archive_path, base_path_project / archive_name)
        # os.remove(archive_path)

    # # remove uncompressed files
    # for file in files_to_compress:
    #     shutil.copy(file, base_path_project / file.name)
    #     os.remove(file)

    # # copy all .nc files to project folder and remove them from output folder
    # for file in base_path_output.glob("*.nc"):
    #     shutil.copy(file, base_path_project / file.name)
    #     os.remove(file)

    return


if __name__ == "__main__":
    main()