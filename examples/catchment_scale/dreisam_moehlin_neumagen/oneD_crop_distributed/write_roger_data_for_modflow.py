import os
import h5netcdf
import xarray as xr
import datetime
from pathlib import Path
import numpy as onp
import pandas as pd
import click
import roger

@click.option("-stm", "--stress-test-meteo", type=click.Choice(["base", "base_2000-2024", "spring-drought", "summer-drought", "spring-summer-drought", "spring-summer-wet"]), default="base", help="Type of meteorological stress test")
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

    if grain_corn_only == "no-grain-corn-only":
        _grain_corn_only = ""
    else:
        _grain_corn_only = "_grain-corn-only"

    # load model parameters
    model_parameters_file = base_path / "parameters_roger.nc"
    with xr.open_dataset(model_parameters_file) as infile:
        mask = infile["maskCatch"].values
        xcoords = infile["x"].values
        ycoords = infile["y"].values
        slope = infile["slope"].values/100
        slope = onp.where(slope > 1, 1, slope)

    time_origin = "2012-12-31 23:00:00"
    # merge model output into single file
    diag_file = str(base_path_output / f"ONEDCROP_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}_{_grain_corn_only}.rate.nc")
    print(f"Processing file: {diag_file}")
    if os.path.exists(diag_file):
        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
            time = df.variables.get("Time")[:]
            date_time = pd.date_range(start=time_origin, periods=len(time), freq="D")
            # get unique years
            years = onp.unique(date_time.year).tolist()[:-1]
            nx = len(df.variables["x"])
            ny = len(df.variables["y"])
            for year in years:
                output_file = base_path_output / f"recharge_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}_{_grain_corn_only}_year{year}.nc"
                if not os.path.exists(output_file):
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"RoGeR recharge simulations for the Dreisam-Moehlin-Neumagen catchment - Year {year}",
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
                        v.attrs.update(time_origin=f"{year-1}-12-31 23:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            v = f.create_variable("recharge", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            var_object = df.variables.get("q_ss")
                            var_object1 = df.variables.get("q_sub_ss") * (1 - slope)
                            v[:, :, :] = var_object[time_indices, :, :] + var_object1[time_indices, :, :]
                            v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"])

                output_file = base_path_output / f"capillary_rise_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}_{_grain_corn_only}_year{year}.nc"
                if not os.path.exists(output_file):
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"RoGeR capillary rise simulations for the Dreisam-Moehlin-Neumagen catchment - Year {year}",
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
                        v.attrs.update(time_origin=f"{year-1}-12-31 23:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            v = f.create_variable("capillary_rise", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            var_object = df.variables.get("cpr_ss")
                            v[:, :, :] = var_object[time_indices, :, :]
                            v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"])


                if irrigation == "irrigation":
                    output_file = base_path_output / f"irrigation_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}_{_grain_corn_only}_year{year}.nc"
                    if not os.path.exists(output_file):
                        with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                            f.attrs.update(
                                date_created=datetime.datetime.today().isoformat(),
                                title=f"RoGeR irrigation supply simulations for the Dreisam-Moehlin-Neumagen catchment - Year {year}",
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
                            v.attrs.update(time_origin=f"{year-1}-12-31 23:00:00", units="days")
                            v[:] = range(dict_dim["Time"])
                            with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                                time_indices = onp.where(date_time.year == year)[0]
                                v = f.create_variable("irrigation", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                                var_object = df.variables.get("irrig")
                                v[:, :, :] = var_object[time_indices, :, :]
                                v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"])

    # merge model output into single file
    diag_file = str(base_path_output / f"ONEDCROP_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}_{_grain_corn_only}.collect.nc")
    print(f"Processing file: {diag_file}")
    if os.path.exists(diag_file):
        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
            time = df.variables.get("Time")[:]
            date_time = pd.date_range(start=time_origin, periods=len(time), freq="D")
            # get unique years
            years = onp.unique(date_time.year).tolist()[:-1]
            nx = len(df.variables["x"])
            ny = len(df.variables["y"])
            for year in years:
                output_file = base_path_output / f"land_use_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}_{_grain_corn_only}_year{year}.nc"
                if not os.path.exists(output_file):
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"RoGeR land use IDs for the Dreisam-Moehlin-Neumagen catchment - Year {year}",
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
                        v.attrs.update(time_origin=f"{year-1}-12-31 23:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            v = f.create_variable("land_use", ("Time", "y", "x"), int, compression="gzip", compression_opts=1)
                            var_object = df.variables.get("lu_id")
                            v[:, :, :] = var_object[time_indices, :, :]
                            v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"])


                output_file = base_path_output / f"tamax_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}_{_grain_corn_only}_year{year}.nc"
                if not os.path.exists(output_file):
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"Maximum daily air temperature for the Dreisam-Moehlin-Neumagen catchment - Year {year}",
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
                        v.attrs.update(time_origin=f"{year-1}-12-31 23:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            v = f.create_variable("ta_max", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            var_object = df.variables.get("ta_max")
                            v[:, :, :] = var_object[time_indices, :, :]
                            v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"])


                output_file = base_path_output / f"irrigation_demand_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}_{_grain_corn_only}_year{year}.nc"
                if not os.path.exists(output_file):
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"RoGeR irrigation demand simulations for the Dreisam-Moehlin-Neumagen catchment - Year {year}",
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
                        v.attrs.update(time_origin=f"{year-1}-12-31 23:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            v = f.create_variable("irrigation_demand", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            var_object = df.variables.get("irr_demand")
                            v[:, :, :] = var_object[time_indices, :, :]
                            v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"])

                output_file = base_path_output / f"root_depth_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}_{_grain_corn_only}_year{year}.nc"
                if not os.path.exists(output_file):
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"RoGeR root depth simulations for the Dreisam-Moehlin-Neumagen catchment - Year {year}",
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
                        v.attrs.update(time_origin=f"{year-1}-12-31 23:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            v = f.create_variable("z_root", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            var_object = df.variables.get("z_root")
                            v[:, :, :] = var_object[time_indices, :, :]
                            v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"])

                    output_file = base_path_output / f"ground_cover_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}_{_grain_corn_only}_year{year}.nc"
                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"RoGeR ground cover simulations for the Dreisam-Moehlin-Neumagen catchment - Year {year}",
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
                        v.attrs.update(time_origin=f"{year-1}-12-31 23:00:00", units="days")
                        v[:] = range(dict_dim["Time"])
                        with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            time_indices = onp.where(date_time.year == year)[0]
                            v = f.create_variable("ground_cover", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            var_object = df.variables.get("ground_cover")
                            v[:, :, :] = var_object[time_indices, :, :]
                            v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"])
    return


if __name__ == "__main__":
    main()