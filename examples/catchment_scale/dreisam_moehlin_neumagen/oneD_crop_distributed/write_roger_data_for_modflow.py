import os
import h5netcdf
import xarray as xr
import datetime
from pathlib import Path
import numpy as onp
import pandas as pd
import click
import roger


@click.command("main")
def main():
    base_path = Path(__file__).parent
    # directory of results
    base_path_output = Path("/Volumes/LaCie/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed") / "output"
    # base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)

    # load model parameters
    model_parameters_file = base_path / "parameters_roger.nc"
    with xr.open_dataset(model_parameters_file) as infile:
        mask = infile["maskCatch"].values
        xcoords = infile["x"].values
        ycoords = infile["y"].values
        slope = infile["slope"].values/100
        slope = onp.where(slope > 1, 1, slope)

    irrigation_scenarios = ["no-irrigation"]
    catch_crop_scenarios = ["no-yellow-mustard"]
    soil_compaction_scenarios = ["no-soil-compaction", "soil-compaction"]
    stress_tests_meteo = ["base", "spring-drought", "summer-drought", "spring-summer-drought", "spring-summer-wet"]
    stress_test_meteo_magnitudes = [0, 2]
    stress_test_meteo_durations = [0, 3]

    meteo_stress_tests = []
    for meteo in stress_tests_meteo:
        if meteo == "base" or meteo == "base_2000-2024":
            meteo_stress_tests.append(f"{meteo}-magnitude0-duration0")
            if meteo == "base":
                time_origin = "1/1/2013"
            elif meteo == "base_2000-2024":
                time_origin = "1/1/2000"
        elif meteo == "spring-summer-wet":
            meteo_stress_tests.append(f"{meteo}-magnitude0-duration0")
        else:
            for magnitude in stress_test_meteo_magnitudes:
                for duration in stress_test_meteo_durations:
                    if magnitude == 0 and duration == 0:
                        continue
                    else:
                        meteo_stress_tests.append(f"{meteo}-magnitude{magnitude}-duration{duration}")

    for meteo_stress_test in meteo_stress_tests:
        for irrigation_scenario in irrigation_scenarios:
            for catch_crop_scenario in catch_crop_scenarios:
                for soil_compaction_scenario in soil_compaction_scenarios:
                    # merge model output into single file
                    diag_file = str(base_path_output / f"ONEDCROP_{meteo_stress_test}_{irrigation_scenario}_{catch_crop_scenario}_{soil_compaction_scenario}.rate.nc")
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
                                output_file = base_path_output / f"recharge_{meteo_stress_test}_{irrigation_scenario}_{catch_crop_scenario}_{soil_compaction_scenario}_year{year}.nc"
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
                                            var_object = df.variables.get("q_ss")* (1 - slope)
                                            var_object1 = df.variables.get("q_sub_ss") * (1 - slope)
                                            v[:, :, :] = var_object[time_indices, :, :] + var_object1[time_indices, :, :]
                                            v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"])

                                # output_file = base_path_output / f"capillary_rise_{meteo_stress_test}_{irrigation_scenario}_{catch_crop_scenario}_{soil_compaction_scenario}_year{year}.nc"
                                # if not os.path.exists(output_file):
                                #     with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                                #         f.attrs.update(
                                #             date_created=datetime.datetime.today().isoformat(),
                                #             title=f"RoGeR capillary rise simulations for the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                                #             institution="University of Freiburg, Chair of Hydrology",
                                #             references="",
                                #             comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                                #             model_structure="ONED model with free drainage and explicit crop growth dynamics",
                                #             roger_version=f"{roger.__version__}",
                                #         )
                                #         # set dimensions with a dictionary
                                #         dict_dim = {
                                #             "x": nx,
                                #             "y": ny,
                                #             "Time": len(onp.where(date_time.year == year)[0]),
                                #         }
                                #         f.dimensions = dict_dim
                                #         v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                                #         v.attrs["long_name"] = "x"
                                #         v.attrs["units"] = "m"
                                #         v[:] = xcoords
                                #         v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                                #         v.attrs["long_name"] = "y"
                                #         v.attrs["units"] = "m"
                                #         v[:] = ycoords
                                #         v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                                #         v.attrs.update(time_origin=f"{year-1}-12-31 23:00:00", units="days")
                                #         v[:] = range(dict_dim["Time"])
                                #         with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                                #             time_indices = onp.where(date_time.year == year)[0]
                                #             v = f.create_variable("capillary_rise", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                                #             var_object = df.variables.get("cpr_ss")
                                #             v[:, :, :] = var_object[time_indices, :, :]
                                #             v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"])


                                # if irrigation_scenario == "irrigation":
                                #     output_file = base_path_output / f"irrigation_{meteo_stress_test}_{irrigation_scenario}_{catch_crop_scenario}_{soil_compaction_scenario}_year{year}.nc"
                                #     if not os.path.exists(output_file):
                                #         with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                                #             f.attrs.update(
                                #                 date_created=datetime.datetime.today().isoformat(),
                                #                 title=f"RoGeR irrigation supply simulations for the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                                #                 institution="University of Freiburg, Chair of Hydrology",
                                #                 references="",
                                #                 comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                                #                 model_structure="ONED model with free drainage and explicit crop growth dynamics",
                                #                 roger_version=f"{roger.__version__}",
                                #             )
                                #             # set dimensions with a dictionary
                                #             dict_dim = {
                                #                 "x": nx,
                                #                 "y": ny,
                                #                 "Time": len(onp.where(date_time.year == year)[0]),
                                #             }
                                #             f.dimensions = dict_dim
                                #             v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                                #             v.attrs["long_name"] = "x"
                                #             v.attrs["units"] = "m"
                                #             v[:] = xcoords
                                #             v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                                #             v.attrs["long_name"] = "y"
                                #             v.attrs["units"] = "m"
                                #             v[:] = ycoords
                                #             v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                                #             v.attrs.update(time_origin=f"{year-1}-12-31 23:00:00", units="days")
                                #             v[:] = range(dict_dim["Time"])
                                #             with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                                #                 time_indices = onp.where(date_time.year == year)[0]
                                #                 v = f.create_variable("irrigation", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                                #                 var_object = df.variables.get("irrig")
                                #                 v[:, :, :] = var_object[time_indices, :, :]
                                #                 v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"])

                    # merge model output into single file
                    diag_file = str(base_path_output / f"ONEDCROP_{meteo_stress_test}_{irrigation_scenario}_{catch_crop_scenario}_{soil_compaction_scenario}.collect.nc")
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
                                output_file = base_path_output / f"land_use_{meteo_stress_test}_{irrigation_scenario}_{catch_crop_scenario}_{soil_compaction_scenario}_year{year}.nc"
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


                            #     output_file = base_path_output / f"tamax_{meteo_stress_test}_{irrigation_scenario}_{catch_crop_scenario}_{soil_compaction_scenario}_year{year}.nc"
                            #     if not os.path.exists(output_file):
                            #         with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                            #             f.attrs.update(
                            #                 date_created=datetime.datetime.today().isoformat(),
                            #                 title=f"Maximum daily air temperature for the Dreisam-Moehlin-Neumagen catchment - Year {year}",
                            #                 institution="University of Freiburg, Chair of Hydrology",
                            #                 references="",
                            #                 comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            #                 model_structure="ONED model with free drainage and explicit crop growth dynamics",
                            #                 roger_version=f"{roger.__version__}",
                            #             )
                            #             # set dimensions with a dictionary
                            #             dict_dim = {
                            #                 "x": nx,
                            #                 "y": ny,
                            #                 "Time": len(onp.where(date_time.year == year)[0]),
                            #             }
                            #             f.dimensions = dict_dim
                            #             v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                            #             v.attrs["long_name"] = "x"
                            #             v.attrs["units"] = "m"
                            #             v[:] = xcoords
                            #             v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                            #             v.attrs["long_name"] = "y"
                            #             v.attrs["units"] = "m"
                            #             v[:] = ycoords
                            #             v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                            #             v.attrs.update(time_origin=f"{year-1}-12-31 23:00:00", units="days")
                            #             v[:] = range(dict_dim["Time"])
                            #             with h5netcdf.File(diag_file, "r", decode_vlen_strings=False) as df:
                            #                 time_indices = onp.where(date_time.year == year)[0]
                            #                 v = f.create_variable("ta_max", ("Time", "y", "x"), float, compression="gzip", compression_opts=1)
                            #                 var_object = df.variables.get("ta_max")
                            #                 v[:, :, :] = var_object[time_indices, :, :]
                            #                 v.attrs.update(long_name=var_object.attrs["long_name"], units=var_object.attrs["units"])


                                output_file = base_path_output / f"irrigation_demand_{meteo_stress_test}_{irrigation_scenario}_{catch_crop_scenario}_{soil_compaction_scenario}_year{year}.nc"
                                if not os.path.exists(output_file):
                                    with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                                        f.attrs.update(
                                            date_created=datetime.datetime.today().isoformat(),
                                            title=f"RoGeR irrigation demandsimulations for the Dreisam-Moehlin-Neumagen catchment - Year {year}",
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
    return


if __name__ == "__main__":
    main()