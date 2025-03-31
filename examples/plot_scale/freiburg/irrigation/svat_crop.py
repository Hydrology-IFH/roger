from pathlib import Path
import os
import shutil
import h5netcdf
import numpy as onp
import pandas as pd
import click
from roger.cli.roger_run_base import roger_base_cli

@click.option("--irrigation-scenario", type=click.Choice(["20-ufc",
                                                          "35-ufc",
                                                          "50-ufc",
                                                          "crop-specific",
                                                          ]), default="crop-specific")
@click.option("--crop-rotation-scenario", type=click.Choice(["grain-corn",
                                                             "grain-corn_yellow-mustard",
                                                             "silage-corn",
                                                             "silage-corn_yellow-mustard",
                                                             "summer-barley",
                                                             "summer-barley_yellow-mustard",
                                                             "clover",
                                                             "winter-wheat",
                                                             "winter-barley",
                                                             "winter-rape",
                                                             "faba-bean",
                                                             "potato-early",
                                                             "sugar-beet",
                                                             "sugar-beet_yellow-mustard",
                                                             "vegetables",
                                                             "strawberry",
                                                             "asparagus",
                                                             "winter-wheat_clover",
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
                                                             "grain-corn",
                                                             "grain-corn_yellow-mustard",
                                                             "winter-wheat",
                                                             "yellow-mustard",
                                                             "miscanthus",
                                                             "bare-grass"]), default="grain-corn")
@click.option("-td", "--tmp-dir", type=str, default=Path(__file__).parent.parent / "output" / "irrigation")
@roger_base_cli
def main(irrigation_scenario, crop_rotation_scenario, tmp_dir):
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at
    from roger.core.surface import calc_parameters_surface_kernel
    from roger.tools.setup import write_forcing, write_crop_rotation
    import roger.lookuptables as lut

    tmp_dir = Path(tmp_dir) / irrigation_scenario

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    class SVATCROPSetup(RogerSetup):
        """A SVAT model including crop phenology/crop rotation."""

        _base_path = Path(__file__).parent
        _input_dir = _base_path.parent / "input"

        def _read_var_from_nc(self, var, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables[var]
                return npx.array(var_obj)

        def _read_var_from_csv(self, var, path_dir, file):
            csv_file = path_dir / file
            infile = pd.read_csv(csv_file, sep=";", skiprows=1)
            var_obj = infile.loc[:, var]
            return npx.array(var_obj)[:, npx.newaxis]

        def _get_nitt(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables["Time"]
                return len(onp.array(var_obj))
            
        def _get_nx(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables["x"]
                return len(onp.array(var_obj))

        def _get_runlen(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables["dt"]
                return onp.sum(onp.array(var_obj))

        def _get_ncr(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables["year_season"]
                return len(onp.array(var_obj))

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = f"SVATCROP_{crop_rotation_scenario}"

            settings.nx = self._get_nx(self._base_path.parent, "parameters.nc")
            settings.ny = 1
            settings.runlen = self._get_runlen(self._input_dir, "forcing.nc")
            settings.nitt_forc = len(self._read_var_from_nc("Time", self._input_dir, "forcing.nc"))

            settings.dx = 1
            settings.dy = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0
            settings.time_origin = "1999-12-31 00:00:00"

            settings.enable_irrigation = True
            settings.enable_net_irrigation = True
            settings.enable_crop_water_stress = True
            settings.enable_crop_phenology = True
            settings.enable_crop_rotation = True
            settings.enable_macropore_lower_boundary_condition = False
            settings.enable_adaptive_time_stepping = True

            if irrigation_scenario == "20-ufc":
                settings.fraction_ufc_of_irrigation = 0.20
            elif irrigation_scenario == "35-ufc":
                settings.fraction_ufc_of_irrigation = 0.35
            elif irrigation_scenario == "50-ufc":
                settings.fraction_ufc_of_irrigation = 0.5
            elif irrigation_scenario == "crop-specific":
                settings.enable_crop_specific_irrigation_demand = True

            if settings.enable_crop_rotation:
                settings.ncrops = 3
                settings.ncr = self._get_ncr(self._input_dir, f"{crop_rotation_scenario}.nc")

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "x",
                "y",
            ],
        )
        def set_grid(self, state):
            vs = state.variables

            # grid of model runs
            dx = allocate(state.dimensions, ("x"))
            dx = update(dx, at[:], 1)
            dy = allocate(state.dimensions, ("y"))
            dy = update(dy, at[:], 1)
            vs.x = update(vs.x, at[3:-2], npx.cumsum(dx[3:-2]))
            vs.y = update(vs.y, at[3:-2], npx.cumsum(dy[3:-2]))

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "lut_ilu",
                "lut_gc",
                "lut_gcm",
                "lut_is",
                "lut_rdlu",
                "lut_crops",
            ],
        )
        def set_look_up_tables(self, state):
            vs = state.variables

            vs.lut_ilu = update(vs.lut_ilu, at[:, :], lut.ARR_ILU)
            vs.lut_gc = update(vs.lut_gc, at[:, :], lut.ARR_GC)
            vs.lut_gcm = update(vs.lut_gcm, at[:, :], lut.ARR_GCM)
            vs.lut_is = update(vs.lut_is, at[:, :], lut.ARR_IS)
            vs.lut_rdlu = update(vs.lut_rdlu, at[:, :], lut.ARR_RDLU)
            vs.lut_crops = update(vs.lut_crops, at[:, :], lut.ARR_CP)

        @roger_routine
        def set_topography(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "lu_id",
                "z_soil",
                "dmpv",
                "lmpv",
                "theta_ac",
                "theta_ufc",
                "theta_pwp",
                "ks",
                "kf",
                "crop_type",
                "z_root",
                "z_root_crop",
                "theta_irr",
                "root_growth_scale",
                "canopy_growth_scale",
            ],
        )
        def set_parameters_setup(self, state):
            vs = state.variables
            settings = state.settings

            vs.crop_type = update(
                vs.crop_type,
                at[2:-2, 2:-2, 0],
                self._read_var_from_nc("crop", self._input_dir, f"{crop_rotation_scenario}.nc")[:, :, 1],
            )
            vs.crop_type = update(
                vs.crop_type,
                at[2:-2, 2:-2, 1],
                self._read_var_from_nc("crop", self._input_dir, f"{crop_rotation_scenario}.nc")[:, :, 2],
            )
            vs.crop_type = update(
                vs.crop_type,
                at[2:-2, 2:-2, 2],
                self._read_var_from_nc("crop", self._input_dir, f"{crop_rotation_scenario}.nc")[:, :, 3],
            )

            vs.z_root = update(vs.z_root, at[2:-2, 2:-2, :2], 200)
            vs.z_root_crop = update(vs.z_root_crop, at[2:-2, 2:-2, :2, 0], vs.z_root[2:-2, 2:-2, :2])

            vs.lu_id = update(vs.lu_id, at[2:-2, 2:-2], vs.crop_type[2:-2, 2:-2, 0])
            vs.z_soil = update(
                vs.z_soil, at[2:-2, 2:-2], self._read_var_from_nc("z_soil", self._base_path.parent, "parameters.nc")
            )
            vs.dmpv = update(
                vs.dmpv, at[2:-2, 2:-2], self._read_var_from_nc("dmpv", self._base_path.parent, "parameters.nc")
            )
            vs.lmpv = update(
                vs.lmpv, at[2:-2, 2:-2], self._read_var_from_nc("lmpv", self._base_path.parent, "parameters.nc")
            )
            vs.theta_ac = update(
                vs.theta_ac, at[2:-2, 2:-2], self._read_var_from_nc("theta_ac", self._base_path.parent, "parameters.nc")
            )
            vs.theta_ufc = update(
                vs.theta_ufc, at[2:-2, 2:-2], self._read_var_from_nc("theta_ufc", self._base_path.parent, "parameters.nc")
            )
            vs.theta_pwp = update(
                vs.theta_pwp, at[2:-2, 2:-2], self._read_var_from_nc("theta_pwp", self._base_path.parent, "parameters.nc")
            )
            vs.ks = update(vs.ks, at[2:-2, 2:-2], self._read_var_from_nc("ks", self._base_path.parent, "parameters.nc"))
            vs.kf = update(vs.kf, at[2:-2, 2:-2], 2500)

            vs.root_growth_scale = update(
                vs.root_growth_scale,
                at[2:-2, 2:-2],
                0.7,
            )
            vs.canopy_growth_scale = update(
                vs.canopy_growth_scale,
                at[2:-2, 2:-2],
                1.0,
            )

            if irrigation_scenario in ["35-ufc", "45-ufc", "50-ufc", "80-ufc"]:
                vs.theta_irr = update(
                    vs.theta_irr,
                    at[2:-2, 2:-2],
                    vs.theta_pwp[2:-2, 2:-2] + (vs.theta_ufc[2:-2, 2:-2] * settings.fraction_ufc_of_irrigation),
                )

        @roger_routine
        def set_parameters(self, state):
            vs = state.variables

            if (vs.month[vs.tau] != vs.month[vs.taum1]) & (vs.itt > 1):
                vs.update(calc_parameters_surface_kernel(state))

        @roger_routine
        def set_initial_conditions_setup(self, state):
            pass

        @roger_routine
        def set_initial_conditions(self, state):
            vs = state.variables

            vs.theta_rz = update(
                vs.theta_rz,
                at[2:-2, 2:-2, : vs.taup1],
                vs.theta_pwp[2:-2, 2:-2, npx.newaxis] + vs.theta_ufc[2:-2, 2:-2, npx.newaxis],
            )
            vs.theta_ss = update(
                vs.theta_ss,
                at[2:-2, 2:-2, : vs.taup1],
                vs.theta_pwp[2:-2, 2:-2, npx.newaxis] + vs.theta_ufc[2:-2, 2:-2, npx.newaxis],
            )

            vs.update(set_initial_conditions_crops_kernel(state))

        @roger_routine
        def set_boundary_conditions_setup(self, state):
            pass

        @roger_routine
        def set_boundary_conditions(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=["PREC", "TA", "TA_MIN", "TA_MAX", "PET"],
        )
        def set_forcing_setup(self, state):
            vs = state.variables

            vs.PREC = update(vs.PREC, at[:], self._read_var_from_nc("PREC", self._input_dir, "forcing.nc")[0, 0, :])
            vs.TA = update(vs.TA, at[:], self._read_var_from_nc("TA", self._input_dir, "forcing.nc")[0, 0, :])
            vs.TA_MIN = update(
                vs.TA_MIN, at[:], self._read_var_from_nc("TA_min", self._input_dir, "forcing.nc")[0, 0, :]
            )
            vs.TA_MAX = update(
                vs.TA_MAX, at[:], self._read_var_from_nc("TA_max", self._input_dir, "forcing.nc")[0, 0, :]
            )
            vs.PET = update(vs.PET, at[:], self._read_var_from_nc("PET", self._input_dir, "forcing.nc")[0, 0, :])

        @roger_routine
        def set_forcing(self, state):
            vs = state.variables
            settings = state.settings

            vs.irrig = update(vs.irrig, at[2:-2, 2:-2], 0)

            condt = vs.time % (24 * 60 * 60) == 0
            if condt:
                vs.itt_day = 0
                vs.year = update(
                    vs.year, at[1], self._read_var_from_nc("YEAR", self._input_dir, "forcing.nc")[vs.itt_forc]
                )
                vs.month = update(
                    vs.month, at[1], self._read_var_from_nc("MONTH", self._input_dir, "forcing.nc")[vs.itt_forc]
                )
                vs.doy = update(
                    vs.doy, at[1], self._read_var_from_nc("DOY", self._input_dir, "forcing.nc")[vs.itt_forc]
                )
                vs.prec_day = update(
                    vs.prec_day, at[:, :, :], vs.PREC[npx.newaxis, npx.newaxis, vs.itt_forc : vs.itt_forc + 6 * 24]
                )
                vs.ta_day = update(
                    vs.ta_day, at[:, :, :], vs.TA[npx.newaxis, npx.newaxis, vs.itt_forc : vs.itt_forc + 6 * 24]
                )
                vs.ta_min = update(
                    vs.ta_min,
                    at[:, :],
                    npx.min(vs.TA_MIN[npx.newaxis, npx.newaxis, vs.itt_forc : vs.itt_forc + 6 * 24], axis=-1),
                )
                vs.ta_max = update(
                    vs.ta_max,
                    at[:, :],
                    npx.max(vs.TA_MAX[npx.newaxis, npx.newaxis, vs.itt_forc : vs.itt_forc + 6 * 24], axis=-1),
                )
                vs.pet_day = update(
                    vs.pet_day, at[:, :, :], vs.PET[npx.newaxis, npx.newaxis, vs.itt_forc : vs.itt_forc + 6 * 24]
                )
                vs.itt_forc = vs.itt_forc + 6 * 24
                if vs.itt_forc < (settings.nitt_forc - 5 * 6 * 24):
                    # irrigate if sum of rainfall for the next 5 days is less than 1 mm
                    sum_rainfall_next5days = npx.sum(vs.PREC[vs.itt_forc:vs.itt_forc + 5 * 6 * 24])
                    if sum_rainfall_next5days <= 1 and vs.month[1] in [4, 5, 6, 7, 8, 9] and (vs.irr_demand[2:-2, 2:-2] > 0).any():
                        # irrigate 30 mm for 4 hours from 06:00 to 10:00
                        vs.prec_day = update(
                            vs.prec_day, at[2:-2, 2:-2, 6*6:10*6], npx.where(vs.irr_demand[2:-2, 2:-2] > 0, 30 / (6 * 4), 0)[:, :, npx.newaxis]
                        )
                        mask_crops = (vs.lu_id > 500) & (vs.lu_id < 599)
                        vs.irrig = update(
                            vs.irrig, at[2:-2, 2:-2], npx.where((vs.irr_demand[2:-2, 2:-2] > 0) & mask_crops[2:-2, 2:-2], 30, 0)
                        )


            if (vs.year[1] != vs.year[0]) & (vs.itt > 1):
                vs.itt_cr = vs.itt_cr + 2
                vs.crop_type = update(vs.crop_type, at[2:-2, 2:-2, 0], vs.crop_type[2:-2, 2:-2, 2])
                vs.crop_type = update(
                    vs.crop_type,
                    at[2:-2, 2:-2, 1],
                    self._read_var_from_nc("crop", self._input_dir, f"{crop_rotation_scenario}.nc")[:, :, vs.itt_cr],
                )
                vs.crop_type = update(
                    vs.crop_type,
                    at[2:-2, 2:-2, 2],
                    self._read_var_from_nc("crop", self._input_dir, f"{crop_rotation_scenario}.nc")[:, :, vs.itt_cr + 1],
                )

        @roger_routine
        def set_diagnostics(self, state, base_path=tmp_dir):
            diagnostics = state.diagnostics

            diagnostics["rate"].output_variables = [
                "prec",
                "pet",
                "aet",
                "transp",
                "evap_soil",
                "inf_mat_rz",
                "inf_mp_rz",
                "inf_sc_rz",
                "inf_ss",
                "q_rz",
                "q_ss",
                "cpr_rz",
                "re_rg",
                "re_rl",
                "q_hof",
            ]
            diagnostics["rate"].output_frequency = 24 * 60 * 60
            diagnostics["rate"].sampling_frequency = 1
            if base_path:
                diagnostics["rate"].base_output_path = base_path

            diagnostics["collect"].output_variables = [
                "S_rz",
                "S_ss",
                "S_s",
                "S",
                "S_pwp_rz",
                "S_fc_rz",
                "S_sat_rz",
                "S_pwp_ss",
                "S_fc_ss",
                "S_sat_ss",
                "theta",
                "z_root",
                "ground_cover",
                "lu_id",
                "ta",
                "ta_max",
                "irr_demand",
                "theta_rz",
                "theta_irr"
            ]
            diagnostics["collect"].output_frequency = 24 * 60 * 60
            diagnostics["collect"].sampling_frequency = 1
            if base_path:
                diagnostics["collect"].base_output_path = base_path

            # maximum bias of deterministic/numerical solution at time step t
            diagnostics["maximum"].output_variables = ["dS_num_error", "dS_rz_num_error", "dS_ss_num_error", "irrig"]
            diagnostics["maximum"].output_frequency = 24 * 60 * 60
            diagnostics["maximum"].sampling_frequency = 1
            if base_path:
                diagnostics["maximum"].base_output_path = base_path

        @roger_routine
        def after_timestep(self, state):
            vs = state.variables

            vs.update(after_timestep_kernel(state))
            vs.update(after_timestep_crops_kernel(state))

    @roger_kernel
    def set_initial_conditions_crops_kernel(state):
        vs = state.variables

        # calculate time since growing
        t_grow = allocate(state.dimensions, ("x", "y", "crops"))
        t_grow = update(
            t_grow,
            at[2:-2, 2:-2, :],
            npx.where(
                vs.z_root_crop[2:-2, 2:-2, vs.taum1, :] > 0,
                (-1 / vs.root_growth_rate[2:-2, 2:-2, :])
                * npx.log(
                    1
                    / (
                        (vs.z_root_crop[2:-2, 2:-2, vs.taum1, :] / 1000 - vs.z_root_crop_max[2:-2, 2:-2, :] / 1000)
                        * (-1 / (vs.z_root_crop_max[2:-2, 2:-2, :] / 1000 - vs.z_evap[2:-2, 2:-2, npx.newaxis] / 1000))
                    )
                ),
                0,
            ),
        )

        vs.t_grow_cc = update(vs.t_grow_cc, at[2:-2, 2:-2, :2, :], t_grow[2:-2, 2:-2, npx.newaxis, :])

        vs.t_grow_root = update(vs.t_grow_root, at[2:-2, 2:-2, :2, :], t_grow[2:-2, 2:-2, npx.newaxis, :])

        return KernelOutput(
            t_grow_cc=vs.t_grow_cc,
            t_grow_root=vs.t_grow_root,
        )

    @roger_kernel
    def after_timestep_kernel(state):
        vs = state.variables

        vs.ta = update(
            vs.ta,
            at[2:-2, 2:-2, vs.taum1],
            vs.ta[2:-2, 2:-2, vs.tau],
        )
        vs.z_root = update(
            vs.z_root,
            at[2:-2, 2:-2, vs.taum1],
            vs.z_root[2:-2, 2:-2, vs.tau],
        )
        vs.ground_cover = update(
            vs.ground_cover,
            at[2:-2, 2:-2, vs.taum1],
            vs.ground_cover[2:-2, 2:-2, vs.tau],
        )
        vs.S_sur = update(
            vs.S_sur,
            at[2:-2, 2:-2, vs.taum1],
            vs.S_sur[2:-2, 2:-2, vs.tau],
        )
        vs.S_int_top = update(
            vs.S_int_top,
            at[2:-2, 2:-2, vs.taum1],
            vs.S_int_top[2:-2, 2:-2, vs.tau],
        )
        vs.S_int_ground = update(
            vs.S_int_ground,
            at[2:-2, 2:-2, vs.taum1],
            vs.S_int_ground[2:-2, 2:-2, vs.tau],
        )
        vs.S_dep = update(
            vs.S_dep,
            at[2:-2, 2:-2, vs.taum1],
            vs.S_dep[2:-2, 2:-2, vs.tau],
        )
        vs.S_snow = update(
            vs.S_snow,
            at[2:-2, 2:-2, vs.taum1],
            vs.S_snow[2:-2, 2:-2, vs.tau],
        )
        vs.swe = update(
            vs.swe,
            at[2:-2, 2:-2, vs.taum1],
            vs.swe[2:-2, 2:-2, vs.tau],
        )
        vs.S_rz = update(
            vs.S_rz,
            at[2:-2, 2:-2, vs.taum1],
            vs.S_rz[2:-2, 2:-2, vs.tau],
        )
        vs.S_ss = update(
            vs.S_ss,
            at[2:-2, 2:-2, vs.taum1],
            vs.S_ss[2:-2, 2:-2, vs.tau],
        )
        vs.S_s = update(
            vs.S_s,
            at[2:-2, 2:-2, vs.taum1],
            vs.S_s[2:-2, 2:-2, vs.tau],
        )
        vs.S = update(
            vs.S,
            at[2:-2, 2:-2, vs.taum1],
            vs.S[2:-2, 2:-2, vs.tau],
        )
        vs.z_sat = update(
            vs.z_sat,
            at[2:-2, 2:-2, vs.taum1],
            vs.z_sat[2:-2, 2:-2, vs.tau],
        )
        vs.z_wf = update(
            vs.z_wf,
            at[2:-2, 2:-2, vs.taum1],
            vs.z_wf[2:-2, 2:-2, vs.tau],
        )
        vs.z_wf_t0 = update(
            vs.z_wf_t0,
            at[2:-2, 2:-2, vs.taum1],
            vs.z_wf_t0[2:-2, 2:-2, vs.tau],
        )
        vs.z_wf_t1 = update(
            vs.z_wf_t1,
            at[2:-2, 2:-2, vs.taum1],
            vs.z_wf_t1[2:-2, 2:-2, vs.tau],
        )
        vs.y_mp = update(
            vs.y_mp,
            at[2:-2, 2:-2, vs.taum1],
            vs.y_mp[2:-2, 2:-2, vs.tau],
        )
        vs.y_sc = update(
            vs.y_sc,
            at[2:-2, 2:-2, vs.taum1],
            vs.y_sc[2:-2, 2:-2, vs.tau],
        )
        vs.theta_rz = update(
            vs.theta_rz,
            at[2:-2, 2:-2, vs.taum1],
            vs.theta_rz[2:-2, 2:-2, vs.tau],
        )
        vs.theta_ss = update(
            vs.theta_ss,
            at[2:-2, 2:-2, vs.taum1],
            vs.theta_ss[2:-2, 2:-2, vs.tau],
        )
        vs.theta = update(
            vs.theta,
            at[2:-2, 2:-2, vs.taum1],
            vs.theta[2:-2, 2:-2, vs.tau],
        )
        vs.k_rz = update(
            vs.k_rz,
            at[2:-2, 2:-2, vs.taum1],
            vs.k_rz[2:-2, 2:-2, vs.tau],
        )
        vs.k_ss = update(
            vs.k_ss,
            at[2:-2, 2:-2, vs.taum1],
            vs.k_ss[2:-2, 2:-2, vs.tau],
        )
        vs.k = update(
            vs.k,
            at[2:-2, 2:-2, vs.taum1],
            vs.k[2:-2, 2:-2, vs.tau],
        )
        vs.h_rz = update(
            vs.h_rz,
            at[2:-2, 2:-2, vs.taum1],
            vs.h_rz[2:-2, 2:-2, vs.tau],
        )
        vs.h_ss = update(
            vs.h_ss,
            at[2:-2, 2:-2, vs.taum1],
            vs.h_ss[2:-2, 2:-2, vs.tau],
        )
        vs.h = update(
            vs.h,
            at[2:-2, 2:-2, vs.taum1],
            vs.h[2:-2, 2:-2, vs.tau],
        )
        vs.z0 = update(
            vs.z0,
            at[2:-2, 2:-2, vs.taum1],
            vs.z0[2:-2, 2:-2, vs.tau],
        )
        # set to 0 for numerical errors
        vs.S_fp_rz = update(
            vs.S_fp_rz,
            at[2:-2, 2:-2],
            npx.where((vs.S_fp_rz[2:-2, 2:-2] > -1e-6) & (vs.S_fp_rz[2:-2, 2:-2] < 0), 0, vs.S_fp_rz[2:-2, 2:-2]),
        )
        vs.S_lp_rz = update(
            vs.S_lp_rz,
            at[2:-2, 2:-2],
            npx.where((vs.S_lp_rz[2:-2, 2:-2] > -1e-6) & (vs.S_lp_rz[2:-2, 2:-2] < 0), 0, vs.S_lp_rz[2:-2, 2:-2]),
        )
        vs.S_fp_ss = update(
            vs.S_fp_ss,
            at[2:-2, 2:-2],
            npx.where((vs.S_fp_ss[2:-2, 2:-2] > -1e-6) & (vs.S_fp_ss[2:-2, 2:-2] < 0), 0, vs.S_fp_ss[2:-2, 2:-2]),
        )
        vs.S_lp_ss = update(
            vs.S_lp_ss,
            at[2:-2, 2:-2],
            npx.where((vs.S_lp_ss[2:-2, 2:-2] > -1e-6) & (vs.S_lp_ss[2:-2, 2:-2] < 0), 0, vs.S_lp_ss[2:-2, 2:-2]),
        )
        vs.prec = update(
            vs.prec,
            at[2:-2, 2:-2, vs.taum1],
            vs.prec[2:-2, 2:-2, vs.tau],
        )
        vs.event_id = update(
            vs.event_id,
            at[vs.taum1],
            vs.event_id[vs.tau],
        )
        vs.year = update(
            vs.year,
            at[vs.taum1],
            vs.year[vs.tau],
        )
        vs.month = update(
            vs.month,
            at[vs.taum1],
            vs.month[vs.tau],
        )
        vs.doy = update(
            vs.doy,
            at[vs.taum1],
            vs.doy[vs.tau],
        )

        return KernelOutput(
            ta=vs.ta,
            z_root=vs.z_root,
            ground_cover=vs.ground_cover,
            S_sur=vs.S_sur,
            S_int_top=vs.S_int_top,
            S_int_ground=vs.S_int_ground,
            S_dep=vs.S_dep,
            S_snow=vs.S_snow,
            swe=vs.swe,
            S_rz=vs.S_rz,
            S_ss=vs.S_ss,
            S_s=vs.S_s,
            S=vs.S,
            z_sat=vs.z_sat,
            z_wf=vs.z_wf,
            z_wf_t0=vs.z_wf_t0,
            z_wf_t1=vs.z_wf_t1,
            y_mp=vs.y_mp,
            y_sc=vs.y_sc,
            theta_rz=vs.theta_rz,
            theta_ss=vs.theta_ss,
            theta=vs.theta,
            h_rz=vs.h_rz,
            h_ss=vs.h_ss,
            h=vs.h,
            k_rz=vs.k_rz,
            k_ss=vs.k_ss,
            k=vs.k,
            z0=vs.z0,
            prec=vs.prec,
            event_id=vs.event_id,
            year=vs.year,
            month=vs.month,
            doy=vs.doy,
            S_fp_rz=vs.S_fp_rz,
            S_lp_rz=vs.S_lp_rz,
            S_fp_ss=vs.S_fp_ss,
            S_lp_ss=vs.S_lp_ss,
        )

    @roger_kernel
    def after_timestep_crops_kernel(state):
        vs = state.variables

        vs.ta_min = update(vs.ta_min, at[2:-2, 2:-2, vs.taum1], vs.ta_min[2:-2, 2:-2, vs.tau])
        vs.ta_max = update(vs.ta_max, at[2:-2, 2:-2, vs.taum1], vs.ta_max[2:-2, 2:-2, vs.tau])
        vs.gdd_sum = update(vs.gdd_sum, at[2:-2, 2:-2, vs.taum1, :], vs.gdd_sum[2:-2, 2:-2, vs.tau, :])
        vs.t_grow_cc = update(vs.t_grow_cc, at[2:-2, 2:-2, vs.taum1, :], vs.t_grow_cc[2:-2, 2:-2, vs.tau, :])
        vs.t_grow_root = update(vs.t_grow_root, at[2:-2, 2:-2, vs.taum1, :], vs.t_grow_root[2:-2, 2:-2, vs.tau, :])
        vs.ccc = update(vs.ccc, at[2:-2, 2:-2, vs.taum1, :], vs.ccc[2:-2, 2:-2, vs.tau, :])
        vs.z_root_crop = update(vs.z_root_crop, at[2:-2, 2:-2, vs.taum1, :], vs.z_root_crop[2:-2, 2:-2, vs.tau, :])
        vs.re_rg_pwp = update(vs.re_rg_pwp, at[2:-2, 2:-2], 0)
        vs.re_rg = update(vs.re_rg, at[2:-2, 2:-2], 0)
        vs.re_rl_pwp = update(vs.re_rl_pwp, at[2:-2, 2:-2], 0)
        vs.re_rl = update(vs.re_rl, at[2:-2, 2:-2], 0)

        return KernelOutput(
            ta_min=vs.ta_min,
            ta_max=vs.ta_max,
            gdd_sum=vs.gdd_sum,
            t_grow_cc=vs.t_grow_cc,
            t_grow_root=vs.t_grow_root,
            ccc=vs.ccc,
            z_root_crop=vs.z_root_crop,
            re_rg_pwp=vs.re_rg_pwp,
            re_rg=vs.re_rg,
            re_rl_pwp=vs.re_rl_pwp,
            re_rl=vs.re_rl,
        )

    model = SVATCROPSetup()
    write_forcing(model._input_dir, enable_crop_phenology=True)
    crop_rotation_dir = model._base_path.parent / "input" / "crop_rotation_scenarios" / crop_rotation_scenario
    write_crop_rotation(crop_rotation_dir)
    if not os.path.exists(model._input_dir / f"{crop_rotation_scenario}.nc"):
        shutil.copy2(crop_rotation_dir / "crop_rotation.nc", model._input_dir / f"{crop_rotation_scenario}.nc")
    model.setup()
    model.run()
    return


if __name__ == "__main__":
    main()
