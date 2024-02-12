from pathlib import Path
import h5netcdf
import numpy as onp
import pandas as pd
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option("--location", type=click.Choice(["freiburg", "lahr", "muellheim", 
                                               "stockach", "gottmadingen", "weingarten", 
                                               "eppingen-elsenz", "bruchsal-heidelsheim", "bretten", 
                                               "ehingen-kirchen", "merklingen", "hayingen", 
                                               "kupferzell", "oehringen", "vellberg-kleinaltdorf"]), 
                                               default="freiburg")
@click.option("--crop-rotation-scenario", type=click.Choice(["winter-wheat_clover",
                                                             "winter-wheat_silage-corn",
                                                             "winter-wheat_winter-rape",
                                                             "summer-wheat_winter-wheat", 
                                                             "summer-wheat_clover_winter-wheat",
                                                             "winter-wheat_clover_silage-corn",
                                                             "winter-wheat_sugar-beet_silage-corn",
                                                             "summer-wheat_winter-wheat_silage-corn",
                                                             "summer-wheat_winter-wheat_winter-rape", 
                                                             "winter-wheat_winter-rape",
                                                             "winter-wheat_winter-grain-pea_winter-rape", 
                                                             "winter-wheat_silage-corn_yellow-mustard", 
                                                             "summer-wheat_winter-wheat_silage-corn_yellow-mustard",
                                                             "winter-wheat_sugar-beet_silage-corn_yellow-mustard",
                                                             "summer-wheat_winter-wheat_silage-corn_yellow-mustard", 
                                                             "summer-wheat_winter-wheat_winter-rape_yellow-mustard"]), default="winter-wheat_silage-corn")
@click.option("-ft", "--fertilization-intensity", type=click.Choice(["low", "medium", "high"]), default="medium")
@click.option("-id", "--id", type=str, default="5-8_2090295_1")
@click.option("--row", type=int, default=0)
@click.option("-td", "--tmp-dir", type=str, default=Path(__file__).parent)
@roger_base_cli
def main(location, crop_rotation_scenario, fertilization_intensity, id, row, tmp_dir):
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, update_add, at, scipy_stats as sstx
    import roger.lookuptables as lut
    from roger.core.utilities import _get_row_no

    class SVATCROPNITRATESetup(RogerSetup):
        """A SVAT-CROP transport model for nitrate."""

        _base_path = Path(__file__).parent
        # if tmp_dir:
        #     # read fluxes and states from local SSD on cluster node
        #     _input_dir = Path(tmp_dir)
        # else:
        #     _input_dir = _base_path.parent / "output" / "svat_crop_nitrate"
        _input_dir = _base_path.parent / "output" / "svat_crop"
        # _input_dir = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "output" / "svat_crop"

        def _read_var_from_nc(self, var, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables[var]
                return npx.array(var_obj, dtype=npx.float32)

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
                var_obj = infile.variables["Time"]
                return len(onp.array(var_obj)) * 60 * 60 * 24

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = f"SVATCROPNITRATE_{id}_{location}_{crop_rotation_scenario}"
            settings.sas_solver = "deterministic"
            settings.sas_solver_substeps = 8

            settings.nx, settings.ny = self._get_nx(self._base_path, "parameters.nc"), 1
            settings.nitt = self._get_nitt(
                self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
            )
            settings.nitt_forc = settings.nitt
            settings.ages = 1000
            settings.nages = settings.ages + 1
            settings.runlen_warmup = 2 * 365 * 24 * 60 * 60
            settings.runlen = self._get_runlen(
                self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
            )

            settings.dx = 1
            settings.dy = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0
            settings.time_origin = "2012-12-31 00:00:00"

            settings.enable_crop_phenology = True
            settings.enable_crop_rotation = True
            settings.enable_offline_transport = True
            settings.enable_nitrate = True
            settings.enable_age_statistics = True

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "dt_secs",
                "dt",
                "ages",
                "nages",
                "x",
                "y",
            ],
        )
        def set_grid(self, state):
            vs = state.variables
            settings = state.settings

            # temporal grid
            vs.dt_secs = 60 * 60 * 24
            vs.dt = 60 * 60 * 24 / (60 * 60)
            vs.ages = update(vs.ages, at[:], npx.arange(1, settings.nages))
            vs.nages = update(vs.nages, at[:], npx.arange(settings.nages))
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
                "lut_fert1",
                "lut_fert2",
                "lut_fert3",
            ],
        )
        def set_look_up_tables(self, state):
            vs = state.variables

            vs.lut_fert1 = update(vs.lut_fert1, at[:, :], lut.ARR_FERT1)
            vs.lut_fert2 = update(vs.lut_fert2, at[:, :], lut.ARR_FERT2)
            vs.lut_fert3 = update(vs.lut_fert3, at[:, :], lut.ARR_FERT3)

        @roger_routine
        def set_topography(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "S_PWP_RZ",
                "S_SAT_RZ",
                "S_PWP_SS",
                "S_SAT_SS",
                "S_pwp_rz",
                "S_pwp_ss",
                "S_sat_rz",
                "S_sat_ss",
                "sas_params_evap_soil",
                "sas_params_cpr_rz",
                "sas_params_transp",
                "sas_params_q_rz",
                "sas_params_q_ss",
                "sas_params_re_rg",
                "sas_params_re_rl",
                "alpha_transp",
                "alpha_q",
                "km_denit_rz",
                "km_denit_ss",
                "dmax_denit_rz",
                "dmax_denit_ss",
                "km_nit_rz",
                "km_nit_ss",
                "dmax_nit_rz",
                "dmax_nit_ss",
                "kmin_rz",
                "kmin_ss",
                "kfix_rz",
                "z_soil",
                "phi_soil_temp",
                "damp_soil_temp",
                "YEAR",
                "DOY",
                "LU_ID",
                "Z_ROOT",
                "c_fert",
            ],
        )
        def set_parameters_setup(self, state):
            vs = state.variables
            settings = state.settings

            vs.S_PWP_RZ = update(
                vs.S_PWP_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "S_pwp_rz", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.S_SAT_RZ = update(
                vs.S_SAT_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "S_sat_rz", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.S_PWP_SS = update(
                vs.S_PWP_SS,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "S_pwp_ss", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.S_SAT_SS = update(
                vs.S_SAT_SS,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "S_sat_ss", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )

            vs.S_pwp_rz = update(
                vs.S_pwp_rz,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "S_pwp_rz", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.S_pwp_ss = update(
                vs.S_pwp_ss,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "S_pwp_ss", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.S_sat_rz = update(
                vs.S_sat_rz,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "S_sat_rz", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                )[:, :, 0],
            )
            vs.S_sat_ss = update(
                vs.S_sat_ss,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "S_sat_ss", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                )[:, :, 0],
            )

            # partition coefficients
            vs.alpha_transp = update(vs.alpha_transp, at[2:-2, 2:-2], self._read_var_from_nc("alpha_transp", self._base_path, "parameters.nc"))
            vs.alpha_q = update(vs.alpha_q, at[2:-2, 2:-2], self._read_var_from_nc("alpha_q", self._base_path, "parameters.nc"))

            # SAS parameters
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.25)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.25)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 62)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], 0.3)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 4], self._read_var_from_nc("c2_transp", self._base_path, "parameters.nc"))
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 61)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], 1.5)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], self._read_var_from_nc("c2_q_rz", self._base_path, "parameters.nc"))
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 61)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], 1.5)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], self._read_var_from_nc("c2_q_ss", self._base_path, "parameters.nc"))
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], vs.S_pwp_ss[2:-2, 2:-2])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2])
            vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 6)
            vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 1], 0.5)
            vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 6)
            vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 1], 10)

            # denitrification parameters
            _c = 0.05
            vs.km_denit_rz = update(vs.km_denit_rz, at[2:-2, 2:-2], self._read_var_from_nc("km_denit", self._base_path, "parameters.nc"))
            vs.km_denit_ss = update(vs.km_denit_ss, at[2:-2, 2:-2], _c * self._read_var_from_nc("km_denit", self._base_path, "parameters.nc"))
            vs.dmax_denit_rz = update(vs.dmax_denit_rz, at[2:-2, 2:-2], self._read_var_from_nc("dmax_denit", self._base_path, "parameters.nc"))
            vs.dmax_denit_ss = update(vs.dmax_denit_ss, at[2:-2, 2:-2], _c * self._read_var_from_nc("dmax_denit", self._base_path, "parameters.nc"))
            # nitrification parameters
            vs.km_nit_rz = update(vs.km_nit_rz, at[2:-2, 2:-2], self._read_var_from_nc("km_nit", self._base_path, "parameters.nc"))
            vs.km_nit_ss = update(vs.km_nit_ss, at[2:-2, 2:-2], _c * self._read_var_from_nc("km_nit", self._base_path, "parameters.nc"))
            vs.dmax_nit_rz = update(vs.dmax_nit_rz, at[2:-2, 2:-2], self._read_var_from_nc("dmax_nit", self._base_path, "parameters.nc"))
            vs.dmax_nit_ss = update(vs.dmax_nit_ss, at[2:-2, 2:-2], _c * self._read_var_from_nc("dmax_nit", self._base_path, "parameters.nc"))
            # soil nitrogen mineralization parameters
            vs.kmin_rz = update(vs.kmin_rz, at[2:-2, 2:-2], self._read_var_from_nc("kmin", self._base_path, "parameters.nc"))
            vs.kmin_ss = update(vs.kmin_ss, at[2:-2, 2:-2], _c * self._read_var_from_nc("kmin", self._base_path, "parameters.nc"))
            # soil nitrogen fixation parameters
            vs.kfix_rz = update(vs.kmin_rz, at[2:-2, 2:-2], self._read_var_from_nc("kfix", self._base_path, "parameters.nc"))

            # soil temperature parameters
            vs.z_soil = update(
                vs.z_soil, at[2:-2, 2:-2], self._read_var_from_nc("z_soil", self._base_path, "z_soil.nc")[row, 0]
            )
            vs.phi_soil_temp = update(vs.phi_soil_temp, at[2:-2, 2:-2], 91)
            # dampening depth of soil temperature depends on clay content
            clay = self._read_var_from_nc("clay", self._base_path, "clay.nc")[row, 0]
            vs.damp_soil_temp = update(vs.damp_soil_temp, at[2:-2, 2:-2], 12 + 4 * (1 - (clay / settings.clay_max)))

            vs.YEAR = update(
                vs.YEAR,
                at[:],
                self._read_var_from_nc(
                    "year", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.DOY = update(
                vs.DOY,
                at[:],
                self._read_var_from_nc(
                    "doy", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.LU_ID = update(
                vs.LU_ID,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "lu_id", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.Z_ROOT = update(
                vs.Z_ROOT,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "z_root", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )

            vs.c_fert = update(vs.c_fert, at[2:-2, 2:-2], self._read_var_from_nc("c_fert", self._base_path, "parameters.nc"))

        @roger_routine
        def set_parameters(self, state):
            vs = state.variables

            vs.S_pwp_rz = update(vs.S_pwp_rz, at[2:-2, 2:-2], vs.S_PWP_RZ[2:-2, 2:-2, vs.itt])
            vs.S_sat_rz = update(vs.S_sat_rz, at[2:-2, 2:-2], vs.S_SAT_RZ[2:-2, 2:-2, vs.itt])
            vs.S_pwp_ss = update(vs.S_pwp_ss, at[2:-2, 2:-2], vs.S_PWP_SS[2:-2, 2:-2, vs.itt])
            vs.S_sat_ss = update(vs.S_sat_ss, at[2:-2, 2:-2], vs.S_SAT_SS[2:-2, 2:-2, vs.itt])

            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], vs.S_pwp_ss[2:-2, 2:-2])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2])

        @roger_routine(
            dist_safe=False,
            local_variables=["S_rz", "S_rz_init", "S_ss", "S_ss_init", "S_s", "itt", "taup1"],
        )
        def set_initial_conditions_setup(self, state):
            vs = state.variables

            vs.S_rz = update(
                vs.S_rz,
                at[2:-2, 2:-2, :vs.taup1],
                self._read_var_from_nc(
                    "S_rz", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                )[:, :, vs.itt, npx.newaxis],
            )
            vs.S_ss = update(
                vs.S_ss,
                at[2:-2, 2:-2, :vs.taup1],
                self._read_var_from_nc(
                    "S_ss", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                )[:, :, vs.itt, npx.newaxis],
            )
            vs.S_s = update(
                vs.S_s, at[2:-2, 2:-2, :vs.taup1], vs.S_rz[2:-2, 2:-2, :vs.taup1] + vs.S_ss[2:-2, 2:-2, :vs.taup1]
            )
            vs.S_rz_init = update(vs.S_rz_init, at[2:-2, 2:-2], vs.S_rz[2:-2, 2:-2, 0])
            vs.S_ss_init = update(vs.S_ss_init, at[2:-2, 2:-2], vs.S_ss[2:-2, 2:-2, 0])

        @roger_routine
        def set_initial_conditions(self, state):
            vs = state.variables
            settings = state.settings

            # uniform StorAge
            arr0 = allocate(state.dimensions, ("x", "y"))
            vs.sa_rz = update(
                vs.sa_rz,
                at[2:-2, 2:-2, :vs.taup1, 1:],
                npx.diff(npx.linspace(arr0[2:-2, 2:-2], vs.S_rz[2:-2, 2:-2, vs.tau], settings.ages, axis=-1), axis=-1)[
                    :, :, npx.newaxis, :
                ],
            )
            vs.sa_ss = update(
                vs.sa_ss,
                at[2:-2, 2:-2, :vs.taup1, 1:],
                npx.diff(npx.linspace(arr0[2:-2, 2:-2], vs.S_ss[2:-2, 2:-2, vs.tau], settings.ages, axis=-1), axis=-1)[
                    :, :, npx.newaxis, :
                ],
            )

            vs.SA_rz = update(
                vs.SA_rz,
                at[2:-2, 2:-2, :, 1:],
                npx.cumsum(vs.sa_rz[2:-2, 2:-2, :, :], axis=-1),
            )

            vs.SA_ss = update(
                vs.SA_ss,
                at[2:-2, 2:-2, :, 1:],
                npx.cumsum(vs.sa_rz[2:-2, 2:-2, :, :], axis=-1),
            )

            vs.sa_s = update(
                vs.sa_s,
                at[2:-2, 2:-2, :, :],
                vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :],
            )
            vs.SA_s = update(
                vs.SA_s,
                at[2:-2, 2:-2, :, 1:],
                npx.cumsum(vs.sa_s[2:-2, 2:-2, :, :], axis=-1),
            )

            # initial nitrate concentration (in mg/l)
            vs.C_rz = update(vs.C_rz, at[2:-2, 2:-2, :vs.taup1], 10)
            vs.C_ss = update(vs.C_ss, at[2:-2, 2:-2, :vs.taup1], 10)
            # initial mineral soil nitrogen
            vs.Nmin_rz = update(vs.Nmin_rz, at[2:-2, 2:-2, :vs.taup1, :], (50 / settings.ages) * settings.dx * settings.dy * 100)
            vs.Nmin_ss = update(vs.Nmin_ss, at[2:-2, 2:-2, :vs.taup1, :], (50 / settings.ages) * settings.dx * settings.dy * 100)
            vs.msa_rz = update(
                vs.msa_rz,
                at[2:-2, 2:-2, :vs.taup1, :], vs.C_rz[2:-2, 2:-2, :vs.taup1, npx.newaxis] * vs.sa_rz[2:-2, 2:-2, :vs.taup1, :],
            )
            vs.msa_ss = update(
                vs.msa_ss,
                at[2:-2, 2:-2, :vs.taup1, :], vs.C_ss[2:-2, 2:-2, :vs.taup1, npx.newaxis] * vs.sa_ss[2:-2, 2:-2, :vs.taup1, :],
            )
            vs.msa_s = update(
                vs.msa_s,
                at[2:-2, 2:-2, :, :], vs.msa_rz[2:-2, 2:-2, :, :] + vs.msa_ss[2:-2, 2:-2, :, :],
            )
            vs.M_rz = update(
                vs.M_rz,
                at[2:-2, 2:-2, :], npx.sum(vs.msa_rz[2:-2, 2:-2, :, :], axis=-1),
            )
            vs.M_ss = update(
                vs.M_ss,
                at[2:-2, 2:-2, :], npx.sum(vs.msa_ss[2:-2, 2:-2, :, :], axis=-1),
            )
            vs.M_s = update(
                vs.M_s,
                at[2:-2, 2:-2, :], npx.sum(vs.msa_s[2:-2, 2:-2, :, :], axis=-1),
                )


        @roger_routine
        def set_boundary_conditions_setup(self, state):
            pass

        @roger_routine
        def set_boundary_conditions(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "PREC_DIST_DAILY",
                "INF_MAT_RZ",
                "INF_PF_RZ",
                "INF_PF_SS",
                "TRANSP",
                "EVAP_SOIL",
                "CPR_RZ",
                "Q_RZ",
                "Q_SS",
                "RE_RG",
                "RE_RL",
                "S_RZ",
                "S_SS",
                "S_S",
                "TA",
                "YEAR",
                "DOY",
                "ta_year",
                "LU_ID",
                "Z_ROOT",
            ],
        )
        def set_forcing_setup(self, state):
            vs = state.variables

            vs.PREC_DIST_DAILY = update(
                vs.PREC_DIST_DAILY,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "prec", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.INF_MAT_RZ = update(
                vs.INF_MAT_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "inf_mat_rz",
                    self._input_dir,
                    f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc",
                ),
            )
            vs.INF_PF_RZ = update(
                vs.INF_PF_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "inf_mp_rz",
                    self._input_dir,
                    f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc",
                )
                + self._read_var_from_nc(
                    "inf_sc_rz",
                    self._input_dir,
                    f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc",
                ),
            )
            vs.INF_PF_SS = update(
                vs.INF_PF_SS,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "inf_ss", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.TRANSP = update(
                vs.TRANSP,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "transp", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.EVAP_SOIL = update(
                vs.EVAP_SOIL,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "evap_soil",
                    self._input_dir,
                    f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc",
                ),
            )
            vs.CPR_RZ = update(
                vs.CPR_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "cpr_rz", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.Q_RZ = update(
                vs.Q_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "q_rz", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.Q_SS = update(
                vs.Q_SS,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "q_ss", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.RE_RG = update(
                vs.RE_RG,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "re_rg", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.RE_RL = update(
                vs.RE_RL,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "re_rl", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.S_RZ = update(
                vs.S_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "S_rz", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.S_SS = update(
                vs.S_SS,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "S_ss", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.S_S = update(vs.S_S, at[2:-2, 2:-2, :], vs.S_RZ[2:-2, 2:-2, :] + vs.S_SS[2:-2, 2:-2, :])
            vs.TA = update(
                vs.TA,
                at[:],
                self._read_var_from_nc(
                    "ta", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                )[0, 0, :],
            )
            vs.YEAR = update(
                vs.YEAR,
                at[:],
                self._read_var_from_nc(
                    "year", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.DOY = update(
                vs.DOY,
                at[:],
                self._read_var_from_nc(
                    "doy", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.ta_year = update(
                vs.ta_year,
                at[2:-2, 2:-2],
                npx.mean(vs.TA[:365])[npx.newaxis, npx.newaxis],
            )
            vs.LU_ID = update(
                vs.LU_ID,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "lu_id", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )
            vs.Z_ROOT = update(
                vs.Z_ROOT,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "z_root", self._input_dir, f"SVATCROP_{id}_{location}_{crop_rotation_scenario}.nc"
                ),
            )

        @roger_routine
        def set_forcing(self, state):
            vs = state.variables
            settings = state.settings

            vs.prec = update(vs.prec, at[2:-2, 2:-2, vs.tau], vs.PREC_DIST_DAILY[2:-2, 2:-2, vs.itt])
            vs.inf_mat_rz = update(vs.inf_mat_rz, at[2:-2, 2:-2], vs.INF_MAT_RZ[2:-2, 2:-2, vs.itt])
            vs.inf_pf_rz = update(vs.inf_pf_rz, at[2:-2, 2:-2], vs.INF_PF_RZ[2:-2, 2:-2, vs.itt])
            vs.inf_pf_ss = update(vs.inf_pf_ss, at[2:-2, 2:-2], vs.INF_PF_SS[2:-2, 2:-2, vs.itt])
            vs.transp = update(vs.transp, at[2:-2, 2:-2], vs.TRANSP[2:-2, 2:-2, vs.itt])
            vs.evap_soil = update(vs.evap_soil, at[2:-2, 2:-2], vs.EVAP_SOIL[2:-2, 2:-2, vs.itt])
            vs.cpr_rz = update(vs.cpr_rz, at[2:-2, 2:-2], vs.CPR_RZ[2:-2, 2:-2, vs.itt])
            vs.q_rz = update(vs.q_rz, at[2:-2, 2:-2], vs.Q_RZ[2:-2, 2:-2, vs.itt])
            vs.q_ss = update(vs.q_ss, at[2:-2, 2:-2], vs.Q_SS[2:-2, 2:-2, vs.itt])
            vs.re_rg = update(vs.re_rg, at[2:-2, 2:-2], vs.RE_RG[2:-2, 2:-2, vs.itt])
            vs.re_rl = update(vs.re_rl, at[2:-2, 2:-2], vs.RE_RL[2:-2, 2:-2, vs.itt])
            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, vs.tau], vs.S_RZ[2:-2, 2:-2, vs.itt])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, vs.tau], vs.S_SS[2:-2, 2:-2, vs.itt])
            vs.S_s = update(vs.S_s, at[2:-2, 2:-2, vs.tau], vs.S_rz[2:-2, 2:-2, vs.tau] + vs.S_ss[2:-2, 2:-2, vs.tau])
            vs.ta = update(vs.ta, at[2:-2, 2:-2, vs.tau], vs.TA[vs.itt])
            vs.doy = update(vs.doy, at[1], vs.DOY[vs.itt])
            vs.year = update(vs.year, at[1], vs.YEAR[vs.itt])
            vs.z_root = update(vs.z_root, at[2:-2, 2:-2, vs.tau], vs.Z_ROOT[2:-2, 2:-2, vs.itt])

            # set main crop type
            if vs.itt <= 1:
                if vs.itt + 364 < settings.nitt:
                    crop_type = npx.nanmax(npx.where(vs.LU_ID[2:-2, 2:-2, vs.itt:vs.itt+364]==599, npx.nan, vs.LU_ID[2:-2, 2:-2, vs.itt:vs.itt+364]), axis=-1)
                else:
                    crop_type = npx.nanmax(npx.where(vs.LU_ID[2:-2, 2:-2, settings.nitt-364:]==599, npx.nan, vs.LU_ID[2:-2, 2:-2, settings.nitt-364:]), axis=-1)
                vs.lu_id = update(vs.lu_id, at[2:-2, 2:-2], crop_type)
            if vs.year[vs.tau] > vs.year[vs.taum1]:
                if vs.itt + 364 < settings.nitt:
                    crop_type = npx.nanmax(npx.where(vs.LU_ID[2:-2, 2:-2, vs.itt:vs.itt+364]==599, npx.nan, vs.LU_ID[2:-2, 2:-2, vs.itt:vs.itt+364]), axis=-1)
                else:
                    crop_type = npx.nanmax(npx.where(vs.LU_ID[2:-2, 2:-2, settings.nitt-364:]==599, npx.nan, vs.LU_ID[2:-2, 2:-2, settings.nitt-364:]), axis=-1)
                vs.lu_id = update(vs.lu_id, at[2:-2, 2:-2], crop_type)
            # apply nitrogen fertilizer
            if vs.itt <= 1:
                vs.update(set_fertilizer_kernel(state))
            if vs.year[vs.tau] > vs.year[vs.taum1]:
                vs.update(set_fertilizer_kernel(state))
            vs.update(apply_fertilizer_kernel(state))

        @roger_routine
        def set_diagnostics(self, state, base_path=tmp_dir):
            diagnostics = state.diagnostics

            diagnostics["rate"].output_variables = ["M_q_ss", "M_transp"]
            diagnostics["rate"].output_frequency = 24 * 60 * 60
            diagnostics["rate"].sampling_frequency = 1
            if base_path:
                diagnostics["rate"].base_output_path = base_path

            diagnostics["average"].output_variables = [
                "tt10_q_ss",
                "tt50_q_ss",
                "tt90_q_ss",
                "ttavg_q_ss",
                "tt10_transp",
                "tt50_transp",
                "tt90_transp",
                "ttavg_transp",
                "rt10_s",
                "rt50_s",
                "rt90_s",
                "rtavg_s",
            ]
            diagnostics["average"].output_frequency = 24 * 60 * 60
            diagnostics["average"].sampling_frequency = 1
            if base_path:
                diagnostics["average"].base_output_path = base_path

            diagnostics["collect"].output_variables = ["M_s", "Nmin_s"]
            diagnostics["collect"].output_frequency = 24 * 60 * 60
            diagnostics["collect"].sampling_frequency = 1
            if base_path:
                diagnostics["collect"].base_output_path = base_path

            # maximum bias of deterministic/numerical solution at time step t
            diagnostics["maximum"].output_variables = ["dS_num_error", "dC_num_error"]
            diagnostics["maximum"].output_frequency = 24 * 60 * 60
            diagnostics["maximum"].sampling_frequency = 1
            if base_path:
                diagnostics["maximum"].base_output_path = base_path

        @roger_routine
        def after_timestep(self, state):
            vs = state.variables

            vs.year = update(
                vs.year,
                at[vs.taum1],
                vs.year[vs.tau],
            )

    @roger_kernel
    def set_fertilizer_kernel(state):
        vs = state.variables

        if fertilization_intensity == "low":
            lut_fert = vs.lut_fert1
        elif fertilization_intensity == "medium":
            lut_fert = vs.lut_fert2
        elif fertilization_intensity == "high":
            lut_fert = vs.lut_fert3

        for i in range(500, 600):
            mask = vs.lu_id == i 
            row_no = _get_row_no(vs.lut_fert1[:, 0], i)
            vs.doy_fert1 = update(
                vs.doy_fert1,
                at[2:-2, 2:-2],
                npx.where(mask[2:-2, 2:-2], lut_fert[row_no, 1], vs.doy_fert1[2:-2, 2:-2]),
            )
            vs.doy_fert2 = update(
                vs.doy_fert2,
                at[2:-2, 2:-2],
                npx.where(mask[2:-2, 2:-2], lut_fert[row_no, 2], vs.doy_fert2[2:-2, 2:-2]),
            )
            vs.doy_fert3 = update(
                vs.doy_fert3,
                at[2:-2, 2:-2],
                npx.where(mask[2:-2, 2:-2], lut_fert[row_no, 3], vs.doy_fert3[2:-2, 2:-2]),
            )
            vs.N_fert1 = update(
                vs.N_fert1,
                at[2:-2, 2:-2],
                npx.where(mask[2:-2, 2:-2], lut_fert[row_no, 4], vs.N_fert1[2:-2, 2:-2]),
            )
            vs.N_fert2 = update(
                vs.N_fert2,
                at[2:-2, 2:-2],
                npx.where(mask[2:-2, 2:-2], lut_fert[row_no, 5], vs.N_fert2[2:-2, 2:-2]),
            )
            vs.N_fert3 = update(
                vs.N_fert3,
                at[2:-2, 2:-2],
                npx.where(mask[2:-2, 2:-2], lut_fert[row_no, 6], vs.N_fert3[2:-2, 2:-2]),
            )

        return KernelOutput(
            doy_fert1=vs.doy_fert1,
            doy_fert2=vs.doy_fert2,
            doy_fert3=vs.doy_fert3,
            N_fert1=vs.N_fert1,
            N_fert2=vs.N_fert2,
            N_fert3=vs.N_fert3,
        )

    @roger_kernel
    def apply_fertilizer_kernel(state):
        vs = state.variables
        settings = state.settings

        # apply nitrogen fertilizer
        vs.Nmin_in = update(vs.Nmin_in, at[2:-2, 2:-2], npx.where((vs.doy_fert1[2:-2, 2:-2] == vs.DOY[vs.itt]), vs.N_fert1[2:-2, 2:-2] * settings.dx * settings.dy * 100, vs.Nmin_in[2:-2, 2:-2]))
        vs.Nmin_in = update(vs.Nmin_in, at[2:-2, 2:-2], npx.where((vs.doy_fert2[2:-2, 2:-2] == vs.DOY[vs.itt]), vs.N_fert2[2:-2, 2:-2] * settings.dx * settings.dy * 100, vs.Nmin_in[2:-2, 2:-2]))
        vs.Nmin_in = update(vs.Nmin_in, at[2:-2, 2:-2], npx.where((vs.doy_fert3[2:-2, 2:-2] == vs.DOY[vs.itt]), vs.N_fert3[2:-2, 2:-2] * settings.dx * settings.dy * 100, vs.Nmin_in[2:-2, 2:-2]))
            
        inf = vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2]
        vs.inf_in_tracer = update(vs.inf_in_tracer, at[2:-2, 2:-2], npx.where((vs.doy_dist[2:-2, 2:-2] == vs.doy_fert1[2:-2, 2:-2]) | (vs.doy_dist[2:-2, 2:-2] == vs.doy_fert2[2:-2, 2:-2]) | (vs.doy_dist[2:-2, 2:-2] == vs.doy_fert3[2:-2, 2:-2]), 0, vs.inf_in_tracer[2:-2, 2:-2]))
        vs.inf_in_tracer = update_add(vs.inf_in_tracer, at[2:-2, 2:-2], inf)
        inf_ratio = npx.where((inf/settings.cum_inf_for_N_input) < 1, inf/settings.cum_inf_for_N_input, 1)
        # dissolved nitrogen input
        vs.M_in = update(vs.M_in, at[2:-2, 2:-2], npx.where(vs.inf_in_tracer[2:-2, 2:-2] > 0, vs.Nmin_in[2:-2, 2:-2] * inf_ratio * vs.c_fert[2:-2, 2:-2], 0))
        # nitrogen deposition (10 kg N/ha/yr)
        Ndep = (10/365) * settings.dx * settings.dy * 100  
        vs.M_in = update_add(vs.M_in, at[2:-2, 2:-2], npx.where(inf > 0, Ndep * vs.c_fert[2:-2, 2:-2], 0))
        vs.Nmin_in = update_add(vs.Nmin_in, at[2:-2, 2:-2], npx.where(inf > 0, Ndep, 0))
        vs.C_in = update(vs.C_in, at[2:-2, 2:-2], npx.where(vs.inf_in_tracer[2:-2, 2:-2] > 0, vs.M_in[2:-2, 2:-2]/inf, 0))
        # undissolved nitrogen input
        vs.Nmin_rz = update_add(
            vs.Nmin_rz,
            at[2:-2, 2:-2, vs.tau, 0],
            npx.where(vs.inf_in_tracer[2:-2, 2:-2] > 0, vs.Nmin_in[2:-2, 2:-2] * inf_ratio * (1 - vs.c_fert[2:-2, 2:-2]), 0),
        )
        vs.Nmin_in = update_add(vs.Nmin_in, at[2:-2, 2:-2], -vs.Nmin_in[2:-2, 2:-2] * inf_ratio)
        vs.Nmin_in = update_add(vs.Nmin_in, at[2:-2, 2:-2], -npx.where(inf > 0, Ndep, 0))
        vs.Nmin_in = update(vs.Nmin_in, at[2:-2, 2:-2], npx.where((vs.Nmin_in[2:-2, 2:-2] < 0), 0, vs.Nmin_in[2:-2, 2:-2]))
        vs.inf_in_tracer = update(vs.inf_in_tracer, at[2:-2, 2:-2], npx.where((vs.inf_in_tracer[2:-2, 2:-2] > settings.cum_inf_for_N_input), 0, vs.inf_in_tracer[2:-2, 2:-2]))

        return KernelOutput(
            Nmin_in=vs.Nmin_in,
            inf_in_tracer=vs.inf_in_tracer,
            M_in=vs.M_in,
            C_in=vs.C_in,
            Nmin_rz=vs.Nmin_rz,
        )


    model = SVATCROPNITRATESetup()
    model.setup()
    model.warmup()
    model.run()
    return


if __name__ == "__main__":
    main()
