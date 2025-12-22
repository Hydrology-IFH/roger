from pathlib import Path
import h5netcdf
import numpy as onp
import yaml
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option("-td", "--tmp-dir", type=str, default=Path(__file__).parent / "output")
@roger_base_cli
def main(tmp_dir):
    from roger import RogerSetup, roger_routine
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at, scipy_stats as sstx

    class SVATCROPNITRATESetup(RogerSetup):
        """A SVAT-CROP transport model for nitrate."""

        _base_path = Path(__file__).parent
        _input_dir = _base_path / "input"
        # load configuration file
        _file_config = _base_path / "config.yml"
        with open(_file_config, "r") as file:
            _config = yaml.safe_load(file)

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

        def _get_runlen(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables["Time"]
                return len(onp.array(var_obj)) * 60 * 60 * 24 - 60 * 60 * 24

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = self._config["identifier"]
            settings.sas_solver = self._config["SAS_SOLVER"]
            settings.sas_solver_substeps = self._config["SAS_SOLVER_SUBSTEPS"]

            # output frequency (in seconds)
            settings.output_frequency = self._config["OUTPUT_FREQUENCY"]
            # total grid numbers in x- and y-direction
            settings.nx, settings.ny = 1, 1
            # derive total number of time steps from forcing
            settings.nitt = self._get_nitt(
                self._input_dir, "SVATCROP.nc"
            )
            settings.nitt_forc = settings.nitt
            # maximum water age (in days)
            settings.ages = 1500
            settings.nages = settings.ages + 1
            # length of warmup (in seconds)
            settings.runlen_warmup = 365 * 24 * 60 * 60
            settings.runlen = self._get_runlen(
                self._input_dir, "SVATCROP.nc"
            )

            # spatial discretization (in meters)
            settings.dx = self._config["dx"]
            settings.dy = self._config["dy"]

            # origin of spatial grid
            settings.x_origin = self._config["x_origin"]
            settings.y_origin = self._config["y_origin"]
            settings.time_origin = "2019-12-31 00:00:00"

            # enable crop phenology
            settings.enable_crop_phenology = True
            settings.enable_crop_rotation = True
            # enable SAS
            settings.enable_offline_transport = True
            settings.enable_age_statistics = True
            # enable soil nitrogen cycle
            settings.enable_nitrate = True

        @roger_routine
        def read_data(self, state):
            pass

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

        @roger_routine
        def set_look_up_tables(self, state):
            pass

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
                "z_soil",
                "phi_soil_temp",
                "damp_soil_temp",
            ],
        )
        def set_parameters_setup(self, state):
            vs = state.variables
            settings = state.settings

            vs.S_PWP_RZ = update(
                vs.S_PWP_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "S_pwp_rz", self._input_dir, "SVATCROP.nc"
                ),
            )
            vs.S_SAT_RZ = update(
                vs.S_SAT_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "S_sat_rz", self._input_dir, "SVATCROP.nc"
                ),
            )
            vs.S_PWP_SS = update(
                vs.S_PWP_SS,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "S_pwp_ss", self._input_dir, "SVATCROP.nc"
                ),
            )
            vs.S_SAT_SS = update(
                vs.S_SAT_SS,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "S_sat_ss", self._input_dir, "SVATCROP.nc"
                ),
            )

            vs.S_pwp_rz = update(
                vs.S_pwp_rz,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "S_pwp_rz", self._input_dir, "SVATCROP.nc"
                )[:, :, 0],
            )
            vs.S_pwp_ss = update(
                vs.S_pwp_ss,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "S_pwp_ss", self._input_dir, "SVATCROP.nc"
                )[:, :, 0],
            )
            vs.S_sat_rz = update(
                vs.S_sat_rz,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "S_sat_rz", self._input_dir, "SVATCROP.nc"
                )[:, :, 0],
            )
            vs.S_sat_ss = update(
                vs.S_sat_ss,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "S_sat_ss", self._input_dir, "SVATCROP.nc"
                )[:, :, 0],
            )

            # partition coefficients
            vs.alpha_transp = update(vs.alpha_transp, at[2:-2, 2:-2], 0.8)
            vs.alpha_q = update(vs.alpha_q, at[2:-2, 2:-2], 0.5)

            # SAS parameters
            # vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
            # vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.25)
            # vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
            # vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.25)
            # vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 62)
            # vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], self._config["C1_TRANSP"])
            # vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 4], self._config["C1_TRANSP"])
            # vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
            # vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
            # vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 61)
            # vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], self._config["C1_Q_RZ"])
            # vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], self._config["C2_Q_RZ"])
            # vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
            # vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
            # vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 61)
            # vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], self._config["C1_Q_SS"])
            # vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], self._config["C2_Q_SS"])
            # vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], vs.S_pwp_ss[2:-2, 2:-2])
            # vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2])
            # vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 6)
            # vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 1], 0.5)
            # vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 6)
            # vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 1], 10)

            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.25)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.25)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 6)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 0.5)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 6)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], 2)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 6)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], 3)
            vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 6)
            vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 1], 0.5)
            vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 6)
            vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 1], 10)

            # denitrification parameters
            vs.km_denit_rz = update(vs.km_denit_rz, at[2:-2, 2:-2], self._config["KM_DENIT_RZ"])
            vs.km_denit_ss = update(vs.km_denit_ss, at[2:-2, 2:-2], self._config["KM_DENIT_SS"])
            vs.dmax_denit_rz = update(vs.dmax_denit_rz, at[2:-2, 2:-2], self._config["DMAX_DENIT_RZ"])
            vs.dmax_denit_ss = update(vs.dmax_denit_ss, at[2:-2, 2:-2], self._config["DMAX_DENIT_SS"])
            # nitrification parameters
            vs.km_nit_rz = update(vs.km_nit_rz, at[2:-2, 2:-2], self._config["KM_NIT_RZ"])
            vs.km_nit_ss = update(vs.km_nit_ss, at[2:-2, 2:-2], self._config["KM_NIT_SS"])
            vs.dmax_nit_rz = update(vs.dmax_nit_rz, at[2:-2, 2:-2], self._config["DMAX_NIT_RZ"])
            vs.dmax_nit_ss = update(vs.dmax_nit_ss, at[2:-2, 2:-2], self._config["DMAX_NIT_SS"])
            # soil nitrogen mineralization parameters
            vs.kmin_rz = update(vs.kmin_rz, at[2:-2, 2:-2], self._config["KMIN_RZ"])
            vs.kmin_ss = update(vs.kmin_ss, at[2:-2, 2:-2], self._config["KMIN_SS"])

            # soil temperature parameters
            vs.z_soil = update(
                vs.z_soil, at[2:-2, 2:-2], self._config["Z_SOIL"]
            )
            vs.phi_soil_temp = update(vs.phi_soil_temp, at[2:-2, 2:-2], 91)
            # dampening depth of soil temperature depends on clay content
            # clay = self._read_var_from_csv("clay", self._base_path, "parameters.csv")
            vs.damp_soil_temp = update(vs.damp_soil_temp, at[2:-2, 2:-2], 12 + 4 * (1 - (self._config["CLAY"] / settings.clay_max)))

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
                    "S_rz", self._input_dir, "SVATCROP.nc"
                )[:, :, vs.itt, npx.newaxis],
            )
            vs.S_ss = update(
                vs.S_ss,
                at[2:-2, 2:-2, :vs.taup1],
                self._read_var_from_nc(
                    "S_ss", self._input_dir, "SVATCROP.nc"
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
            vs.C_rz = update(vs.C_rz, at[2:-2, 2:-2, :vs.taup1], self._config["C_RZ"])
            vs.C_ss = update(vs.C_ss, at[2:-2, 2:-2, :vs.taup1], self._config["C_SS"])
            # exponential distribution of mineral soil nitrogen
            # mineral soil nitrogen is decreasing with increasing age
            p_dec = allocate(state.dimensions, ("x", "y", 2, "ages"))
            p_dec = update(p_dec, at[:, :, :vs.taup1, :], sstx.expon.pdf(npx.linspace(sstx.expon.ppf(0.001), sstx.expon.ppf(0.999), settings.ages))[npx.newaxis, npx.newaxis, npx.newaxis, :])
            vs.Nmin_rz = update(vs.Nmin_rz, at[2:-2, 2:-2, :vs.taup1, :], self._config["NMIN_RZ"] * p_dec[2:-2, 2:-2, :, :] * settings.dx * settings.dy * 100)
            vs.Nmin_ss = update(vs.Nmin_ss, at[2:-2, 2:-2, :vs.taup1, :], self._config["NMIN_SS"] * p_dec[2:-2, 2:-2, :, :] * settings.dx * settings.dy * 100)
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
                    "prec", self._input_dir, "SVATCROP.nc"
                ),
            )
            vs.INF_MAT_RZ = update(
                vs.INF_MAT_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "inf_mat_rz",
                    self._input_dir,
                    "SVATCROP.nc",
                ),
            )
            vs.INF_PF_RZ = update(
                vs.INF_PF_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "inf_mp_rz",
                    self._input_dir,
                    "SVATCROP.nc",
                )
                + self._read_var_from_nc(
                    "inf_sc_rz",
                    self._input_dir,
                    "SVATCROP.nc",
                ),
            )
            vs.INF_PF_SS = update(
                vs.INF_PF_SS,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "inf_ss", self._input_dir, "SVATCROP.nc"
                ),
            )
            vs.TRANSP = update(
                vs.TRANSP,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "transp", self._input_dir, "SVATCROP.nc"
                ),
            )
            vs.EVAP_SOIL = update(
                vs.EVAP_SOIL,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "evap_soil",
                    self._input_dir,
                    "SVATCROP.nc",
                ),
            )
            vs.CPR_RZ = update(
                vs.CPR_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "cpr_rz", self._input_dir, "SVATCROP.nc"
                ),
            )
            vs.Q_RZ = update(
                vs.Q_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "q_rz", self._input_dir, "SVATCROP.nc"
                ),
            )
            vs.Q_SS = update(
                vs.Q_SS,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "q_ss", self._input_dir, "SVATCROP.nc"
                ),
            )
            vs.RE_RG = update(
                vs.RE_RG,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "re_rg", self._input_dir, "SVATCROP.nc"
                ),
            )
            vs.RE_RL = update(
                vs.RE_RL,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "re_rl", self._input_dir, "SVATCROP.nc"
                ),
            )
            vs.S_RZ = update(
                vs.S_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "S_rz", self._input_dir, "SVATCROP.nc"
                ),
            )
            vs.S_SS = update(
                vs.S_SS,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "S_ss", self._input_dir, "SVATCROP.nc"
                ),
            )
            vs.S_S = update(vs.S_S, at[2:-2, 2:-2, :], vs.S_RZ[2:-2, 2:-2, :] + vs.S_SS[2:-2, 2:-2, :])
            vs.TA = update(
                vs.TA,
                at[:],
                self._read_var_from_nc(
                    "ta", self._input_dir, "SVATCROP.nc"
                ),
            )
            vs.YEAR = update(
                vs.YEAR,
                at[:],
                self._read_var_from_nc(
                    "year", self._input_dir, "SVATCROP.nc"
                ),
            )
            vs.DOY = update(
                vs.DOY,
                at[:],
                self._read_var_from_nc(
                    "doy", self._input_dir, "SVATCROP.nc"
                ),
            )
            vs.LU_ID = update(
                vs.LU_ID,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "lu_id", self._input_dir, "SVATCROP.nc"
                ),
            )
            vs.Z_ROOT = update(
                vs.Z_ROOT,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "z_root", self._input_dir, "SVATCROP.nc"
                ),
            )

        @roger_routine
        def set_forcing(self, state):
            vs = state.variables

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

            # apply nitrogen fertilization
            if vs.itt == 70:
                inf = vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2]
                vs.Nmin_in = update(vs.Nmin_in, at[2:-2, 2:-2], 40 * 100)
                vs.M_in = update(vs.M_in, at[2:-2, 2:-2], vs.Nmin_in[2:-2, 2:-2] * 0.3)
                vs.C_in = update(vs.C_in, at[2:-2, 2:-2], vs.M_in[2:-2, 2:-2]/inf)
            else:
                vs.Nmin_in = update(vs.Nmin_in, at[2:-2, 2:-2], 0)
                vs.M_in = update(vs.M_in, at[2:-2, 2:-2], 0)
                vs.C_in = update(vs.C_in, at[2:-2, 2:-2], 0)

        @roger_routine
        def set_diagnostics(self, state, base_path=tmp_dir):
            diagnostics = state.diagnostics

            # variables written to output files
            diagnostics["rate"].output_variables = self._config["OUTPUT_RATE"]
            diagnostics["rate"].output_frequency = self._config["OUTPUT_FREQUENCY"]
            diagnostics["rate"].sampling_frequency = 1
            if base_path:
                diagnostics["rate"].base_output_path = base_path

            diagnostics["collect"].output_variables = self._config["OUTPUT_COLLECT"]
            diagnostics["collect"].output_frequency = self._config["OUTPUT_FREQUENCY"]
            diagnostics["collect"].sampling_frequency = 1
            if base_path:
                diagnostics["collect"].base_output_path = base_path

            diagnostics["average"].output_variables = self._config["OUTPUT_AVERAGE"]
            diagnostics["average"].output_frequency = self._config["OUTPUT_FREQUENCY"]
            diagnostics["average"].sampling_frequency = 1
            if base_path:
                diagnostics["average"].base_output_path = base_path

            diagnostics["maximum"].output_variables = ["dS_num_error", "dC_num_error"]
            diagnostics["maximum"].output_frequency = self._config["OUTPUT_FREQUENCY"]
            diagnostics["maximum"].sampling_frequency = 1
            if base_path:
                diagnostics["maximum"].base_output_path = base_path

        @roger_routine
        def after_timestep(self, state):
            pass

    model = SVATCROPNITRATESetup()
    model.setup()
    model.warmup()
    model.run()
    return


if __name__ == "__main__":
    main()
