from pathlib import Path
import h5netcdf
import numpy as onp
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option("--location", type=click.Choice(["freiburg", "altheim", "kupferzell"]), default="freiburg")
@click.option("--land-cover-scenario", type=click.Choice(["grass"]), default="grass")
@click.option(
    "--climate-scenario",
    type=click.Choice(["observed", "CCCma-CanESM2_CCLM4-8-17", "MPI-M-MPI-ESM-LR_RCA4"]),
    default="MPI-M-MPI-ESM-LR_RCA4",
)
@click.option("--period", type=click.Choice(["1985-2014", "2030-2059", "2070-2099"]), default="2030-2059")
@click.option("-td", "--tmp-dir", type=str, default=None)
@roger_base_cli
def main(location, land_cover_scenario, climate_scenario, period, tmp_dir):
    from roger import RogerSetup, roger_routine
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at

    class SVATTRANSPORTSetup(RogerSetup):
        """A SVAT transport model for virtual tracer."""

        _base_path = Path(__file__).parent
        if tmp_dir:
            # read fluxes and states from local SSD on cluster node
            _input_dir = Path(tmp_dir)
        else:
            _input_dir = _base_path.parent / "output" / "svat"

        def _read_var_from_nc(self, var, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables[var]
                return npx.array(var_obj)

        def _get_nitt(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables["Time"]
                return len(onp.array(var_obj))

        def _get_runlen(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables["Time"]
                return len(onp.array(var_obj)) * 60 * 60 * 24

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = f"SVATTRANSPORT_{location}_grass_{climate_scenario}_{period}"
            settings.sas_solver = "deterministic"
            settings.sas_solver_substeps = 8

            settings.nx, settings.ny = 676, 1
            settings.nitt = self._get_nitt(self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc")
            settings.ages = 1000
            settings.nages = settings.ages + 1
            settings.runlen_warmup = 2 * 365 * 24 * 60 * 60
            settings.runlen = self._get_runlen(self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc")

            settings.dx = 1
            settings.dy = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0
            year0 = int(period.split("-")[0]) - 1
            settings.time_origin = f"{year0}-12-31 00:00:00"

            settings.enable_offline_transport = True
            settings.enable_virtualtracer = True
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

        @roger_routine
        def set_look_up_tables(self, state):
            pass

        @roger_routine
        def set_topography(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "S_pwp_rz",
                "S_pwp_ss",
                "S_sat_rz",
                "S_sat_ss",
                "alpha_transp",
                "alpha_q",
                "sas_params_evap_soil",
                "sas_params_cpr_rz",
                "sas_params_transp",
                "sas_params_q_rz",
                "sas_params_q_ss",
            ],
        )
        def set_parameters_setup(self, state):
            vs = state.variables

            vs.S_pwp_rz = update(
                vs.S_pwp_rz,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "S_pwp_rz", self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
                )[:, :, 0],
            )
            vs.S_pwp_ss = update(
                vs.S_pwp_ss,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "S_pwp_ss", self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
                )[:, :, 0],
            )
            vs.S_sat_rz = update(
                vs.S_sat_rz,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "S_sat_rz", self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
                )[:, :, 0],
            )
            vs.S_sat_ss = update(
                vs.S_sat_ss,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "S_sat_ss", self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
                )[:, :, 0],
            )

            # partition coefficients
            vs.alpha_transp = update(vs.alpha_transp, at[2:-2, 2:-2], 1)
            vs.alpha_q = update(vs.alpha_q, at[2:-2, 2:-2], 1)

            # SAS parameters
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.2)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.2)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 62)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], 0.3)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 4], 0.6)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 61)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], 1.5)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], 1.5)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 61)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], 1.5)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], 1.5)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], vs.S_pwp_ss[2:-2, 2:-2])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2])

        @roger_routine
        def set_parameters(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=["S_rz", "S_rz_init", "S_ss", "S_ss_init", "S_s", "itt", "taup1"],
        )
        def set_initial_conditions_setup(self, state):
            vs = state.variables

            vs.S_rz = update(
                vs.S_rz,
                at[2:-2, 2:-2, : vs.taup1],
                self._read_var_from_nc(
                    "S_rz", self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
                )[:, :, vs.itt, npx.newaxis],
            )
            vs.S_ss = update(
                vs.S_ss,
                at[2:-2, 2:-2, : vs.taup1],
                self._read_var_from_nc(
                    "S_ss", self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
                )[:, :, vs.itt, npx.newaxis],
            )
            vs.S_s = update(
                vs.S_s, at[2:-2, 2:-2, : vs.taup1], vs.S_rz[2:-2, 2:-2, : vs.taup1] + vs.S_ss[2:-2, 2:-2, : vs.taup1]
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
                at[2:-2, 2:-2, : vs.taup1, 1:],
                npx.diff(npx.linspace(arr0[2:-2, 2:-2], vs.S_rz[2:-2, 2:-2, vs.tau], settings.ages, axis=-1), axis=-1)[
                    :, :, npx.newaxis, :
                ],
            )
            vs.sa_ss = update(
                vs.sa_ss,
                at[2:-2, 2:-2, : vs.taup1, 1:],
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
                "S_RZ",
                "S_SS",
                "S_S",
            ],
        )
        def set_forcing_setup(self, state):
            vs = state.variables

            vs.PREC_DIST_DAILY = update(
                vs.PREC_DIST_DAILY,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "prec", self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
                ),
            )
            vs.INF_MAT_RZ = update(
                vs.INF_MAT_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "inf_mat_rz", self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
                ),
            )
            vs.INF_PF_RZ = update(
                vs.INF_PF_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "inf_mp_rz", self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
                )
                + self._read_var_from_nc(
                    "inf_sc_rz", self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
                ),
            )
            vs.INF_PF_SS = update(
                vs.INF_PF_SS,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "inf_ss", self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
                ),
            )
            vs.TRANSP = update(
                vs.TRANSP,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "transp", self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
                ),
            )
            vs.EVAP_SOIL = update(
                vs.EVAP_SOIL,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "evap_soil", self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
                ),
            )
            vs.CPR_RZ = update(
                vs.CPR_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "cpr_rz", self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
                ),
            )
            vs.Q_RZ = update(
                vs.Q_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "q_rz", self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
                ),
            )
            vs.Q_SS = update(
                vs.Q_SS,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "q_ss", self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
                ),
            )
            vs.S_RZ = update(
                vs.S_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "S_rz", self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
                ),
            )
            vs.S_SS = update(
                vs.S_SS,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc(
                    "S_ss", self._input_dir, f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
                ),
            )
            vs.S_S = update(vs.S_S, at[2:-2, 2:-2, :], vs.S_RZ[2:-2, 2:-2, :] + vs.S_SS[2:-2, 2:-2, :])

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
            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, vs.tau], vs.S_RZ[2:-2, 2:-2, vs.itt])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, vs.tau], vs.S_SS[2:-2, 2:-2, vs.itt])
            vs.S_s = update(vs.S_s, at[2:-2, 2:-2, vs.tau], vs.S_rz[2:-2, 2:-2, vs.tau] + vs.S_ss[2:-2, 2:-2, vs.tau])

            # apply virtual tracer
            inf = vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2]
            vs.C_in = update(vs.C_in, at[2:-2, 2:-2], npx.where(inf > 0, 1, 0))
            vs.M_in = update(vs.M_in, at[2:-2, 2:-2], npx.where(inf > 0, vs.C_in[2:-2, 2:-2] * inf, 0))

        @roger_routine
        def set_diagnostics(self, state, base_path=tmp_dir):
            diagnostics = state.diagnostics

            diagnostics["rate"].output_variables = ["M_in", "M_q_ss", "M_transp", "M_evap_soil"]
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

            diagnostics["collect"].output_variables = ["M_s"]
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
            pass

    model = SVATTRANSPORTSetup()
    model.setup()
    model.warmup()
    model.run()
    return


if __name__ == "__main__":
    main()
