from pathlib import Path
import h5netcdf
import numpy as onp
import yaml
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option("-td", "--tmp-dir", type=str, default=None)
@roger_base_cli
def main(tmp_dir):
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at
    from roger.core.transport import delta_to_conc, conc_to_delta

    class SVATTRANSPORTSetup(RogerSetup):
        """A SVAT oxygen18 transport model.
        """
        # custom attributes required by helper functions
        _base_path = Path(__file__).parent
        _input_dir = _base_path / "input"
        # load configuration file
        _file_config = _base_path / "config.yml"
        with open(_file_config, 'r') as file:
            _config = yaml.safe_load(file)
        _tm_structure = _config["TRANSPORT_MODEL_STRUCTURE"].replace("_", " ")

        # custom helper functions
        def _read_var_from_nc(self, var, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables[var]
                return npx.array(var_obj)

        def _get_nitt(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['Time']
                return len(onp.array(var_obj)) + 1

        def _get_runlen(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['Time']
                return len(onp.array(var_obj)) * 60 * 60 * 24

        def _get_time_origin(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                date = infile.variables['Time'].attrs['time_origin'].split(" ")[0]
                return f"{date} 00:00:00"

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = f'SVATOXYGEN18_{self._config["TRANSPORT_MODEL_STRUCTURE"]}_{self._config["SAS_SOLVER"]}'
            settings.sas_solver = self._config["SAS_SOLVER"]
            # number of substeps
            settings.sas_solver_substeps = 6
            if settings.sas_solver in ['RK4', 'Euler']:
                settings.h = 1 / settings.sas_solver_substeps

            # total grid numbers in x- and y-direction
            settings.nx, settings.ny = 12, 24
            # length of simulation (in seconds)
            settings.runlen = self._get_runlen(self._input_dir, 'forcing_tracer.nc')
            # length of warmup simulation (in seconds)
            settings.runlen_warmup = settings.runlen
            # total number of iterations
            settings.nitt = self._get_nitt(self._input_dir, 'forcing_tracer.nc')
            # maximum water age (in days)
            settings.ages = 1500
            settings.nages = settings.ages + 1

            # spatial discretization (in meters)
            settings.dx = 1
            settings.dy = 1

            # origin of spatial grid
            settings.x_origin = 0.0
            settings.y_origin = 0.0
            # origin of time steps (e.g. 01-01-2023)
            settings.time_origin = '31-10-2019'

            settings.enable_offline_transport = True
            settings.enable_oxygen18 = True
            settings.tm_structure = self._tm_structure
            settings.enable_age_statistics = True

            settings.d18O_min = -15.0
            settings.d18O_max = -4.9

        @roger_routine
        def set_grid(self, state):
            vs = state.variables
            settings = state.settings

            # temporal grid
            vs.dt_secs = 60 * 60 * 24
            vs.dt = 60 * 60 * 24 / (60 * 60)
            vs.ages = update(vs.ages, at[:], npx.arange(1, settings.nages))
            vs.nages = update(vs.nages, at[:], npx.arange(settings.nages))
            # spatial grid
            dx = allocate(state.dimensions, ("x"))
            dx = update(dx, at[:], settings.dx)
            dy = allocate(state.dimensions, ("y"))
            dy = update(dy, at[:], settings.dy)
            # distance from origin
            vs.x = update(vs.x, at[3:-2], settings.x_origin + npx.cumsum(dx[3:-2]))
            vs.y = update(vs.y, at[3:-2], settings.y_origin + npx.cumsum(dy[3:-2]))

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
                "S_fc_ss",
                "S_sat_rz",
                "S_sat_ss",
                "sas_params_evap_soil",
                "sas_params_cpr_rz",
                "sas_params_transp",
                "sas_params_q_rz",
                "sas_params_q_ss",
                "itt"
            ],
        )
        def set_parameters_setup(self, state):
            vs = state.variables
            settings = state.settings

            # storage volumes at permanent wilting point and at saturation
            vs.S_pwp_rz = update(vs.S_pwp_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_rz", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.S_sat_rz = update(vs.S_sat_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_rz", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.S_pwp_ss = update(vs.S_pwp_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_ss", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.S_fc_ss = update(vs.S_fc_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_fc_ss", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.S_sat_ss = update(vs.S_sat_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_ss", self._base_path, 'states_hm.nc')[:, :, vs.itt])

            # SAS parameterization
            if settings.tm_structure == "kumaraswamy":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 2], 100)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 2], 100)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 1.5)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 2], 20)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], 3)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 2], 1.5)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], 5)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 2], 1.5)
            elif settings.tm_structure == "power":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 0.3)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], 2)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], 3)

        @roger_routine
        def set_parameters(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "S_pwp_rz",
                "S_pwp_ss",
                "S_snow",
                "S_rz",
                "S_rz_init",
                "S_ss",
                "S_ss_init",
                "S_s",
                "itt",
                "taup1"
            ],
        )
        def set_initial_conditions_setup(self, state):
            vs = state.variables

            vs.S_snow = update(vs.S_snow, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_snow", self._base_path, 'states_hm.nc')[:, :, vs.itt, npx.newaxis])
            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_rz", self._base_path, 'states_hm.nc')[:, :, vs.itt, npx.newaxis] - vs.S_pwp_rz[2:-2, 2:-2, npx.newaxis])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_ss", self._base_path, 'states_hm.nc')[:, :, vs.itt, npx.newaxis] - vs.S_pwp_ss[2:-2, 2:-2, npx.newaxis])
            vs.S_s = update(vs.S_s, at[2:-2, 2:-2, :vs.taup1], vs.S_rz[2:-2, 2:-2, :vs.taup1] + vs.S_ss[2:-2, 2:-2, :vs.taup1])
            vs.S_rz_init = update(vs.S_rz_init, at[2:-2, 2:-2], vs.S_rz[2:-2, 2:-2, 0])
            vs.S_ss_init = update(vs.S_ss_init, at[2:-2, 2:-2], vs.S_ss[2:-2, 2:-2, 0])

        @roger_routine
        def set_initial_conditions(self, state):
            vs = state.variables
            settings = state.settings

            arr0 = allocate(state.dimensions, ("x", "y"))
            vs.sa_rz = update(
                vs.sa_rz,
                at[2:-2, 2:-2, :vs.taup1, 1:], npx.diff(npx.linspace(arr0[2:-2, 2:-2], vs.S_rz[2:-2, 2:-2, vs.tau], settings.ages, axis=-1), axis=-1)[:, :, npx.newaxis, :],
            )
            vs.sa_ss = update(
                vs.sa_ss,
                at[2:-2, 2:-2, :vs.taup1, 1:], npx.diff(npx.linspace(arr0[2:-2, 2:-2], vs.S_ss[2:-2, 2:-2, vs.tau], settings.ages, axis=-1), axis=-1)[:, :, npx.newaxis, :],
            )

            vs.SA_rz = update(
                vs.SA_rz,
                at[2:-2, 2:-2, :, 1:], npx.cumsum(vs.sa_rz[2:-2, 2:-2, :, :], axis=-1),
            )

            vs.SA_ss = update(
                vs.SA_ss,
                at[2:-2, 2:-2, :, 1:], npx.cumsum(vs.sa_rz[2:-2, 2:-2, :, :], axis=-1),
            )

            vs.sa_s = update(
                vs.sa_s,
                at[2:-2, 2:-2, :, :], vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :],
            )
            vs.SA_s = update(
                vs.SA_s,
                at[2:-2, 2:-2, :, 1:], npx.cumsum(vs.sa_s[2:-2, 2:-2, :, :], axis=-1),
            )

            if settings.enable_oxygen18:
                vs.C_iso_snow = update(vs.C_iso_snow, at[2:-2, 2:-2, :vs.taup1], npx.nan)
                vs.C_iso_rz = update(vs.C_iso_rz, at[2:-2, 2:-2, :vs.taup1], self._config["d18O_RZ"])
                vs.C_iso_ss = update(vs.C_iso_ss, at[2:-2, 2:-2, :vs.taup1], self._config["d18O_SS"])
                vs.C_rz = update(
                    vs.C_rz,
                    at[2:-2, 2:-2, :vs.taup1], delta_to_conc(state, vs.C_iso_rz[2:-2, 2:-2, vs.tau, npx.newaxis]),
                )
                vs.msa_rz = update(
                    vs.msa_rz,
                    at[2:-2, 2:-2, :vs.taup1, :], vs.C_rz[2:-2, 2:-2, :vs.taup1, npx.newaxis],
                )
                vs.msa_rz = update(
                    vs.msa_rz,
                    at[2:-2, 2:-2, :vs.taup1, 0], 0,
                )
                vs.C_ss = update(
                    vs.C_ss,
                    at[2:-2, 2:-2, :vs.taup1], delta_to_conc(state, vs.C_iso_ss[2:-2, 2:-2, vs.tau, npx.newaxis]),
                )
                vs.msa_ss = update(
                    vs.msa_ss,
                    at[2:-2, 2:-2, :vs.taup1, :], vs.C_ss[2:-2, 2:-2, :vs.taup1, npx.newaxis],
                )
                vs.msa_ss = update(
                    vs.msa_ss,
                    at[2:-2, 2:-2, :vs.taup1, 0], 0,
                )
                vs.msa_s = update(
                    vs.msa_s,
                    at[2:-2, 2:-2, :, :], npx.where(vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :] > 0, vs.msa_rz[2:-2, 2:-2, :, :] * (vs.sa_rz[2:-2, 2:-2, :, :] / (vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :])) + vs.msa_ss[2:-2, 2:-2, :, :] * (vs.sa_ss[2:-2, 2:-2, :, :] / (vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :])), 0),
                )
                vs.msa_s = update(
                    vs.msa_s,
                    at[2:-2, 2:-2, :vs.taup1, 0], 0,
                )
                vs.C_s = update(
                    vs.C_s,
                    at[2:-2, 2:-2, vs.tau], npx.sum(npx.where(vs.sa_s[2:-2, 2:-2, vs.tau, :] > 0, vs.msa_s[2:-2, 2:-2, vs.tau, :] * (vs.sa_s[2:-2, 2:-2, vs.tau, :] / npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis]), 0), axis=-1),
                )
                vs.C_s = update(
                    vs.C_s,
                    at[2:-2, 2:-2, vs.taum1], vs.C_s[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2],
                )
                vs.C_iso_s = update(
                    vs.C_iso_s,
                    at[2:-2, 2:-2, vs.taum1], conc_to_delta(state, vs.C_s[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
                )
                vs.C_iso_s = update(
                    vs.C_iso_s,
                    at[2:-2, 2:-2, vs.tau], conc_to_delta(state, vs.C_s[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
                )
                vs.csa_rz = update(
                    vs.csa_rz,
                    at[2:-2, 2:-2, vs.tau, :], conc_to_delta(state, vs.msa_rz[2:-2, 2:-2, vs.tau, :]),
                )
                vs.csa_ss = update(
                    vs.csa_ss,
                    at[2:-2, 2:-2, vs.tau, :], conc_to_delta(state, vs.msa_ss[2:-2, 2:-2, vs.tau, :]),
                )
                vs.csa_s = update(
                    vs.csa_s,
                    at[2:-2, 2:-2, vs.tau, :], conc_to_delta(state, vs.msa_s[2:-2, 2:-2, vs.tau, :]),
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
                "S_SNOW",
                "C_ISO_IN",
                "C_IN",
            ],
        )
        def set_forcing_setup(self, state):
            vs = state.variables

            vs.PREC_DIST_DAILY = update(vs.PREC_DIST_DAILY, at[2:-2, 2:-2, :], self._read_var_from_nc("prec", self._base_path, 'states_hm.nc'))
            vs.INF_MAT_RZ = update(vs.INF_MAT_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_mat_rz", self._base_path, 'states_hm.nc'))
            vs.INF_PF_RZ = update(vs.INF_PF_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_mp_rz", self._base_path, 'states_hm.nc') + self._read_var_from_nc("inf_sc_rz", self._base_path, 'states_hm.nc'))
            vs.INF_PF_SS = update(vs.INF_PF_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_ss", self._base_path, 'states_hm.nc'))
            vs.TRANSP = update(vs.TRANSP, at[2:-2, 2:-2, :], self._read_var_from_nc("transp", self._base_path, 'states_hm.nc'))
            vs.EVAP_SOIL = update(vs.EVAP_SOIL, at[2:-2, 2:-2, :], self._read_var_from_nc("evap_soil", self._base_path, 'states_hm.nc'))
            vs.CPR_RZ = update(vs.CPR_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("cpr_rz", self._base_path, 'states_hm.nc'))
            vs.Q_RZ = update(vs.Q_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("q_rz", self._base_path, 'states_hm.nc'))
            vs.Q_SS = update(vs.Q_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("q_ss", self._base_path, 'states_hm.nc'))
            vs.S_RZ = update(vs.S_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_rz", self._base_path, 'states_hm.nc') - self._read_var_from_nc("S_pwp_rz", self._base_path, 'states_hm.nc'))
            vs.S_SS = update(vs.S_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_ss", self._base_path, 'states_hm.nc') - self._read_var_from_nc("S_pwp_ss", self._base_path, 'states_hm.nc'))
            vs.S_S = update(vs.S_S, at[2:-2, 2:-2, :], vs.S_RZ[2:-2, 2:-2, :] + vs.S_SS[2:-2, 2:-2, :])
            vs.S_SNOW = update(vs.S_SNOW, at[2:-2, 2:-2, :], self._read_var_from_nc("S_snow", self._base_path, 'states_hm.nc'))

            vs.C_ISO_IN = update(vs.C_ISO_IN, at[2:-2, 2:-2, 0], npx.nan)
            vs.C_ISO_IN = update(vs.C_ISO_IN, at[2:-2, 2:-2, 1:], self._read_var_from_nc("d18O", self._input_dir, 'forcing_tracer.nc'))
            vs.C_IN = update(vs.C_IN, at[2:-2, 2:-2, :], delta_to_conc(state, vs.C_ISO_IN)[2:-2, 2:-2, :])

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
            vs.S_snow = update(vs.S_snow, at[2:-2, 2:-2, vs.tau], vs.S_SNOW[2:-2, 2:-2, vs.itt])

            vs.C_in = update(vs.C_in, at[2:-2, 2:-2], vs.C_IN[2:-2, 2:-2, vs.itt])
            # mixing of isotopes while snow accumulation
            vs.C_snow = update(
                vs.C_snow,
                at[2:-2, 2:-2, vs.tau], npx.where(vs.S_snow[2:-2, 2:-2, vs.tau] > 0, npx.where(npx.isnan(vs.C_snow[2:-2, 2:-2, vs.tau]), vs.C_in[2:-2, 2:-2], (vs.prec[2:-2, 2:-2, vs.tau] / (vs.prec[2:-2, 2:-2, vs.tau] + vs.S_snow[2:-2, 2:-2, vs.tau])) * vs.C_in[2:-2, 2:-2] + (vs.S_snow[2:-2, 2:-2, vs.tau] / (vs.prec[2:-2, 2:-2, vs.tau] + vs.S_snow[2:-2, 2:-2, vs.tau])) * vs.C_snow[2:-2, 2:-2, vs.taum1]), npx.nan),
            )
            vs.C_snow = update(
                vs.C_snow,
                at[2:-2, 2:-2, vs.tau], npx.where(vs.S_snow[2:-2, 2:-2, vs.tau] <= 0, npx.nan, vs.C_snow[2:-2, 2:-2, vs.tau]),
            )
            vs.C_iso_snow = update(
                vs.C_iso_snow,
                at[2:-2, 2:-2, vs.tau], conc_to_delta(state, vs.C_snow[2:-2, 2:-2, vs.tau]),
            )

            # mix isotopes from snow melt and rainfall
            vs.C_in = update(
                vs.C_in,
                at[2:-2, 2:-2], npx.where(npx.isfinite(vs.C_snow[2:-2, 2:-2, vs.taum1]), vs.C_snow[2:-2, 2:-2, vs.taum1], npx.where(vs.prec[2:-2, 2:-2, vs.tau] > 0, vs.C_IN[2:-2, 2:-2, vs.itt], 0)),
            )
            vs.C_iso_in = update(vs.C_iso_in, at[2:-2, 2:-2], conc_to_delta(state, vs.C_in[2:-2, 2:-2]))

        @roger_routine
        def set_diagnostics(self, state, base_path=tmp_dir):
            diagnostics = state.diagnostics

            diagnostics["average"].output_variables = self._config["OUTPUT_AVERAGE"]
            diagnostics["average"].output_frequency = 24 * 60 * 60
            diagnostics["average"].sampling_frequency = 1
            if base_path:
                diagnostics["average"].base_output_path = base_path

            # maximum bias of numerical solution at time step t
            diagnostics["maximum"].output_variables = ["dS_num_error", "dC_num_error"]
            diagnostics["maximum"].output_frequency = 24 * 60 * 60
            diagnostics["maximum"].sampling_frequency = 1
            if base_path:
                diagnostics["maximum"].base_output_path = base_path

        @roger_routine
        def after_timestep(self, state):
            vs = state.variables

            vs.update(after_timestep_kernel(state))

    @roger_kernel
    def after_timestep_kernel(state):
        vs = state.variables

        vs.S_snow = update(
            vs.S_snow,
            at[2:-2, 2:-2, vs.taum1], vs.S_snow[2:-2, 2:-2, vs.tau],
        )
        vs.C_snow = update(
            vs.C_snow,
            at[2:-2, 2:-2, vs.taum1], vs.C_snow[2:-2, 2:-2, vs.tau],
        )
        vs.prec = update(
            vs.prec,
            at[2:-2, 2:-2, vs.taum1], vs.prec[2:-2, 2:-2, vs.tau],
        )

        return KernelOutput(
            prec=vs.prec,
            C_snow=vs.C_snow,
            S_snow=vs.S_snow,
            )

    # initializes the model structure
    model = SVATTRANSPORTSetup()
    # runs the model setup
    model.setup()
    # runs the model warmup
    model.warmup()
    # iterate over time steps
    model.run()
    return


if __name__ == "__main__":
    main()
