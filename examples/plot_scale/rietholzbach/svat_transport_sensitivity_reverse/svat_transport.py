from pathlib import Path
import glob
import os
import datetime
import h5netcdf

from roger import runtime_settings as rs, runtime_state as rst
rs.backend = "numpy"
rs.force_overwrite = True
if rs.mpi_comm:
    rs.num_proc = (rst.proc_num, 1)
from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, at
from roger.tools.setup import write_forcing_tracer
import numpy as onp


class SVATTRANSPORTSetup(RogerSetup):
    """A SVAT transport model.
    """
    _base_path = Path(__file__).parent
    _bounds = {
        'num_vars': 6,
        'names': ['dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks'],
        'bounds': [[1, 400],
                   [1, 1500],
                   [0.05, 0.33],
                   [0.05, 0.33],
                   [0.05, 0.33],
                   [0.1, 120]]
    }
    _input_dir = None

    def _set_input_dir(self, path):
        if os.path.exists(path):
            self._input_dir = path
        else:
            self._input_dir = path
            if not os.path.exists(self._input_dir):
                os.mkdir(self._input_dir)

    def _read_var_from_nc(self, var, path_dir, file, group=None):
        nc_file = path_dir / file
        if group:
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.groups[group].variables[var]
                return npx.array(var_obj)
        else:
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

    def _get_nx(self, path_dir, file):
        nc_file = path_dir / file
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['x']
            return len(onp.array(var_obj))

    @roger_routine
    def set_settings(self, state):
        settings = state.settings
        settings.identifier = "SVATTRANSPORT"

        settings.nx, settings.ny, settings.nz = self._get_nx(self._base_path, 'states_hm_sensitivity.nc'), 1, 1
        settings.nitt = self._get_nitt(self._base_path, 'states_hm_sensitivity.nc')
        settings.ages = settings.nitt
        settings.nages = settings.nitt + 1
        settings.runlen = self._get_runlen(self._base_path, 'states_hm_sensitivity.nc')

        # lysimeter surface 3.14 square meter (2m diameter)
        settings.dx = 2
        settings.dy = 2
        settings.dz = 1

        settings.x_origin = 0.0
        settings.y_origin = 0.0
        settings.time_origin = "1996-12-31 00:00:00"

        settings.enable_offline_transport = True
        settings.enable_oxygen18 = True
        settings.tm_structure = "complete-mixing"

    @roger_routine(
        dist_safe=False,
        local_variables=[
            "DT_SECS",
            "DT",
            "dt_secs",
            "dt",
            "t",
            "ages",
            "nages",
            "itt",
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
        vs.DT_SECS = update(vs.DT_SECS, at[:], vs.dt_secs)
        vs.DT = update(vs.DT, at[:], vs.dt)
        vs.t = update(vs.t, at[:], npx.cumsum(vs.DT))
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
            "S_PWP_SS",
            "S_SAT_RZ",
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
        ],
    )
    def set_parameters_setup(self, state):
        vs = state.variables
        settings = state.settings

        vs.S_PWP_RZ = update(vs.S_PWP_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_pwp_rz", self._base_path, 'states_hm_sensitivity.nc'))
        vs.S_PWP_SS = update(vs.S_PWP_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_pwp_ss", self._base_path, 'states_hm_sensitivity.nc'))
        vs.S_SAT_RZ = update(vs.S_SAT_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_sat_rz", self._base_path, 'states_hm_sensitivity.nc'))
        vs.S_SAT_SS = update(vs.S_SAT_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_sat_ss", self._base_path, 'states_hm_sensitivity.nc'))

        vs.S_pwp_rz = update(vs.S_pwp_rz, at[2:-2, 2:-2], vs.S_PWP_RZ[2:-2, 2:-2, 0])
        vs.S_pwp_ss = update(vs.S_pwp_ss, at[2:-2, 2:-2], vs.S_PWP_SS[2:-2, 2:-2, 0])
        vs.S_sat_rz = update(vs.S_sat_rz, at[2:-2, 2:-2], vs.S_SAT_RZ[2:-2, 2:-2, 0])
        vs.S_sat_ss = update(vs.S_sat_ss, at[2:-2, 2:-2], vs.S_SAT_SS[2:-2, 2:-2, 0])

        if settings.tm_structure == "complete-mixing":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 1)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 1)
        elif settings.tm_structure == "piston":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 22)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 22)
        elif settings.tm_structure in ["advection-dispersion", "preferential", "time-variant advection-dispersion", "time-variant preferential", "time-variant"]:
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_transp", self._base_path, 'sas_params.nc', group=settings.tm_structure))
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_rz", self._base_path, 'sas_params.nc', group=settings.tm_structure))
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_ss", self._base_path, 'sas_params.nc', group=settings.tm_structure))
        elif settings.tm_structure == "complete-mixing + advection-dispersion":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_rz", self._base_path, 'sas_params.nc', group=settings.tm_structure))
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_ss", self._base_path, 'sas_params.nc', group=settings.tm_structure))

    @roger_routine
    def set_parameters(self, state):
        pass

    @roger_routine(
        dist_safe=False,
        local_variables=[
            "S_SNOW",
            "S_RZ",
            "S_SS",
        ],
    )
    def set_initial_conditions_setup(self, state):
        vs = state.variables

        vs.S_SNOW = update(vs.S_SNOW, at[2:-2, 2:-2, :], self._read_var_from_nc("S_snow", self._base_path, 'states_hm_sensitivity.nc'))
        vs.S_RZ = update(vs.S_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_rz", self._base_path, 'states_hm_sensitivity.nc'))
        vs.S_SS = update(vs.S_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_ss", self._base_path, 'states_hm_sensitivity.nc'))

    @roger_routine
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings

        vs.S_S = update(vs.S_S, at[2:-2, 2:-2, :], vs.S_RZ[2:-2, 2:-2, :] + vs.S_SS[2:-2, 2:-2, :])

        vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, :vs.taup1], vs.S_RZ[2:-2, 2:-2, 0, npx.newaxis] - vs.S_pwp_rz[2:-2, 2:-2, npx.newaxis])
        vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, :vs.taup1], vs.S_SS[2:-2, 2:-2, 0, npx.newaxis] - vs.S_pwp_ss[2:-2, 2:-2, npx.newaxis])
        vs.S_s = update(vs.S_s, at[2:-2, 2:-2, :vs.taup1], vs.S_S[2:-2, 2:-2, 0, npx.newaxis] - (vs.S_pwp_rz[2:-2, 2:-2, npx.newaxis] + vs.S_pwp_ss[2:-2, 2:-2, npx.newaxis]))

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

        if (settings.enable_oxygen18 | settings.enable_deuterium):
            vs.C_snow = update(vs.C_snow, at[2:-2, 2:-2, :vs.taup1], npx.NaN)
            vs.C_rz = update(vs.C_rz, at[2:-2, 2:-2, :vs.taup1], -13)
            vs.C_ss = update(vs.C_ss, at[2:-2, 2:-2, :vs.taup1], -7)
            vs.msa_rz = update(
                vs.msa_rz,
                at[2:-2, 2:-2, :vs.taup1, :], vs.C_rz[2:-2, 2:-2, :vs.taup1, npx.newaxis],
            )
            vs.msa_rz = update(
                vs.msa_rz,
                at[2:-2, 2:-2, :vs.taup1, 0], npx.NaN,
            )
            vs.msa_ss = update(
                vs.msa_ss,
                at[2:-2, 2:-2, :vs.taup1, :], vs.C_ss[2:-2, 2:-2, :vs.taup1, npx.newaxis],
            )
            vs.msa_ss = update(
                vs.msa_ss,
                at[2:-2, 2:-2, :vs.taup1, 0], npx.NaN,
            )
            iso_rz = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
            iso_ss = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
            iso_rz = update(
                iso_rz,
                at[2:-2, 2:-2, :, :], npx.where(npx.isnan(vs.msa_rz), 0, vs.msa_rz)[2:-2, 2:-2, :, :],
            )
            iso_ss = update(
                iso_ss,
                at[2:-2, 2:-2, :, :], npx.where(npx.isnan(vs.msa_ss), 0, vs.msa_ss)[2:-2, 2:-2, :, :],
            )
            vs.msa_s = update(
                vs.msa_s,
                at[2:-2, 2:-2, :, :], (vs.sa_rz[2:-2, 2:-2, :, :] / vs.sa_s[2:-2, 2:-2, :, :]) * iso_rz[2:-2, 2:-2, :, :] + (vs.sa_ss[2:-2, 2:-2, :, :] / vs.sa_s[2:-2, 2:-2, :, :]) * iso_ss[2:-2, 2:-2, :, :],
            )

            vs.C_s = update(
                vs.C_s,
                at[2:-2, 2:-2, vs.tau], calc_conc_iso_storage(state, vs.sa_s, vs.msa_s)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
            )

            vs.C_s = update(
                vs.C_s,
                at[2:-2, 2:-2, vs.taum1], vs.C_s[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2],
            )

    @roger_routine(
        dist_safe=False,
        local_variables=[
            "PREC",
            "TA",
            "INF_MAT_RZ",
            "INF_PF_RZ",
            "INF_PF_SS",
            "TRANSP",
            "EVAP_SOIL",
            "CPR_RZ",
            "Q_RZ",
            "Q_SS",
            "C_IN",
        ],
    )
    def set_forcing_setup(self, state):
        vs = state.variables
        settings = state.settings

        vs.TA = update(vs.TA, at[2:-2, 2:-2, :], self._read_var_from_nc("ta", self._base_path, 'states_hm_sensitivity.nc'))
        vs.PREC = update(vs.PREC, at[2:-2, 2:-2, :], self._read_var_from_nc("prec", self._base_path, 'states_hm_sensitivity.nc'))
        vs.INF_MAT_RZ = update(vs.INF_MAT_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_mat_rz", self._base_path, 'states_hm_sensitivity.nc'))
        vs.INF_PF_RZ = update(vs.INF_PF_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_mp_rz", self._base_path, 'states_hm_sensitivity.nc') + self._read_var_from_nc("inf_sc_rz", self._base_path, 'states_hm_sensitivity.nc'))
        vs.INF_PF_SS = update(vs.INF_PF_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_ss", self._base_path, 'states_hm_sensitivity.nc'))
        vs.TRANSP = update(vs.TRANSP, at[2:-2, 2:-2, :], self._read_var_from_nc("transp", self._base_path, 'states_hm_sensitivity.nc'))
        vs.EVAP_SOIL = update(vs.EVAP_SOIL, at[2:-2, 2:-2, :], self._read_var_from_nc("evap_soil", self._base_path, 'states_hm_sensitivity.nc'))
        vs.CPR_RZ = update(vs.CPR_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("cpr_rz", self._base_path, 'states_hm_sensitivity.nc'))
        vs.Q_RZ = update(vs.Q_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("q_rz", self._base_path, 'states_hm_sensitivity.nc'))
        vs.Q_SS = update(vs.Q_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("q_ss", self._base_path, 'states_hm_sensitivity.nc'))

        if settings.enable_deuterium:
            vs.C_IN = update(vs.C_IN, at[2:-2, 2:-2, 0], npx.NaN)
            vs.C_IN = update(vs.C_IN, at[2:-2, 2:-2, 1:], self._read_var_from_nc("d2H", self._input_dir, 'forcing_tracer.nc'))

        if settings.enable_oxygen18:
            vs.C_IN = update(vs.C_IN, at[2:-2, 2:-2, 0], npx.NaN)
            vs.C_IN = update(vs.C_IN, at[2:-2, 2:-2, 1:], self._read_var_from_nc("d18O", self._input_dir, 'forcing_tracer.nc'))

        if settings.enable_deuterium or settings.enable_oxygen18:
            vs.update(set_iso_input_kernel(state))

    @roger_routine
    def set_forcing(self, state):
        vs = state.variables
        settings = state.settings

        vs.update(set_states_kernel(state))

        if settings.enable_deuterium or settings.enable_oxygen18:
            vs.update(set_forcing_iso_kernel(state))

    @roger_routine
    def set_diagnostics(self, state):
        diagnostics = state.diagnostics

        diagnostics["averages"].output_variables = ["C_q_ss"]
        diagnostics["averages"].output_frequency = 24 * 60 * 60
        diagnostics["averages"].sampling_frequency = 1

    @roger_routine
    def after_timestep(self, state):
        vs = state.variables

        vs.update(after_timestep_kernel(state))


@roger_kernel
def set_iso_input_kernel(state):
    vs = state.variables

    vs.C_IN = update(vs.C_IN, at[2:-2, 2:-2, :], _ffill_3d(state, vs.C_IN)[2:-2, 2:-2, :])

    return KernelOutput(
        C_IN=vs.C_IN,
    )


@roger_kernel
def set_forcing_iso_kernel(state):
    vs = state.variables

    # mixing of isotopes while snow accumulation
    vs.C_snow = update(
        vs.C_snow,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.S_SNOW[2:-2, 2:-2, vs.itt] > 0, npx.where(npx.isnan(vs.C_snow[2:-2, 2:-2, vs.tau]), vs.C_IN[2:-2, 2:-2, vs.itt], (vs.PREC[2:-2, 2:-2, vs.itt] / (vs.PREC[2:-2, 2:-2, vs.itt] + vs.S_SNOW[2:-2, 2:-2, vs.itt])) * vs.C_IN[2:-2, 2:-2, vs.itt] + (vs.S_SNOW[2:-2, 2:-2, vs.itt] / (vs.PREC[2:-2, 2:-2, vs.itt] + vs.S_SNOW[2:-2, 2:-2, vs.itt])) * vs.C_snow[2:-2, 2:-2, vs.taum1]), npx.NaN) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_snow = update(
        vs.C_snow,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.S_SNOW[2:-2, 2:-2, vs.itt] <= 0, npx.NaN, vs.C_snow[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    # mix isotopes from snow melt and rainfall
    vs.C_in = update(
        vs.C_in,
        at[2:-2, 2:-2], npx.where(npx.isfinite(vs.C_snow[2:-2, 2:-2, vs.taum1]), vs.C_snow[2:-2, 2:-2, vs.taum1], npx.where(vs.PREC[2:-2, 2:-2, vs.itt] > 0, vs.C_IN[2:-2, 2:-2, vs.itt], npx.NaN)) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.M_in = update(
        vs.M_in,
        at[2:-2, 2:-2], vs.C_in[2:-2, 2:-2] * vs.PREC[2:-2, 2:-2, vs.itt] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(
        M_in=vs.M_in,
        C_in=vs.C_in,
        C_snow=vs.C_snow,
    )


@roger_kernel
def set_states_kernel(state):
    vs = state.variables

    vs.inf_mat_rz = update(vs.inf_mat_rz, at[2:-2, 2:-2], vs.INF_MAT_RZ[2:-2, 2:-2, vs.itt])
    vs.inf_pf_rz = update(vs.inf_pf_rz, at[2:-2, 2:-2], vs.INF_PF_RZ[2:-2, 2:-2, vs.itt])
    vs.inf_pf_ss = update(vs.inf_pf_ss, at[2:-2, 2:-2], vs.INF_PF_SS[2:-2, 2:-2, vs.itt])
    vs.transp = update(vs.transp, at[2:-2, 2:-2], vs.TRANSP[2:-2, 2:-2, vs.itt])
    vs.evap_soil = update(vs.evap_soil, at[2:-2, 2:-2], vs.EVAP_SOIL[2:-2, 2:-2, vs.itt])
    vs.q_rz = update(vs.q_rz, at[2:-2, 2:-2], vs.Q_RZ[2:-2, 2:-2, vs.itt])
    vs.q_ss = update(vs.q_ss, at[2:-2, 2:-2], vs.Q_SS[2:-2, 2:-2, vs.itt])
    vs.cpr_rz = update(vs.cpr_ss, at[2:-2, 2:-2], vs.CPR_RZ[2:-2, 2:-2, vs.itt])

    vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, vs.tau], vs.S_RZ[2:-2, 2:-2, vs.itt])
    vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, vs.tau], vs.S_SS[2:-2, 2:-2, vs.itt])
    vs.S_s = update(vs.S_s, at[2:-2, 2:-2, vs.tau], vs.S_S[2:-2, 2:-2, vs.itt])
    vs.S_snow = update(vs.S_snow, at[2:-2, 2:-2, vs.tau], vs.S_SNOW[2:-2, 2:-2, vs.itt])

    return KernelOutput(
        inf_mat_rz=vs.inf_mat_rz,
        inf_pf_rz=vs.inf_pf_rz,
        inf_pf_ss=vs.inf_pf_ss,
        transp=vs.transp,
        evap_soil=vs.evap_soil,
        q_rz=vs.q_rz,
        q_ss=vs.q_ss,
        cpr_rz=vs.cpr_rz,
        S_rz=vs.S_rz,
        S_ss=vs.S_ss,
        S_snow=vs.S_snow,
    )


@roger_kernel
def after_timestep_kernel(state):
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, vs.taum1, :], vs.SA_rz[2:-2, 2:-2, vs.tau, :],
    )
    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, vs.taum1, :], vs.sa_rz[2:-2, 2:-2, vs.tau, :],
    )
    vs.MSA_rz = update(
        vs.MSA_rz,
        at[2:-2, 2:-2, vs.taum1, :], vs.MSA_rz[2:-2, 2:-2, vs.tau, :],
    )
    vs.msa_rz = update(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.taum1, :], vs.msa_rz[2:-2, 2:-2, vs.tau, :],
    )
    vs.M_rz = update(
        vs.M_rz,
        at[2:-2, 2:-2, vs.taum1], vs.M_rz[2:-2, 2:-2, vs.tau],
    )
    vs.C_rz = update(
        vs.C_rz,
        at[2:-2, 2:-2, vs.taum1], vs.C_rz[2:-2, 2:-2, vs.tau],
    )
    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, vs.taum1, :], vs.SA_ss[2:-2, 2:-2, vs.tau, :],
    )
    vs.sa_ss = update(
        vs.sa_ss,
        at[2:-2, 2:-2, vs.taum1, :], vs.sa_ss[2:-2, 2:-2, vs.tau, :],
    )
    vs.MSA_ss = update(
        vs.MSA_ss,
        at[2:-2, 2:-2, vs.taum1, :], vs.MSA_ss[2:-2, 2:-2, vs.tau, :],
    )
    vs.msa_ss = update(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.taum1, :], vs.msa_ss[2:-2, 2:-2, vs.tau, :],
    )
    vs.M_ss = update(
        vs.M_ss,
        at[2:-2, 2:-2, vs.taum1], vs.M_ss[2:-2, 2:-2, vs.tau],
    )
    vs.C_ss = update(
        vs.C_ss,
        at[2:-2, 2:-2, vs.taum1], vs.C_ss[2:-2, 2:-2, vs.tau],
    )
    vs.SA_s = update(
        vs.SA_s,
        at[2:-2, 2:-2, vs.taum1, :], vs.SA_s[2:-2, 2:-2, vs.tau, :],
    )
    vs.sa_s = update(
        vs.sa_s,
        at[2:-2, 2:-2, vs.taum1, :], vs.sa_s[2:-2, 2:-2, vs.tau, :],
    )
    vs.MSA_s = update(
        vs.MSA_s,
        at[2:-2, 2:-2, vs.taum1, :], vs.MSA_s[2:-2, 2:-2, vs.tau, :],
    )
    vs.msa_s = update(
        vs.msa_s,
        at[2:-2, 2:-2, vs.taum1, :], vs.msa_s[2:-2, 2:-2, vs.tau, :],
    )
    vs.M_s = update(
        vs.M_s,
        at[2:-2, 2:-2, vs.taum1], vs.M_s[2:-2, 2:-2, vs.tau],
    )
    vs.C_s = update(
        vs.C_s,
        at[2:-2, 2:-2, vs.taum1], vs.C_s[2:-2, 2:-2, vs.tau],
    )
    vs.C_snow = update(
        vs.C_snow,
        at[2:-2, 2:-2, vs.taum1], vs.C_snow[2:-2, 2:-2, vs.tau],
    )

    return KernelOutput(
        SA_rz=vs.SA_rz,
        sa_rz=vs.sa_rz,
        MSA_rz=vs.MSA_rz,
        msa_rz=vs.msa_rz,
        M_rz=vs.M_rz,
        C_rz=vs.C_rz,
        SA_ss=vs.SA_ss,
        sa_ss=vs.sa_ss,
        MSA_ss=vs.MSA_ss,
        msa_ss=vs.msa_ss,
        M_ss=vs.M_ss,
        C_ss=vs.C_ss,
        SA_s=vs.SA_s,
        sa_s=vs.sa_s,
        MSA_s=vs.MSA_s,
        msa_s=vs.msa_s,
        M_s=vs.M_s,
        C_s=vs.C_s,
        C_snow=vs.C_snow,
        )


@roger_kernel
def calc_conc_iso_storage(state, sa, msa):
    """Calculates isotope signal of storage.
    """
    vs = state.variables

    mask = npx.isfinite(msa[:, :, vs.tau, :])
    vals = allocate(state.dimensions, ("x", "y", "ages"))
    weights = allocate(state.dimensions, ("x", "y", "ages"))
    vals = update(
        vals,
        at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2, :], msa[2:-2, 2:-2, vs.tau, :], 0),
    )
    weights = update(
        weights,
        at[2:-2, 2:-2, :], npx.where(sa[2:-2, 2:-2, vs.tau, :] * mask[2:-2, 2:-2, :] > 0, sa[2:-2, 2:-2, vs.tau, :] / npx.sum(sa[2:-2, 2:-2, vs.tau, :] * mask[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis], 0),
    )
    conc = allocate(state.dimensions, ("x", "y"))
    # calculate weighted average
    conc = update(
        conc,
        at[2:-2, 2:-2], npx.sum(vals[2:-2, 2:-2, :] * weights[2:-2, 2:-2, :], axis=-1),
    )
    conc = update(
        conc,
        at[2:-2, 2:-2], npx.where(conc[2:-2, 2:-2] != 0, conc[2:-2, 2:-2], npx.NaN),
    )

    return conc


@roger_kernel
def _ffill_3d(state, arr):
    idx_shape = tuple([slice(None)] + [npx.newaxis] * (3 - 2 - 1))
    idx = allocate(state.dimensions, ("x", "y", "t"), dtype=int)
    arr1 = allocate(state.dimensions, ("x", 1, 1), dtype=int)
    arr2 = allocate(state.dimensions, (1, "y", 1), dtype=int)
    arr3 = allocate(state.dimensions, ("x", "y", "t"), dtype=int)
    arr_fill = allocate(state.dimensions, ("x", "y", "t"))
    idx = update(
        idx,
        at[:, :, :], npx.where(npx.isfinite(arr), npx.arange(npx.shape(arr)[2])[idx_shape], 0),
    )
    idx = update(
        idx,
        at[:, :, :], npx.maximum.accumulate(idx, axis=-1),
    )
    arr1 = update(
        arr1,
        at[:, 0, 0], npx.arange(npx.shape(arr)[0]),
    )
    arr2 = update(
        arr2,
        at[0, :, 0], npx.arange(npx.shape(arr)[1]),
    )
    arr3 = update(
        arr3,
        at[:, :, :], idx,
    )
    arr_fill = update(
        arr_fill,
        at[:, :, :], arr[arr1, arr2, arr3],
    )

    return arr_fill


tm_structures = ['complete-mixing', 'piston',
                 'preferential', 'advection-dispersion',
                 'time-variant preferential',
                 'time-variant advection-dispersion']
for tm_structure in tm_structures:
    tms = tm_structure.replace(" ", "_")
    model = SVATTRANSPORTSetup()
    model._set_tm_structure(tm_structure)
    input_path = model._base_path / "input"
    model._set_input_dir(input_path)
    forcing_path = model._input_dir / "forcing_tracer.nc"
    if not os.path.exists(forcing_path):
        write_forcing_tracer(input_path, 'd18O')
    model.setup()
    model.warmup()
    model.run()

    # merge model output into single file
    if rst.proc_rank == 0 or not rs.mpi_comm:
        path = str(model._base_path / f"{model.state.settings.identifier}.*.nc")
        diag_files = glob.glob(path)
        states_tm_file = model._base_path / "states_tm_sensitivity_reverse.nc"
        with h5netcdf.File(states_tm_file, 'a', decode_vlen_strings=False) as f:
            if tm_structure not in list(f.groups.keys()):
                f.create_group(tm_structure)
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title=f'RoGeR {tm_structure} transport model saltelli results (reverse) at Rietholzbach Lysimeter site',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment=f'SVAT {tm_structure} transport model with free drainage'
            )
            for dfs in diag_files:
                with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                    # set dimensions with a dictionary
                    dict_dim = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'Time': len(df.variables['Time']), 'ages': len(df.variables['ages']), 'nages': len(df.variables['nages'])}
                    if not f.groups[tm_structure].dimensions:
                        f.groups[tm_structure].dimensions = dict_dim
                        v = f.groups[tm_structure].create_variable('x', ('x',), float)
                        v.attrs['long_name'] = 'model run'
                        v.attrs['units'] = ''
                        v[:] = npx.arange(dict_dim["x"])
                        v = f.groups[tm_structure].create_variable('y', ('y',), float)
                        v.attrs['long_name'] = ''
                        v.attrs['units'] = ''
                        v[:] = npx.arange(dict_dim["y"])
                        v = f.groups[tm_structure].create_variable('Time', ('Time',), float)
                        var_obj = df.variables.get('Time')
                        v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                        units=var_obj.attrs["units"])
                        v[:] = npx.array(var_obj)
                        v = f.groups[tm_structure].create_variable('ages', ('ages',), float)
                        v.attrs['long_name'] = 'Water ages'
                        v.attrs['units'] = 'days'
                        v[:] = npx.arange(1, dict_dim["ages"]+1)
                        v = f.groups[tm_structure].create_variable('nages', ('nages',), float)
                        v.attrs['long_name'] = 'Water ages (cumulated)'
                        v.attrs['units'] = 'days'
                        v[:] = npx.arange(0, dict_dim["nages"])
                    for var_sim in list(df.variables.keys()):
                        var_obj = df.variables.get(var_sim)
                        if var_sim not in list(dict_dim.keys()) and ('Time', 'y', 'x') == var_obj.dimensions:
                            v = f.groups[tm_structure].create_variable(var_sim, ('x', 'y', 'Time'), float)
                            vals = npx.array(var_obj)
                            v[:, :, :] = vals.swapaxes(0, 2)
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
                        elif var_sim not in list(dict_dim.keys()) and ('Time', 'n_sas_params', 'y', 'x') == var_obj.dimensions:
                            v = f.groups[tm_structure].create_variable(var_sim, ('x', 'y', 'n_sas_params'), float)
                            vals = npx.array(var_obj)
                            vals = vals.swapaxes(0, 3)
                            vals = vals.swapaxes(1, 2)
                            v[:, :, :] = vals[:, :, :, 0]
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
                        elif var_sim not in list(dict_dim.keys()) and ('Time', 'ages', 'y', 'x') == var_obj.dimensions:
                            v = f.groups[tm_structure].create_variable(var_sim, ('x', 'y', 'Time', 'ages'), float)
                            vals = npx.array(var_obj)
                            vals = vals.swapaxes(0, 3)
                            vals = vals.swapaxes(1, 2)
                            vals = vals.swapaxes(2, 3)
                            v[:, :, :, :] = vals
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
                        elif var_sim not in list(dict_dim.keys()) and ('Time', 'nages', 'y', 'x') == var_obj.dimensions:
                            v = f.groups[tm_structure].create_variable(var_sim, ('x', 'y', 'Time', 'nages'), float)
                            vals = npx.array(var_obj)
                            vals = vals.swapaxes(0, 3)
                            vals = vals.swapaxes(1, 2)
                            vals = vals.swapaxes(2, 3)
                            v[:, :, :, :] = vals
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
