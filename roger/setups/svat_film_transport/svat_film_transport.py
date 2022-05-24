from pathlib import Path
import os
import h5netcdf
from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at, for_loop, where
from roger.core.utilities import _get_row_no
import roger.lookuptables as lut
import numpy as onp


class SVATFILMTRANSPORTSetup(RogerSetup):
    """A SVAT transport model including including gravity-driven infiltration
    and percolation.
    """
    _base_path = Path(__file__).parent
    _input_dir = None

    def _set_input_dir(self, path):
        if os.path.exists(path):
            self._input_dir = path
        else:
            self._input_dir = path
            if not os.path.exists(self._input_dir):
                os.mkdir(self._input_dir)

    def _read_var_from_nc(self, var, path_dir, file):
        nc_file = path_dir / file
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables[var]
            return npx.array(var_obj)

    def _get_nitt(self, path_dir, file):
        nc_file = path_dir / file
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['Time']
            return len(onp.array(var_obj))

    def _get_runlen(self, path_dir, file):
        nc_file = path_dir / file
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['Time']
            return onp.array(var_obj)[-1] * 60 * 60

    @roger_routine
    def set_settings(self, state):
        settings = state.settings
        settings.identifier = "SVATFILMTRANSPORT"

        settings.nx, settings.ny, settings.nz = 1, 1, 1
        settings.nitt = self._get_nitt(self._base_path, 'states_hm.nc')
        settings.ages = settings.nitt
        settings.nages = settings.nitt + 1
        settings.runlen = self._get_runlen(self._base_path, 'states_hm.nc')

        settings.dx = 1
        settings.dy = 1
        settings.dz = 1

        settings.x_origin = 0.0
        settings.y_origin = 0.0

        settings.enable_offline_transport = True
        settings.enable_film_flow = True
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
        # spatial grid
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

    @roger_routine
    def set_parameters_setup(self, state):
        vs = state.variables
        settings = state.settings

        vs.S_PWP_RZ = update(vs.S_PWP_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_pwp_rz", self._base_path / 'states_hm.nc'))
        vs.S_PWP_SS = update(vs.S_PWP_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_pwp_ss", self._base_path / 'states_hm.nc'))
        vs.S_SAT_RZ = update(vs.S_SAT_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_sat_rz", self._base_path / 'states_hm.nc'))
        vs.S_SAT_SS = update(vs.S_SAT_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_sat_ss", self._base_path / 'states_hm.nc'))

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
        elif settings.tm_structure == "preferential":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 2], 30)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 2], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 2], 2)
        elif settings.tm_structure == "advection-dispersion":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 2], 30)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], 2)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 2], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 2], 1)
        elif settings.tm_structure == "complete-mixing + advection-dispersion":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], 2)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 2], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 2], 1)
        elif settings.tm_structure == "time-variant advection-dispersion":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 31)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 4], 30)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 32)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], 2)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 32)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2])
        elif settings.tm_structure == "time-variant preferential":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 31)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 4], 30)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 31)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], 3)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 31)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], 2)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2])
        elif settings.tm_structure == "time-variant":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 35)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 4], 30)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 35)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], 3)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 35)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], 2)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2])

    @roger_routine
    def set_parameters(self, state):
        pass

    @roger_routine
    def set_initial_conditions_setup(self, state):
        vs = state.variables

        vs.S_RZ = update(vs.S_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_rz", self._base_path / 'states_hm.nc'))
        vs.S_SS = update(vs.S_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_ss", self._base_path / 'states_hm.nc'))

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

    @roger_routine
    def set_forcing_setup(self, state):
        vs = state.variables

        vs.TA = update(vs.TA, at[2:-2, 2:-2, :], self._read_var_from_nc("ta", self._base_path / 'states_hm.nc'))
        vs.PREC = update(vs.PREC, at[2:-2, 2:-2, :], self._read_var_from_nc("prec", self._base_path / 'states_hm.nc'))
        vs.INF_MAT_RZ = update(vs.INF_MAT_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_mat_rz", self._base_path / 'states_hm.nc'))
        vs.INF_PF_RZ = update(vs.INF_PF_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_mp_rz", self._base_path / 'states_hm.nc') + self._read_var_from_nc("inf_sc_rz", self._base_path / 'states_hm.nc'))
        vs.INF_PF_SS = update(vs.INF_PF_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_ss", self._base_path / 'states_hm.nc'))
        vs.TRANSP = update(vs.TRANSP, at[2:-2, 2:-2, :], self._read_var_from_nc("transp", self._base_path / 'states_hm.nc'))
        vs.EVAP_SOIL = update(vs.EVAP_SOIL, at[2:-2, 2:-2, :], self._read_var_from_nc("evap_soil", self._base_path / 'states_hm.nc'))
        vs.Q_RZ = update(vs.Q_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("q_rz", self._base_path / 'states_hm.nc'))
        vs.Q_SS = update(vs.Q_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("q_ss", self._base_path / 'states_hm.nc'))

    @roger_routine
    def set_forcing(self, state):
        vs = state.variables

        vs.update(set_states_kernel(state))
        vs.update(set_forcing_kernel(state))

    @roger_routine
    def set_diagnostics(self, state):
        pass

    @roger_routine
    def after_timestep(self, state):
        vs = state.variables

        vs.update(after_timestep_kernel(state))


@roger_kernel
def set_forcing_kernel(state):
    vs = state.variables

    inf = vs.inf_mat_rz + vs.inf_mat_rz + vs.inf_pf_rz
    vs.M_in = update(
        vs.M_in,
        at[2:-2, 2:-2], inf[2:-2, 2:-2],
    )
    vs.C_in = update(
        vs.C_in,
        at[2:-2, 2:-2], npx.where(inf > 0, 1, 0)[2:-2, 2:-2],
    )

    return KernelOutput(
        M_in=vs.M_in,
        C_in=vs.C_in,
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
        )
