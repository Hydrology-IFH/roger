from pathlib import Path
import h5netcdf
import numpy as onp
from roger import RogerSetup, roger_routine
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, at, where


class SVATTRANSPORTSetup(RogerSetup):
    """A SVAT bromide transport model.
    """
    # custom attributes required by helper functions
    _base_path = Path(__file__).parent
    _tm_structure = "complete-mixing"
    _input_dir = _base_path / "input"
    _identifier = "SVATBROMIDE_complete-mixing"
    _sas_solver = "deterministic"

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

    def _set_year(self, year):
        self._year = year

    def _set_tm_structure(self, tm_structure):
        self._tm_structure = tm_structure

    def _set_sas_solver(self, sas_solver):
        self._sas_solver = sas_solver

    def _set_identifier(self, identifier):
        self._identifier = identifier

    def _set_bromide_input(self, state, nn_rain, nn_sol, prec, ta):
        vs = state.variables

        M_IN = allocate(state.dimensions, ("x", "y", "t"))

        mask_rain = (prec > 0) & (ta > 0)
        mask_sol = (vs.M_IN > 0)
        sol_idx = npx.zeros((nn_sol,), dtype=int)
        sol_idx = update(sol_idx, at[:], where(npx.any(mask_sol, axis=(0, 1)), size=nn_sol, fill_value=0)[0])
        rain_idx = npx.zeros((nn_rain,), dtype=int)
        rain_idx = update(rain_idx, at[:], where(npx.any(mask_rain, axis=(0, 1)), size=nn_rain, fill_value=0)[0])
        end_rain = npx.zeros((1,), dtype=int)

        # join solute input on closest rainfall event
        for i in range(nn_sol):
            rain_sum = allocate(state.dimensions, ("x", "y"))
            nn_end = allocate(state.dimensions, ("x", "y"))
            input_itt = npx.nanargmin(npx.where(rain_idx - sol_idx[i] < 0, npx.nan, rain_idx - sol_idx[i]))
            start_rain = rain_idx[input_itt]
            rain_sum = update(
                rain_sum,
                at[:, :], npx.max(npx.where(npx.cumsum(prec[:, :, start_rain:], axis=-1) <= 20, npx.max(npx.cumsum(prec[:, :, start_rain:], axis=-1), axis=-1)[:, :, npx.newaxis], 0), axis=-1),
            )
            nn_end = npx.max(npx.where(npx.cumsum(prec[:, :, start_rain:]) <= 20, npx.max(npx.arange(npx.shape(prec)[2])[npx.newaxis, npx.newaxis, npx.shape(prec)[2]-start_rain], axis=-1), 0))
            end_rain = update(end_rain, at[:], start_rain + nn_end)
            end_rain = update(end_rain, at[:], npx.where(end_rain > npx.shape(prec)[2], npx.shape(prec)[2], end_rain))

            # proportions for redistribution
            M_IN = update(
                M_IN,
                at[:, :, start_rain:end_rain[0]], vs.M_IN[:, :, sol_idx[i], npx.newaxis] * (prec[:, :, start_rain:end_rain[0]] / rain_sum[:, :, npx.newaxis]),
            )

        C_IN = npx.where(prec > 0, M_IN / prec, 0)

        return M_IN, C_IN

    @roger_routine
    def set_settings(self, state):
        settings = state.settings
        settings.identifier = self._identifier
        # set the solver schemes
        settings.sas_solver = self._sas_solver
        # number of substepss
        settings.sas_solver_substeps = 6
        if settings.sas_solver in ['RK4', 'Euler']:
            # time increment of substep (in days)
            settings.h = 1 / settings.sas_solver_substeps

        # total grid numbers in x- and y-direction
        settings.nx, settings.ny = 1, 1
        # number of iterations (i.e. number of days)
        settings.nitt = self._get_nitt(self._input_dir, 'forcing_tracer.nc')
        # maximum water age (in days)
        settings.ages = settings.nitt
        settings.nages = settings.nitt + 1
        # length of simulation (in seconds)
        settings.runlen = self._get_runlen(self._input_dir, 'forcing_tracer.nc')

        # spatial discretization (in meters)
        settings.dx = 1
        settings.dy = 1

        # origin of spatial grid
        settings.x_origin = 0.0
        settings.y_origin = 0.0
        # origin of time steps (e.g. 01-01-2023)
        settings.time_origin = self._get_time_origin(self._input_dir, 'forcing_tracer.nc')

        # enable transport
        settings.enable_offline_transport = True
        # enable bromide
        settings.enable_bromide = True
        # set model structure
        settings.tm_structure = self._tm_structure
        # enable calculation of age statistics
        settings.enable_age_statistics = True

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
            "S_sat_rz",
            "S_sat_ss",
            "alpha_transp",
            "alpha_q",
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

        vs.S_pwp_rz = update(vs.S_pwp_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_rz", self._input_dir, 'states_hm.nc')[:, :, vs.itt])
        vs.S_pwp_ss = update(vs.S_pwp_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_ss", self._input_dir, 'states_hm.nc')[:, :, vs.itt])
        vs.S_sat_rz = update(vs.S_sat_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_rz", self._input_dir, 'states_hm.nc')[:, :, vs.itt])
        vs.S_sat_ss = update(vs.S_sat_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_ss", self._input_dir, 'states_hm.nc')[:, :, vs.itt])

        # partition coefficient of transpiration
        vs.alpha_transp = update(vs.alpha_transp, at[2:-2, 2:-2], 0.5)
        # partition coefficient of percolation
        vs.alpha_q = update(vs.alpha_q, at[2:-2, 2:-2], 0.5)

        # SAS parameterization
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

        vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_rz", self._input_dir, 'states_hm.nc')[:, :, vs.itt] - vs.S_pwp_rz[2:-2, 2:-2, npx.newaxis])
        vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_ss", self._input_dir, 'states_hm.nc')[:, :, vs.itt] - vs.S_pwp_ss[2:-2, 2:-2, npx.newaxis])
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

    @roger_routine
    def set_boundary_conditions_setup(self, state):
        pass

    @roger_routine
    def set_boundary_conditions(self, state):
        pass

    @roger_routine(
        dist_safe=False,
        local_variables=[
            "M_IN",
            "C_IN",
        ],
    )
    def set_forcing_setup(self, state):
        vs = state.variables

        TA = self._read_var_from_nc("ta", self._input_dir, 'states_hm.nc')
        PREC = self._read_var_from_nc("prec", self._input_dir, 'states_hm.nc')

        vs.M_IN = update(vs.M_IN, at[2:-2, 2:-2, 1:], self._read_var_from_nc("Br", self._input_dir, 'forcing_tracer.nc'))

        mask_rain = (PREC > 0) & (TA > 0)
        mask_sol = (vs.M_IN > 0)
        nn_rain = npx.int64(npx.sum(npx.any(mask_rain, axis=(0, 1))))
        nn_sol = npx.int64(npx.sum(npx.any(mask_sol, axis=(0, 1))))
        M_IN, C_IN = self._set_bromide_input(state, nn_rain, nn_sol, PREC, TA)
        vs.M_IN = update(vs.M_IN, at[:, :, :], M_IN)
        vs.C_IN = update(vs.C_IN, at[:, :, :], C_IN)

    @roger_routine(
        dist_safe=False,
        local_variables=[
            "ta",
            "prec",
            "inf_mat_rz",
            "inf_pf_rz",
            "inf_pf_ss",
            "transp",
            "evap_soil",
            "cpr_rz",
            "q_rz",
            "q_ss",
            "S_pwp_rz",
            "S_rz",
            "S_pwp_ss",
            "S_ss",
            "S_s",
            "S_snow",
            "tau",
            "taum1",
            "itt",
            "C_in",
            "C_IN",
            "M_in"

        ],
    )
    def set_forcing(self, state):
        vs = state.variables

        vs.ta = update(vs.ta, at[2:-2, 2:-2], self._read_var_from_nc("ta", self._input_dir, 'states_hm.nc')[:, :, vs.itt])
        vs.prec = update(vs.prec, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("prec", self._input_dir, 'states_hm.nc')[:, :, vs.itt])
        vs.inf_mat_rz = update(vs.inf_mat_rz, at[2:-2, 2:-2], self._read_var_from_nc("inf_mat_rz", self._input_dir, 'states_hm.nc')[:, :, vs.itt])
        vs.inf_pf_rz = update(vs.inf_pf_rz, at[2:-2, 2:-2], self._read_var_from_nc("inf_mp_rz", self._input_dir, 'states_hm.nc')[:, :, vs.itt] + self._read_var_from_nc("inf_sc_rz", self._input_dir, 'states_hm.nc')[:, :, vs.itt])
        vs.inf_pf_ss = update(vs.inf_pf_ss, at[2:-2, 2:-2], self._read_var_from_nc("inf_ss", self._input_dir, 'states_hm.nc')[:, :, vs.itt])
        vs.transp = update(vs.transp, at[2:-2, 2:-2], self._read_var_from_nc("transp", self._input_dir, 'states_hm.nc')[:, :, vs.itt])
        vs.evap_soil = update(vs.evap_soil, at[2:-2, 2:-2], self._read_var_from_nc("evap_soil", self._input_dir, 'states_hm.nc')[:, :, vs.itt])
        vs.cpr_rz = update(vs.cpr_rz, at[2:-2, 2:-2], self._read_var_from_nc("cpr_rz", self._input_dir, 'states_hm.nc')[:, :, vs.itt])
        vs.q_rz = update(vs.q_rz, at[2:-2, 2:-2], self._read_var_from_nc("q_rz", self._input_dir, 'states_hm.nc')[:, :, vs.itt])
        vs.q_ss = update(vs.q_ss, at[2:-2, 2:-2], self._read_var_from_nc("q_ss", self._input_dir, 'states_hm.nc')[:, :, vs.itt])

        vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("S_rz", self._input_dir, 'states_hm.nc')[:, :, vs.itt] - vs.S_pwp_rz[2:-2, 2:-2, npx.newaxis])
        vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("S_ss", self._input_dir, 'states_hm.nc')[:, :, vs.itt] - vs.S_pwp_ss[2:-2, 2:-2, npx.newaxis])
        vs.S_s = update(vs.S_s, at[2:-2, 2:-2, vs.tau], vs.S_rz[2:-2, 2:-2, vs.tau] + vs.S_ss[2:-2, 2:-2, vs.tau])

        vs.C_in = update(vs.C_in, at[2:-2, 2:-2], vs.C_IN[2:-2, 2:-2, vs.itt])
        vs.M_in = update(
            vs.M_in,
            at[2:-2, 2:-2], vs.C_in[2:-2, 2:-2] * vs.prec[2:-2, 2:-2, vs.tau],
        )

    @roger_routine
    def set_diagnostics(self, state):
        pass

    @roger_routine
    def after_timestep(self, state):
        pass
