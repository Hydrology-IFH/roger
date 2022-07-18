from pathlib import Path
import os
import h5netcdf
import numpy as onp
from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at
import roger.lookuptables as lut


class SVATFILMEVENTSetup(RogerSetup):
    """A SVAT model including gravity-driven infiltration and percolation.
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

    @roger_routine
    def set_settings(self, state):
        settings = state.settings
        settings.identifier = "SVATFILMEVENT"

        settings.nx, settings.ny, settings.nz = 1, 1, 1
        settings.nitt = self._get_nitt(self._input_dir, 'forcing.nc')
        settings.runlen = settings.nitt * 10 * 60
        settings.nittevent_ff_p1 = settings.nittevent_ff + 1
        settings.nevent_ff = 1

        settings.dx = 1
        settings.dy = 1
        settings.dz = 1

        settings.x_origin = 0.0
        settings.y_origin = 0.0

        settings.enable_film_flow = True
        settings.enable_macropore_lower_boundary_condition = False

        settings.ff_tc = 0.15

    @roger_routine(
        dist_safe=False,
        local_variables=[
            "DT_SECS",
            "DT",
            "dt_secs",
            "dt",
            "t",
            "itt",
            "x",
            "y",
        ],
    )
    def set_grid(self, state):
        vs = state.variables
        settings = state.settings

        # temporal grid
        vs.DT_SECS = update(vs.DT_SECS, at[:], self._read_var_from_nc("dt", self._input_dir, 'forcing.nc'))
        vs.DT = update(vs.DT, at[:], vs.DT_SECS / (60 * 60))
        vs.dt_secs = vs.DT_SECS[vs.itt]
        vs.dt = vs.DT[vs.itt]
        vs.t = update(vs.t, at[:], npx.linspace(0, vs.dt * settings.nitt, num=settings.nitt))
        # spatial grid
        dx = allocate(state.dimensions, ("x"))
        dx = update(dx, at[:], 1)
        dy = allocate(state.dimensions, ("y"))
        dy = update(dy, at[:], 1)
        vs.x = update(vs.x, at[3:-2], npx.cumsum(dx[3:-2]))
        vs.y = update(vs.y, at[3:-2], npx.cumsum(dy[3:-2]))

    @roger_routine
    def set_look_up_tables(self, state):
        vs = state.variables

        vs.lut_ilu = update(vs.lut_ilu, at[:, :], lut.ARR_ILU)
        vs.lut_gc = update(vs.lut_gc, at[:, :], lut.ARR_GC)
        vs.lut_gcm = update(vs.lut_gcm, at[:, :], lut.ARR_GCM)
        vs.lut_is = update(vs.lut_is, at[:, :], lut.ARR_IS)
        vs.lut_rdlu = update(vs.lut_rdlu, at[:, :], lut.ARR_RDLU)

    @roger_routine
    def set_topography(self, state):
        pass

    @roger_routine
    def set_parameters_setup(self, state):
        vs = state.variables

        vs.lu_id = update(vs.lu_id, at[2:-2, 2:-2], 8)
        vs.slope = update(vs.slope, at[2:-2, 2:-2], 0)
        vs.z_soil = update(vs.z_soil, at[2:-2, 2:-2], 1350)
        vs.dmpv = update(vs.dmpv, at[2:-2, 2:-2], 50)
        vs.lmpv = update(vs.lmpv, at[2:-2, 2:-2], 1000)
        vs.theta_ac = update(vs.theta_ac, at[2:-2, 2:-2], 0.13)
        vs.theta_ufc = update(vs.theta_ufc, at[2:-2, 2:-2], 0.24)
        vs.theta_pwp = update(vs.theta_pwp, at[2:-2, 2:-2], 0.23)
        vs.ks = update(vs.ks, at[2:-2, 2:-2], 25)
        vs.kf = update(vs.kf, at[2:-2, 2:-2], 2500)
        vs.a_ff = update(vs.a_ff, at[2:-2, 2:-2], 0.19)
        vs.c_ff = update(vs.c_ff, at[2:-2, 2:-2], 0.001)

    @roger_routine
    def set_parameters(self, state):
        pass

    @roger_routine
    def set_initial_conditions_setup(self, state):
        pass

    @roger_routine
    def set_initial_conditions(self, state):
        vs = state.variables

        vs.S_int_top = update(vs.S_int_top, at[2:-2, 2:-2, :vs.taup1], 0)
        vs.swe_top = update(vs.swe_top, at[2:-2, 2:-2, :vs.taup1], 0)
        vs.S_int_ground = update(vs.S_int_ground, at[2:-2, 2:-2, :vs.taup1], 0)
        vs.swe_ground = update(vs.swe_ground, at[2:-2, 2:-2, :vs.taup1], 0)
        vs.S_dep = update(vs.S_dep, at[2:-2, 2:-2, :vs.taup1], 0)
        vs.S_snow = update(vs.S_snow, at[2:-2, 2:-2, :vs.taup1], 0)
        vs.swe = update(vs.swe, at[2:-2, 2:-2, :vs.taup1], 0)
        vs.theta_rz = update(vs.theta_rz, at[2:-2, 2:-2, :vs.taup1], 0.4)
        vs.theta_ss = update(vs.theta_ss, at[2:-2, 2:-2, :vs.taup1], 0.47)

    @roger_routine
    def set_forcing_setup(self, state):
        vs = state.variables

        vs.PREC = update(vs.PREC, at[2:-2, 2:-2, :], self._read_var_from_nc("PREC", self._input_dir, 'forcing.nc'))
        vs.TA = update(vs.TA, at[2:-2, 2:-2, :], self._read_var_from_nc("TA", self._input_dir, 'forcing.nc'))
        vs.PET = update(vs.PET, at[2:-2, 2:-2, :], self._read_var_from_nc("PET", self._input_dir, 'forcing.nc'))
        vs.EVENT_ID = update(vs.EVENT_ID, at[2:-2, 2:-2, :], 1)
        vs.EVENT_ID_FF = update(vs.EVENT_ID_FF, at[2:-2, 2:-2, :], 1)

    @roger_routine
    def set_forcing(self, state):
        vs = state.variables
        settings = state.settings

        vs.itt_event_ff = update(
            vs.itt_event_ff,
            at[:], vs.itt - vs.event_start_ff,
        )
        arr_itt = allocate(state.dimensions, ("x", "y"))
        arr_itt = update(
            arr_itt,
            at[2:-2, 2:-2], vs.itt,
        )
        cond1 = ((vs.EVENT_ID_FF[2:-2, 2:-2, vs.itt-1] == 0) & (vs.EVENT_ID_FF[2:-2, 2:-2, vs.itt] >= 1) & (arr_itt >= 1))
        if cond1.any():
            vs.event_no_ff = vs.event_no_ff + 1
            # number of event
            vs.event_id_ff = npx.max(vs.EVENT_ID_FF[2:-2, 2:-2, vs.itt])
            # iteration at event start
            vs.event_start_ff = update(vs.event_start_ff, at[vs.event_no_ff - 1], vs.itt)
            # iteration at event end
            vs.event_end_ff = update(vs.event_end_ff, at[vs.event_no_ff - 1], vs.itt + settings.nittevent_ff)
            if (vs.event_end_ff[vs.event_no_ff - 1] >= settings.nitt):
                vs.event_end_ff = update(vs.event_end_ff, at[vs.event_no_ff - 1], settings.nitt)
            vs.rain_event = update(
                vs.rain_event,
                at[2:-2, 2:-2, :], 0,
            )
            vs.rain_event = update(
                vs.rain_event,
                at[2:-2, 2:-2, 0:vs.event_end_ff[vs.event_no_ff - 1]-vs.event_start_ff[vs.event_no_ff - 1]], npx.where(vs.EVENT_ID_FF == vs.event_id_ff, vs.PREC, 0)[2:-2, 2:-2, vs.event_start_ff[vs.event_no_ff - 1]:vs.event_end_ff[vs.event_no_ff - 1]],
            )
            vs.rain_event_sum = update(
                vs.rain_event_sum,
                at[2:-2, 2:-2], npx.sum(vs.rain_event[2:-2, 2:-2, :], axis=-1),
            )
            vs.rain_event_csum = update(
                vs.rain_event_csum,
                at[2:-2, 2:-2, :], npx.cumsum(vs.rain_event[2:-2, 2:-2, :], axis=-1),
            )
            # subtract interception storage
            vs.rain_event = update(
                vs.rain_event,
                at[2:-2, 2:-2, 0], vs.rain_event[2:-2, 2:-2, 0] - (vs.S_int_top_tot[2:-2, 2:-2] - vs.S_int_top[2:-2, 2:-2, vs.tau]) + (vs.S_int_ground_tot[2:-2, 2:-2] - vs.S_int_ground[2:-2, 2:-2, vs.tau]),
            )
            vs.rain_event = update(
                vs.rain_event,
                at[2:-2, 2:-2, 1:], npx.diff(npx.where(npx.cumsum(vs.rain_event[2:-2, 2:-2, :], axis=-1) < 0, 0, npx.cumsum(vs.rain_event[2:-2, 2:-2, :], axis=-1)), axis=-1),
            )
            vs.rain_event = update(
                vs.rain_event,
                at[2:-2, 2:-2, 0], npx.where(vs.rain_event[2:-2, 2:-2, 0] < 0, 0, vs.rain_event[2:-2, 2:-2, 0]),
            )

        vs.update(set_forcing_kernel(state))

    @roger_routine
    def set_diagnostics(self, state):
        pass

    @roger_routine
    def after_timestep(self, state):
        vs = state.variables

        vs.update(after_timestep_kernel(state))
        vs.update(after_timestep_film_flow_kernel(state))


@roger_kernel
def set_forcing_kernel(state):
    vs = state.variables

    # update precipitation with available interception storage while film flow event
    mask_noff = (vs.EVENT_ID_FF[:, :, vs.itt] == 0)
    vs.prec = update(vs.prec, at[2:-2, 2:-2], 0)
    vs.prec = update_add(vs.prec, at[2:-2, 2:-2],
                         npx.where((vs.EVENT_ID_FF[2:-2, 2:-2, vs.itt] >= 1) & ((vs.S_int_top_tot[2:-2, 2:-2] - vs.S_int_top[2:-2, 2:-2, vs.tau]) + (vs.S_int_ground_tot[2:-2, 2:-2] - vs.S_int_ground[2:-2, 2:-2, vs.tau]) > 0),
                         npx.where(vs.PREC[2:-2, 2:-2, vs.itt] > (vs.S_int_top_tot[2:-2, 2:-2] - vs.S_int_top[2:-2, 2:-2, vs.tau]) + (vs.S_int_ground_tot[2:-2, 2:-2] - vs.S_int_ground[2:-2, 2:-2, vs.tau]), (vs.S_int_top_tot[2:-2, 2:-2] - vs.S_int_top[2:-2, 2:-2, vs.tau]) + (vs.S_int_ground_tot[2:-2, 2:-2] - vs.S_int_ground[2:-2, 2:-2, vs.tau]), vs.PREC[2:-2, 2:-2, vs.itt]), 0))
    vs.prec = update(vs.prec, at[2:-2, 2:-2], npx.where(mask_noff[2:-2, 2:-2], vs.PREC[2:-2, 2:-2, vs.itt], 0))
    vs.ta = update(vs.ta, at[2:-2, 2:-2, vs.tau], vs.TA[2:-2, 2:-2, vs.itt])
    vs.pet = update(vs.pet, at[2:-2, 2:-2], vs.PET[2:-2, 2:-2, vs.itt])
    vs.pet_res = update(vs.pet, at[2:-2, 2:-2], vs.PET[2:-2, 2:-2, vs.itt])

    vs.dt_secs = vs.DT_SECS[vs.itt]
    vs.dt = vs.DT[vs.itt]
    vs.year = vs.YEAR[vs.itt]
    vs.month = vs.MONTH[vs.itt]
    vs.doy = vs.DOY[vs.itt]

    # reset fluxes at beginning of time step
    vs.q_sur = update(
        vs.q_sur,
        at[2:-2, 2:-2], 0,
    )

    vs.q_hof = update(
        vs.q_hof,
        at[2:-2, 2:-2], 0,
    )

    return KernelOutput(
        prec=vs.prec,
        ta=vs.ta,
        pet=vs.pet,
        pet_res=vs.pet_res,
        dt=vs.dt,
        dt_secs=vs.dt_secs,
        year=vs.year,
        month=vs.month,
        doy=vs.doy,
        q_sur=vs.q_sur,
        q_hof=vs.q_hof,
    )


@roger_kernel
def after_timestep_kernel(state):
    vs = state.variables

    vs.ta = update(
        vs.ta,
        at[2:-2, 2:-2, vs.taum1], vs.ta[2:-2, 2:-2, vs.tau],
    )
    vs.z_root = update(
        vs.z_root,
        at[2:-2, 2:-2, vs.taum1], vs.z_root[2:-2, 2:-2, vs.tau],
    )
    vs.ground_cover = update(
        vs.ground_cover,
        at[2:-2, 2:-2, vs.taum1], vs.ground_cover[2:-2, 2:-2, vs.tau],
    )
    vs.S_sur = update(
        vs.S_sur,
        at[2:-2, 2:-2, vs.taum1], vs.S_sur[2:-2, 2:-2, vs.tau],
    )
    vs.S_int_top = update(
        vs.S_int_top,
        at[2:-2, 2:-2, vs.taum1], vs.S_int_top[2:-2, 2:-2, vs.tau],
    )
    vs.S_int_ground = update(
        vs.S_int_ground,
        at[2:-2, 2:-2, vs.taum1], vs.S_int_ground[2:-2, 2:-2, vs.tau],
    )
    vs.S_dep = update(
        vs.S_dep,
        at[2:-2, 2:-2, vs.taum1], vs.S_dep[2:-2, 2:-2, vs.tau],
    )
    vs.S_snow = update(
        vs.S_snow,
        at[2:-2, 2:-2, vs.taum1], vs.S_snow[2:-2, 2:-2, vs.tau],
    )
    vs.swe = update(
        vs.swe,
        at[2:-2, 2:-2, vs.taum1], vs.swe[2:-2, 2:-2, vs.tau],
    )
    vs.S_rz = update(
        vs.S_rz,
        at[2:-2, 2:-2, vs.taum1], vs.S_rz[2:-2, 2:-2, vs.tau],
    )
    vs.S_ss = update(
        vs.S_ss,
        at[2:-2, 2:-2, vs.taum1], vs.S_ss[2:-2, 2:-2, vs.tau],
    )
    vs.S_s = update(
        vs.S_s,
        at[2:-2, 2:-2, vs.taum1], vs.S_s[2:-2, 2:-2, vs.tau],
    )
    vs.S = update(
        vs.S,
        at[2:-2, 2:-2, vs.taum1], vs.S[2:-2, 2:-2, vs.tau],
    )
    vs.z_sat = update(
        vs.z_sat,
        at[2:-2, 2:-2, vs.taum1], vs.z_sat[2:-2, 2:-2, vs.tau],
    )
    vs.z_wf = update(
        vs.z_wf,
        at[2:-2, 2:-2, vs.taum1], vs.z_wf[2:-2, 2:-2, vs.tau],
    )
    vs.z_wf_t0 = update(
        vs.z_wf_t0,
        at[2:-2, 2:-2, vs.taum1], vs.z_wf_t0[2:-2, 2:-2, vs.tau],
    )
    vs.z_wf_t1 = update(
        vs.z_wf_t1,
        at[2:-2, 2:-2, vs.taum1], vs.z_wf_t1[2:-2, 2:-2, vs.tau],
    )
    vs.y_mp = update(
        vs.y_mp,
        at[2:-2, 2:-2, vs.taum1], vs.y_mp[2:-2, 2:-2, vs.tau],
    )
    vs.y_sc = update(
        vs.y_sc,
        at[2:-2, 2:-2, vs.taum1], vs.y_sc[2:-2, 2:-2, vs.tau],
    )
    vs.prec_event_sum = update(
        vs.prec_event_sum,
        at[2:-2, 2:-2, vs.taum1], vs.prec_event_sum[2:-2, 2:-2, vs.tau],
    )
    vs.t_event_sum = update(
        vs.t_event_sum,
        at[2:-2, 2:-2, vs.taum1], vs.t_event_sum[2:-2, 2:-2, vs.tau],
    )
    vs.theta_rz = update(
        vs.theta_rz,
        at[2:-2, 2:-2, vs.taum1], vs.theta_rz[2:-2, 2:-2, vs.tau],
    )
    vs.theta_ss = update(
        vs.theta_ss,
        at[2:-2, 2:-2, vs.taum1], vs.theta_ss[2:-2, 2:-2, vs.tau],
    )
    vs.theta = update(
        vs.theta,
        at[2:-2, 2:-2, vs.taum1], vs.theta[2:-2, 2:-2, vs.tau],
    )
    vs.k_rz = update(
        vs.k_rz,
        at[2:-2, 2:-2, vs.taum1], vs.k_rz[2:-2, 2:-2, vs.tau],
    )
    vs.k_ss = update(
        vs.k_ss,
        at[2:-2, 2:-2, vs.taum1], vs.k_ss[2:-2, 2:-2, vs.tau],
    )
    vs.k = update(
        vs.k,
        at[2:-2, 2:-2, vs.taum1], vs.k[2:-2, 2:-2, vs.tau],
    )
    vs.h_rz = update(
        vs.h_rz,
        at[2:-2, 2:-2, vs.taum1], vs.h_rz[2:-2, 2:-2, vs.tau],
    )
    vs.h_ss = update(
        vs.h_ss,
        at[2:-2, 2:-2, vs.taum1], vs.h_ss[2:-2, 2:-2, vs.tau],
    )
    vs.h = update(
        vs.h,
        at[2:-2, 2:-2, vs.taum1], vs.h[2:-2, 2:-2, vs.tau],
    )
    vs.z0 = update(
        vs.z0,
        at[2:-2, 2:-2, vs.taum1], vs.z0[2:-2, 2:-2, vs.tau],
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
        t_event_sum=vs.t_event_sum,
        prec_event_sum=vs.prec_event_sum,
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
    )


@roger_kernel
def after_timestep_film_flow_kernel(state):
    vs = state.variables

    vs.z_wf_ff = update(
        vs.z_wf_ff,
        at[2:-2, 2:-2, :, vs.taum1], vs.z_wf_ff[2:-2, 2:-2, :, vs.tau],
    )
    vs.z_pf_ff = update(
        vs.z_pf_ff,
        at[2:-2, 2:-2, :, vs.taum1], vs.z_pf_ff[2:-2, 2:-2, :, vs.tau],
    )
    vs.theta_rz_ff = update(
        vs.theta_rz_ff,
        at[2:-2, 2:-2, vs.taum1], vs.theta_rz_ff[2:-2, 2:-2, vs.tau],
    )
    vs.theta_ss_ff = update(
        vs.theta_ss_ff,
        at[2:-2, 2:-2, vs.taum1], vs.theta_ss_ff[2:-2, 2:-2, vs.tau],
    )
    vs.theta_ff = update(
        vs.theta_ff,
        at[2:-2, 2:-2, vs.taum1], vs.theta_ff[2:-2, 2:-2, vs.tau],
    )
    vs.z_pf = update(
        vs.z_pf,
        at[2:-2, 2:-2, vs.taum1], vs.z_pf[2:-2, 2:-2, vs.tau],
    )

    return KernelOutput(
        z_wf_ff=vs.z_wf_ff,
        z_pf_ff=vs.z_pf_ff,
        theta_rz_ff=vs.theta_rz_ff,
        theta_ss_ff=vs.theta_ss_ff,
        theta_ff=vs.theta_ff,
        z_pf=vs.z_pf,
    )
