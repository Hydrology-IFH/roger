from benchmark_base import benchmark_cli
from pathlib import Path


@benchmark_cli
def main(backend, device, size):
    from roger import runtime_settings as rs, runtime_state as rst
    rs.backend = backend
    rs.backend = device
    rs.force_overwrite = True
    if rs.mpi_comm:
        rs.num_proc = (rst.proc_num, rst.proc_num)
    from roger import roger_routine, roger_kernel, KernelOutput
    from roger.setups.svat import ONEDEVENTSetup
    from roger.core.operators import update, at

    class ONEDEVENT2Benchmark(ONEDEVENTSetup):
        _base_path = Path(__file__).parent
        _input_dir = _base_path / 'input' / 'oneD_event_benchmark'

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = "ONEDEVENT2Benchmark"

            settings.nx, settings.ny, settings.nz = size
            settings.nitt = self._get_nitt(self._input_dir, 'forcing.nc')
            settings.nittevent = self._get_nitt(self._input_dir, 'forcing.nc')
            settings.nittevent_p1 = settings.nittevent + 1
            settings.runlen = settings.nitt * 10 * 60

            settings.dx = 1
            settings.dy = 1
            settings.dz = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0

            settings.enable_groundwater_boundary = False
            settings.enable_lateral_flow = True
            settings.enable_routing = False

        @roger_routine
        def set_parameters_setup(self, state):
            vs = state.variables

            vs.lu_id = update(vs.lu_id, at[2:-2, 2:-2], 8)
            vs.sealing = update(vs.sealing, at[2:-2, 2:-2], 0)
            vs.slope = update(vs.slope, at[2:-2, 2:-2], 0.05)
            vs.slope_per = update(vs.slope_per, at[2:-2, 2:-2], vs.slope * 100)
            vs.S_dep_tot = update(vs.S_dep_tot, at[2:-2, 2:-2], 0)
            vs.z_soil = update(vs.z_soil, at[2:-2, 2:-2], 2200)
            vs.dmpv = update(vs.dmpv, at[2:-2, 2:-2], 100)
            vs.dmph = update(vs.dmph, at[2:-2, 2:-2], 100)
            vs.lmpv = update(vs.lmpv, at[2:-2, 2:-2], 1000)
            vs.theta_ac = update(vs.theta_ac, at[2:-2, 2:-2], 0.13)
            vs.theta_ufc = update(vs.theta_ufc, at[2:-2, 2:-2], 0.24)
            vs.theta_pwp = update(vs.theta_pwp, at[2:-2, 2:-2], 0.23)
            vs.ks = update(vs.ks, at[2:-2, 2:-2], 25)
            vs.kf = update(vs.kf, at[2:-2, 2:-2], 2500)

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
            vs.z_sat = update(vs.z_sat, at[2:-2, 2:-2, :vs.taup1], 0)

        @roger_routine
        def set_forcing_setup(self, state):
            vs = state.variables

            vs.PREC = update(vs.PREC, at[2:-2, 2:-2, :], self._read_var_from_nc("PREC", self._input_dir, 'forcing.nc'))
            vs.TA = update(vs.TA, at[2:-2, 2:-2, :], self._read_var_from_nc("TA", self._input_dir, 'forcing.nc'))
            vs.EVENT_ID = update(vs.EVENT_ID, at[2:-2, 2:-2, 1:], 1)

        @roger_routine
        def set_forcing(self, state):
            vs = state.variables

            vs.update(set_forcing_kernel(state))

        @roger_routine
        def after_timestep(self, state):
            vs = state.variables

            vs.update(after_timestep_kernel(state))

    @roger_kernel
    def set_forcing_kernel(state):
        vs = state.variables

        vs.prec = update(vs.prec, at[2:-2, 2:-2], vs.PREC[2:-2, 2:-2, vs.itt])
        vs.ta = update(vs.ta, at[2:-2, 2:-2, vs.tau], vs.TA[2:-2, 2:-2, vs.itt])

        vs.dt_secs = vs.DT_SECS[vs.itt]
        vs.dt = vs.DT[vs.itt]

        return KernelOutput(
            prec=vs.prec,
            ta=vs.ta,
            dt=vs.dt,
            dt_secs=vs.dt_secs,
        )

    @roger_kernel
    def after_timestep_kernel(state):
        vs = state.variables

        vs.ta = update(
            vs.ta,
            at[2:-2, 2:-2, vs.taum1], vs.ta[2:-2, 2:-2, vs.tau],
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

        return KernelOutput(
            ta=vs.ta,
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
        )

    model = ONEDEVENT2Benchmark()
    model.setup()
    model.run()
    return


if __name__ == "__main__":
    main()
