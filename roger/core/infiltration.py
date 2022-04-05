from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, for_loop, at


@roger_kernel
def calc_inf_mat(state):
    """
    Calculates matrix infiltration
    """
    vs = state.variables

    a = allocate(state.dimensions, ("x", "y"))
    b = allocate(state.dimensions, ("x", "y"))
    inf_mat_pot_sat = allocate(state.dimensions, ("x", "y"))
    inf_mat_pot_rec = allocate(state.dimensions, ("x", "y"))
    l1 = allocate(state.dimensions, ("x", "y"))
    inf_mat_pot = allocate(state.dimensions, ("x", "y"))

    mask1 = (vs.t_sat < vs.t_event_sum[:, :, vs.tau])
    mask2 = (vs.t_sat >= vs.t_event_sum[:, :, vs.tau])
    mask3 = (vs.t_sat > vs.t_event_sum[:, :, vs.taum1]) & (vs.t_sat < vs.t_event_sum[:, :, vs.tau])
    a = update(
        a,
        at[:, :], vs.ks * (vs.t_event_sum[:, :, vs.tau] - vs.t_sat) * vs.maskCatch,
    )
    b = update(
        b,
        at[:, :], vs.Fs + 2 * vs.theta_d * vs.wfs * vs.maskCatch,
    )
    l1 = update(
        l1,
        at[:, :], npx.where(vs.q_hof > vs.ks * vs.dt, (vs.ks * vs.dt * vs.wfs * vs.theta_d) / (vs.q_hof - vs.ks * vs.dt), (vs.ks * vs.dt * vs.wfs * vs.theta_d) / (vs.ks * vs.dt)) * vs.maskCatch,
    )

    # first wetting front
    # potential matrix infiltration rate after saturation is reached
    inf_mat_pot = allocate(state.dimensions, ("x", "y"))
    vs.inf_mat_pot = update(
        vs.inf_mat_pot,
        at[:, :], inf_mat_pot,
    )
    vs.inf_mat_pot = update(
        vs.inf_mat_pot,
        at[:, :], npx.where(mask1, (vs.ks*vs.dt/2) * (1 + (1 + 2*b/a) / (1 + (4*b/a) + (4*vs.Fs_t0**2 / a**2))**0.5) * ((100 - vs.sealing) / 100), vs.inf_mat_pot) * vs.maskCatch,
    )
    # potential matrix infiltration rate before saturation is reached
    vs.inf_mat_pot = update(
        vs.inf_mat_pot,
        at[:, :], npx.where(mask2, vs.ks * vs.dt * (1 + ((vs.wfs * vs.theta_d) / l1)) * ((100 - vs.sealing) / 100), vs.inf_mat_pot) * vs.maskCatch,
    )
    # potential matrix infiltration rate when saturation is reached within
    # time step
    inf_mat_pot_rec = update(
        inf_mat_pot_rec,
        at[:, :], npx.where(mask3, (vs.ks*vs.dt/2) * (1 + (1 + 2*b/a) / (1 + (4*b/a) + (4*vs.Fs_t0**2 / a**2))**0.5), inf_mat_pot_rec) * vs.maskCatch,
    )
    inf_mat_pot_sat = update(
        inf_mat_pot_sat,
        at[:, :], npx.where(mask3, vs.q_hof * (vs.t_sat - vs.t_event_sum[:, :, vs.taum1]), inf_mat_pot_sat) * vs.maskCatch,
    )
    vs.inf_mat_pot = update(
        vs.inf_mat_pot,
        at[:, :], npx.where(mask3, inf_mat_pot_sat + inf_mat_pot_rec * ((100 - vs.sealing) / 100), vs.inf_mat_pot) * vs.maskCatch,
    )

    inf_mat_pot = update(
        inf_mat_pot,
        at[:, :], vs.inf_mat_pot * vs.maskCatch,
    )

    # matrix infiltration
    mask7 = (vs.q_hof < vs.inf_mat_pot)
    mask8 = (vs.q_hof >= vs.inf_mat_pot)
    vs.inf_mat = update(
        vs.inf_mat,
        at[:, :], npx.where(mask7, vs.q_hof, vs.inf_mat) * vs.maskCatch,
    )
    vs.inf_mat = update(
        vs.inf_mat,
        at[:, :], npx.where(mask8, vs.inf_mat_pot, vs.inf_mat) * vs.maskCatch,
    )
    vs.inf_mat = update(
        vs.inf_mat,
        at[:, :], npx.where(vs.inf_mat < 0, 0, vs.inf_mat) * vs.maskCatch,
    )

    # matrix infiltration
    mask9 = (vs.q_hof < vs.inf_mat_pot)
    mask10 = (vs.q_hof >= vs.inf_mat_pot)
    vs.inf_mat = update(
        vs.inf_mat,
        at[:, :], npx.where(mask9, vs.q_hof, vs.inf_mat) * vs.maskCatch,
    )
    vs.inf_mat = update(
        vs.inf_mat,
        at[:, :], npx.where(mask10, vs.inf_mat_pot, vs.inf_mat) * vs.maskCatch,
    )
    vs.inf_mat = update(
        vs.inf_mat,
        at[:, :], npx.where(vs.inf_mat < 0, 0, vs.inf_mat) * vs.maskCatch,
    )

    # update cumulated infiltration while event
    vs.inf_mat_event_csum = update_add(
        vs.inf_mat_event_csum,
        at[:, :], vs.inf_mat * vs.maskCatch,
    )
    vs.inf_mat_pot_event_csum = update_add(
        vs.inf_mat_pot_event_csum,
        at[:, :], vs.inf_mat_pot * vs.maskCatch,
    )

    # wetting front depth
    # change in wetting front depth
    dz_wf = allocate(state.dimensions, ("x", "y"))
    mask11 = (vs.no_wf == 1)
    mask12 = (vs.no_wf == 2)
    dz_wf = update(
        dz_wf,
        at[:, :], (vs.inf_mat / vs.theta_d_t0) * mask11 * vs.maskCatch,
    )
    dz_wf = update(
        dz_wf,
        at[:, :], npx.where(mask12, vs.inf_mat / vs.theta_d, dz_wf) * vs.maskCatch,
    )

    vs.z_wf_t0 = update_add(
        vs.z_wf_t0,
        at[:, :, vs.tau], npx.where(npx.isfinite(dz_wf), dz_wf, 0) * vs.maskCatch,
    )
    vs.z_wf_t1 = update_add(
        vs.z_wf_t1,
        at[:, :, vs.tau], npx.where(npx.isfinite(dz_wf), dz_wf, 0) * vs.maskCatch,
    )
    mask13 = (vs.z_wf_t0[:, :, vs.tau] > vs.z_soil)
    vs.z_wf_t0 = update(
        vs.z_wf_t0,
        at[:, :, vs.tau], npx.where(mask13, vs.z_soil, vs.z_wf_t0[:, :, vs.tau]) * vs.maskCatch,
    )
    mask19 = (vs.z_wf_t1[:, :, vs.tau] > vs.z_soil)
    vs.z_wf_t1 = update(
        vs.z_wf_t1,
        at[:, :, vs.tau], npx.where(mask19, vs.z_soil, vs.z_wf_t1[:, :, vs.tau]) * vs.maskCatch,
    )

    # hortonian overland flow after matrix infiltration
    vs.q_hof = update(
        vs.q_hof,
        at[:, :], vs.q_hof - vs.inf_mat * vs.maskCatch,
    )
    vs.q_hof = update(
        vs.q_hof,
        at[:, :], npx.where(vs.q_hof < 0, 0, vs.q_hof) * vs.maskCatch,
    )
    vs.q_sur = update(
        vs.q_sur,
        at[:, :], vs.q_hof * vs.maskCatch,
    )

    # change in potential wetting front depth while rainfall pause
    dz_wf_t0 = allocate(state.dimensions, ("x", "y"))
    dz_wf_t0 = update(
        dz_wf_t0,
        at[:, :], npx.where((vs.z_wf_fc > 0) & (vs.rain_ground <= 0) & (vs.no_wf == 1), vs.inf_mat_pot / vs.theta_d_t0, 0) * vs.maskCatch,
    )
    # wetting front moves downwards
    vs.z_wf_t0 = update_add(
        vs.z_wf_t0,
        at[:, :, vs.tau], npx.where(npx.isfinite(dz_wf_t0), dz_wf_t0, 0) * vs.maskCatch,
    )
    mask17 = (vs.z_wf_t0[:, :, vs.tau] > vs.z_wf_fc) & (vs.z_wf_fc > 0)
    vs.z_wf_t0 = update(
        vs.z_wf_t0,
        at[:, :, vs.tau], npx.where(mask17, vs.z_wf_fc, vs.z_wf_t0[:, :, vs.tau]) * vs.maskCatch,
    )
    vs.z_wf_t0 = update(
        vs.z_wf_t0,
        at[:, :, vs.tau], npx.where(vs.z_wf_t0[:, :, vs.tau] > vs.z_soil, vs.z_soil, vs.z_wf_t0[:, :, vs.tau]) * vs.maskCatch,
    )

    dz_wf_t1 = allocate(state.dimensions, ("x", "y"))
    dz_wf_t1 = update(
        dz_wf_t1,
        at[:, :], npx.where((vs.z_wf_fc > 0) & (vs.rain_ground <= 0) & (vs.no_wf == 2), vs.inf_mat_pot / vs.theta_d, 0) * vs.maskCatch,
    )
    # wetting front moves downwards
    vs.z_wf_t1 = update_add(
        vs.z_wf_t1,
        at[:, :, vs.tau], npx.where(npx.isfinite(dz_wf_t1), dz_wf_t1, 0) * vs.maskCatch,
    )
    mask18 = (vs.z_wf_t1[:, :, vs.tau] > vs.z_wf_fc) & (vs.z_wf_fc > 0)
    vs.z_wf_t1 = update(
        vs.z_wf_t1,
        at[:, :, vs.tau], npx.where(mask18, vs.z_wf_fc, vs.z_wf_t1[:, :, vs.tau]) * vs.maskCatch,
    )
    vs.z_wf_t1 = update(
        vs.z_wf_t1,
        at[:, :, vs.tau], npx.where(vs.z_wf_t1[:, :, vs.tau] > vs.z_soil, vs.z_soil, vs.z_wf_t1[:, :, vs.tau]) * vs.maskCatch,
    )

    # update wetting front depth and soil moisture deficit
    mask14 = (vs.z_wf_t0[:, :, vs.tau] >= vs.z_wf_t1[:, :, vs.tau]) & (vs.z_wf_t1[:, :, vs.tau] <= 0)
    mask15 = (vs.z_wf_t0[:, :, vs.tau] > vs.z_wf_t1[:, :, vs.tau]) & (vs.z_wf_t1[:, :, vs.tau] > 0)
    mask20 = (vs.z_wf_t0[:, :, vs.tau] <= vs.z_wf_t1[:, :, vs.tau]) & (vs.z_wf_t1[:, :, vs.tau] > 0)
    vs.z_wf = update(
        vs.z_wf,
        at[:, :, vs.tau], npx.where(mask14, vs.z_wf_t0[:, :, vs.tau], vs.z_wf[:, :, vs.tau]) * vs.maskCatch,
    )
    vs.theta_d = update(
        vs.theta_d,
        at[:, :], npx.where(mask14, vs.theta_d_t0, vs.theta_d) * vs.maskCatch,
    )
    vs.theta_d_rel = update(
        vs.theta_d_rel,
        at[:, :], npx.where(mask14, vs.theta_d_rel_t0, vs.theta_d_rel) * vs.maskCatch,
    )
    vs.z_wf = update(
        vs.z_wf,
        at[:, :, vs.taum1], npx.where(mask15, 0, vs.z_wf[:, :, vs.taum1]) * vs.maskCatch,
    )
    vs.z_wf = update(
        vs.z_wf,
        at[:, :, vs.tau], npx.where(mask15, vs.z_wf_t1[:, :, vs.tau], vs.z_wf[:, :, vs.tau]) * vs.maskCatch,
    )
    vs.no_wf = update(
        vs.no_wf,
        at[:, :], npx.where(mask20, 1, vs.no_wf) * vs.maskCatch,
    )
    vs.z_wf = update(
        vs.z_wf,
        at[:, :, vs.tau], npx.where(mask20, vs.z_wf_t0[:, :, vs.tau], vs.z_wf[:, :, vs.tau]) * vs.maskCatch,
    )
    vs.theta_d = update(
        vs.theta_d,
        at[:, :], npx.where(mask20, vs.theta_d_t0, vs.theta_d) * vs.maskCatch,
    )
    vs.theta_d_rel = update(
        vs.theta_d_rel,
        at[:, :], npx.where(mask20, vs.theta_d_rel_t0, vs.theta_d_rel) * vs.maskCatch,
    )

    mask16 = (vs.z_wf[:, :, vs.tau] > vs.z_soil)
    vs.z_wf = update(
        vs.z_wf,
        at[:, :, vs.tau], npx.where(mask16, vs.z_soil, vs.z_wf[:, :, vs.tau]) * vs.maskCatch,
    )
    mask17 = (vs.theta_d_t1 <= 0)
    vs.theta_d = update(
        vs.theta_d,
        at[:, :], npx.where(mask17, vs.theta_d_t0, vs.theta_d) * vs.maskCatch,
    )

    return KernelOutput(inf_mat_pot=vs.inf_mat_pot, inf_mat=vs.inf_mat, q_hof=vs.q_hof, q_sur=vs.q_sur, z_wf=vs.z_wf, z_wf_t0=vs.z_wf_t0, z_wf_t1=vs.z_wf_t1, theta_d_rel=vs.theta_d_rel, theta_d=vs.theta_d)


@roger_kernel
def calc_inf_mp(state):
    """
    Calculates macropore infiltration
    """
    vs = state.variables
    settings = state.settings

    z_wf = allocate(state.dimensions, ("x", "y"))
    z_wf_m1 = allocate(state.dimensions, ("x", "y"))
    dz_wf = allocate(state.dimensions, ("x", "y"))

    # update dual wetting front depth
    z_wf = update(
        z_wf,
        at[:, :], npx.where(vs.no_wf == 1, 0, vs.z_wf_t0[:, :, vs.tau]) * vs.maskCatch,
    )
    z_wf = update(
        z_wf,
        at[:, :], npx.where(vs.no_wf == 2, 0, vs.z_wf_t1[:, :, vs.tau]) * vs.maskCatch,
    )
    z_wf_m1 = update(
        z_wf,
        at[:, :], npx.where(vs.no_wf == 1, 0, vs.z_wf_t0[:, :, vs.taum1]) * vs.maskCatch,
    )
    z_wf_m1 = update(
        z_wf,
        at[:, :], npx.where(vs.no_wf == 2, 0, vs.z_wf_t1[:, :, vs.taum1]) * vs.maskCatch,
    )

    # length of non saturated vertical macropores at beginning of time step (in mm)
    vs.lmpv_non_sat = update(
        vs.lmpv_non_sat,
        at[:, :], vs.lmpv - z_wf * vs.maskCatch,
    )
    vs.lmpv_non_sat = update(
        vs.lmpv_non_sat,
        at[:, :], npx.where(vs.lmpv_non_sat < 0, 0, vs.lmpv_non_sat) * vs.maskCatch,
    )

    # delta of wetting front depth (in mm)
    dz_wf = update(
        dz_wf,
        at[:, :], z_wf - z_wf_m1 * vs.maskCatch,
    )
    mask1 = (z_wf >= vs.lmpv)
    dz_wf = update(
        dz_wf,
        at[:, :], npx.where(mask1, vs.lmpv_non_sat, dz_wf) * vs.maskCatch,
    )
    dz_wf = update(
        dz_wf,
        at[:, :], npx.where(vs.lmpv_non_sat <= 0, 0, dz_wf) * vs.maskCatch,
    )
    dz_wf = update(
        dz_wf,
        at[:, :], npx.where(dz_wf <= 0, 0, dz_wf) * vs.maskCatch,
    )

    # length of non-saturated macropores at beginning of time step (in mm)
    vs.lmpv_non_sat = update(
        vs.lmpv_non_sat,
        at[:, :], vs.lmpv - vs.z_wf[:, :, vs.tau] * vs.maskCatch,
    )
    vs.lmpv_non_sat = update(
        vs.lmpv_non_sat,
        at[:, :], npx.where(vs.lmpv_non_sat < 0, 0, vs.lmpv_non_sat) * vs.maskCatch,
    )

    # determine computation steps. additional computation steps for hourly
    # time step.
    computation_steps = npx.int64(npx.round(vs.dt / (1 / 5), 0))  # based on hours

    vs.lmpv_non_sat = update(
        vs.lmpv_non_sat,
        at[:, :], npx.where(computation_steps == 1, vs.lmpv_non_sat + dz_wf / 1.39, vs.lmpv_non_sat) * vs.maskCatch,
    )

    # variables to calculate y(t)
    # 0 = y1
    # 1 = y2
    # 2 = y3
    # 3 = ym1
    # 4 = a
    # 5 = b1
    # 6 = b2
    # 7 = c
    # 8 = inf_mp_pot_di
    # 9 = inf_mp_di
    # 10 = q_hof_di
    # 11 = inf_mp
    # 12 = inf_mp_pot
    # 13 = inf_mp_event_csum
    # 14 = t
    # 15 = y
    loop_arr = allocate(state.dimensions, ("x", "y", 16))

    loop_arr = update(
        loop_arr,
        at[:, :, 2], (settings.r_mp / 2) * vs.maskCatch,
    )
    loop_arr = update(
        loop_arr,
        at[:, :, 4], vs.theta_d * settings.r_mp ** 2 * vs.maskCatch,
    )
    loop_arr = update(
        loop_arr,
        at[:, :, 13], vs.inf_mp_event_csum * vs.maskCatch,
    )
    loop_arr = update(
        loop_arr,
        at[:, :, 3], vs.y_mp[:, :, vs.taum1] * vs.maskCatch,
    )

    def loop_body(i, loop_arr):
        # determine computation steps. additional computation steps for hourly
        # time step.
        computation_steps = npx.int64(npx.round(vs.dt / (1 / 5), 0))  # based on hours
        loop_arr = update(
            loop_arr,
            at[:, :, 10], vs.q_hof * (vs.mp_drain_area / computation_steps) * vs.maskCatch,
        )
        loop_arr = update_add(
            loop_arr,
            at[:, :, 14], (vs.dt / computation_steps) * vs.maskCatch,
        )
        loop_arr = update(
            loop_arr,
            at[:, :, 7], vs.ks * vs.wfs * loop_arr[:, :, 14] * vs.maskCatch,
        )
        loop_arr = update(
            loop_arr,
            at[:, :, 7], npx.where(npx.isnan(loop_arr[:, :, 7]), 0, loop_arr[:, :, 7]) * vs.maskCatch,
        )
        loop_arr = update(
            loop_arr,
            at[:, :, 5], (6**0.5 * 2 * (loop_arr[:, :, 7]*(6*loop_arr[:, :, 7] - loop_arr[:, :, 4]))**0.5) * vs.maskCatch,
        )
        loop_arr = update(
            loop_arr,
            at[:, :, 5], npx.where(npx.isnan(loop_arr[:, :, 5]), 0, loop_arr[:, :, 5]) * vs.maskCatch,
        )
        loop_arr = update(
            loop_arr,
            at[:, :, 6], (settings.r_mp*vs.theta_d**2) * (12*loop_arr[:, :, 7] - loop_arr[:, :, 4] + loop_arr[:, :, 5]) * vs.maskCatch,
        )
        loop_arr = update(
            loop_arr,
            at[:, :, 6], npx.where(npx.isnan(loop_arr[:, :, 6]), 0, loop_arr[:, :, 6]) * vs.maskCatch,
        )
        loop_arr = update(
            loop_arr,
            at[:, :, 6], npx.where(loop_arr[:, :, 6] <= 0, 0, loop_arr[:, :, 6]) * vs.maskCatch,
        )
        loop_arr = update(
            loop_arr,
            at[:, :, 0], ((loop_arr[:, :, 6]**(1/3)) / vs.theta_d) * 0.5 * vs.maskCatch,
        )
        loop_arr = update(
            loop_arr,
            at[:, :, 1], (loop_arr[:, :, 4] / (loop_arr[:, :, 6]**(1/3))) * 0.5 * vs.maskCatch,
        )
        # loop_arr = update(
        #     loop_arr,
        #     at[:, :, 2], npx.where(loop_arr[:, :, 6] <= 0, 0, loop_arr[:, :, 3]) * vs.maskCatch,
        # )
        loop_arr = update(
            loop_arr,
            at[:, :, 15], (loop_arr[:, :, 0] + loop_arr[:, :, 1] + loop_arr[:, :, 2]) * vs.maskCatch,
        )
        # potential radial length of macropore wetting front
        loop_arr = update(
            loop_arr,
            at[:, :, 15], npx.where(loop_arr[:, :, 15] < settings.r_mp, settings.r_mp, loop_arr[:, :, 15]) * vs.maskCatch,
        )
        # constrain the potential radial length of macropore wetting front
        # to the radial length at beginning of computation step
        loop_arr = update(
            loop_arr,
            at[:, :, 15], npx.where(loop_arr[:, :, 15] < loop_arr[:, :, 3], loop_arr[:, :, 3], loop_arr[:, :, 15]) * vs.maskCatch,
        )
        # potential macropore infiltration (in mm/dt)
        loop_arr = update(
            loop_arr,
            at[:, :, 8], (settings.pi * (loop_arr[:, :, 15]**2 - loop_arr[:, :, 3]**2) * vs.lmpv_non_sat * vs.theta_d * vs.dmpv * 1e-06) * vs.maskCatch,
        )
        loop_arr = update_add(
            loop_arr,
            at[:, :, 12], loop_arr[:, :, 8] * vs.maskCatch,
        )
        # constrain macropore infiltration with matrix infiltration excess
        mask1 = (loop_arr[:, :, 8] > loop_arr[:, :, 10])
        # actual macropore infiltration
        loop_arr = update(
            loop_arr,
            at[:, :, 9], loop_arr[:, :, 8] * vs.maskCatch,
        )
        loop_arr = update(
            loop_arr,
            at[:, :, 9], npx.where(mask1, loop_arr[:, :, 10], loop_arr[:, :, 9]) * vs.maskCatch,
        )
        loop_arr = update(
            loop_arr,
            at[:, :, 9], npx.where(vs.lmpv_non_sat == 0, 0, loop_arr[:, :, 9]) * vs.maskCatch,
        )
        loop_arr = update_add(
            loop_arr,
            at[:, :, 11], loop_arr[:, :, 9] * vs.maskCatch,
        )

        # calculate radial length of macropore wetting front
        loop_arr = update_add(
            loop_arr,
            at[:, :, 13], loop_arr[:, :, 9] * vs.maskCatch,
        )
        loop_arr = update(
            loop_arr,
            at[:, :, 15], settings.r_mp + ((loop_arr[:, :, 13] / (vs.dmpv * vs.theta_d)) / settings.pi)**.5 * vs.maskCatch,
        )
        loop_arr = update(
            loop_arr,
            at[:, :, 15], npx.where(loop_arr[:, :, 15] < settings.r_mp, settings.r_mp, loop_arr[:, :, 15]) * vs.maskCatch,
        )

        # duration of macropore infiltration
        loop_arr = update(
            loop_arr,
            at[:, :, 14], vs.theta_d / (vs.ks * vs.wfs * settings.r_mp) * (loop_arr[:, :, 15]**3 / 3. - loop_arr[:, :, 15]**2 * settings.r_mp / 2. + settings.r_mp**3 / 6.) * vs.maskCatch,
        )

        loop_arr = update(
            loop_arr,
            at[:, :, 11], npx.where(loop_arr[:, :, 11] < 0, 0, loop_arr[:, :, 11]) * vs.maskCatch,
        )

        loop_arr = update(
            loop_arr,
            at[:, :, 3], loop_arr[:, :, 15] * vs.maskCatch,
        )

        return loop_arr

    loop_arr = for_loop(0, computation_steps, loop_body, loop_arr)

    vs.y_mp = update(
        vs.y_mp,
        at[:, :, vs.tau], loop_arr[:, :, 15] * vs.maskCatch,
    )

    vs.y_mp = update(
        vs.y_mp,
        at[:, :, vs.tau], npx.where(npx.isnan(vs.y_mp[:, :, vs.tau]), 0, vs.y_mp[:, :, vs.tau]) * vs.maskCatch,
    )

    vs.inf_mp = update(
        vs.inf_mp,
        at[:, :], loop_arr[:, :, 11] * vs.maskCatch,
    )

    vs.inf_mp = update(
        vs.inf_mp,
        at[:, :], npx.where(npx.isnan(vs.inf_mp), 0, vs.inf_mp) * vs.maskCatch,
    )

    vs.inf_mp_event_csum = update_add(
        vs.inf_mp_event_csum,
        at[:, :], vs.inf_mp * vs.maskCatch,
    )

    # potential hortonian overland flow after matrix and macropore
    # infiltration
    vs.q_hof = update(
        vs.q_hof,
        at[:, :], (vs.q_hof - vs.inf_mp) * vs.maskCatch,
    )
    vs.q_hof = update(
        vs.q_hof,
        at[:, :], npx.where(vs.q_hof < 0, 0, vs.q_hof) * vs.maskCatch,
    )

    vs.q_sur = update(
        vs.q_sur,
        at[:, :], vs.q_hof * vs.maskCatch,
    )

    # lower boundary condition of vertical macropores
    if settings.enable_macropore_lower_boundary_condition:
        npx.where((vs.q_hof > vs.ks * vs.dt * vs.mp_drain_area), vs.ks * vs.dt * vs.mp_drain_area, vs.q_hof)
        mask_lbc = (vs.z_wf[:, :, vs.tau] > vs.lmpv[:, :])
        vs.inf_mp = update_add(
            vs.inf_mp,
            at[:, :], npx.where(mask_lbc, npx.where((vs.q_hof > vs.ks * vs.dt * vs.mp_drain_area), vs.ks * vs.dt * vs.mp_drain_area, vs.q_hof), 0) * vs.maskCatch,
        )
        vs.q_hof = update_add(
            vs.q_hof,
            at[:, :], -npx.where(mask_lbc, npx.where((vs.q_hof > vs.ks * vs.dt * vs.mp_drain_area), vs.ks * vs.dt * vs.mp_drain_area, vs.q_hof), 0) * vs.maskCatch,
        )
        vs.q_hof = update(
            vs.q_hof,
            at[:, :], npx.where(vs.q_hof < 0, 0, vs.q_hof) * vs.maskCatch,
        )

        vs.q_sur = update(
            vs.q_sur,
            at[:, :], vs.q_hof * vs.maskCatch,
        )

    return KernelOutput(inf_mp=vs.inf_mp, inf_mp_event_csum=vs.inf_mp_event_csum, y_mp=vs.y_mp, q_hof=vs.q_hof, q_sur=vs.q_sur)


@roger_kernel
def calc_inf_sc(state):
    """
    Calculates shrinkage crack infiltration
    """
    vs = state.variables
    settings = state.settings

    z_wf = allocate(state.dimensions, ("x", "y"))
    z_wf_m1 = allocate(state.dimensions, ("x", "y"))
    dz_wf = allocate(state.dimensions, ("x", "y"))

    # update dual wetting front depth
    z_wf = update(
        z_wf,
        at[:, :], npx.where(vs.no_wf == 1, 0, vs.z_wf_t0[:, :, vs.tau]) * vs.maskCatch,
    )
    z_wf = update(
        z_wf,
        at[:, :], npx.where(vs.no_wf == 2, 0, vs.z_wf_t1[:, :, vs.tau]) * vs.maskCatch,
    )
    z_wf_m1 = update(
        z_wf,
        at[:, :], npx.where(vs.no_wf == 1, 0, vs.z_wf_t0[:, :, vs.taum1]) * vs.maskCatch,
    )
    z_wf_m1 = update(
        z_wf,
        at[:, :], npx.where(vs.no_wf == 2, 0, vs.z_wf_t1[:, :, vs.taum1]) * vs.maskCatch,
    )

    # length of non saturated shrinkage crack at beginning of time step (in mm)
    vs.z_sc_non_sat = update(
        vs.z_sc_non_sat,
        at[:, :], vs.z_sc - z_wf * vs.maskCatch,
    )
    vs.z_sc_non_sat = update(
        vs.z_sc_non_sat,
        at[:, :], npx.where(vs.z_sc_non_sat < 0, 0, vs.z_sc_non_sat) * vs.maskCatch,
    )

    # delta of wetting front depth (in mm)
    dz_wf = update(
        dz_wf,
        at[:, :], z_wf - z_wf_m1 * vs.maskCatch,
    )
    mask1 = (z_wf >= vs.z_sc)
    dz_wf = update(
        dz_wf,
        at[:, :], npx.where(mask1, vs.z_sc_non_sat, dz_wf) * vs.maskCatch,
    )
    dz_wf = update(
        dz_wf,
        at[:, :], npx.where(vs.z_sc_non_sat <= 0, 0, dz_wf) * vs.maskCatch,
    )
    dz_wf = update(
        dz_wf,
        at[:, :], npx.where(dz_wf <= 0, 0, dz_wf) * vs.maskCatch,
    )

    # length of non-saturated macropores at beginning of time step (in mm)
    vs.z_sc_non_sat = update(
        vs.z_sc_non_sat,
        at[:, :], vs.z_sc - vs.z_wf[:, :, vs.tau] * vs.maskCatch,
    )
    vs.z_sc_non_sat = update(
        vs.z_sc_non_sat,
        at[:, :], npx.where(vs.z_sc_non_sat < 0, 0, vs.z_sc_non_sat) * vs.maskCatch,
    )

    # determine computation steps. additional computation steps for hourly
    # time step.
    computation_steps = npx.int64(npx.round(vs.dt / (1 / 5), 0))  # based on hours

    vs.z_sc_non_sat = update(
        vs.z_sc_non_sat,
        at[:, :], npx.where(computation_steps == 1, vs.z_sc_non_sat + dz_wf / 1.39, vs.z_sc_non_sat) * vs.maskCatch,
    )

    # variables to calculate y(t)
    # 0 = y
    # 1 = ym1
    # 2 = inf_sc_pot_di
    # 3 = inf_sc_di
    # 4 = q_hof_di
    # 5 = inf_sc_event_csum
    # 6 = t
    # 7 = inf_sc
    loop_arr = allocate(state.dimensions, ("x", "y", 8))

    loop_arr = update(
        loop_arr,
        at[:, :, 5], vs.inf_sc_event_csum * vs.maskCatch,
    )
    loop_arr = update(
        loop_arr,
        at[:, :, 1], vs.y_sc[:, :, vs.taum1] * vs.maskCatch,
    )

    def loop_body(i, loop_arr):
        # determine computation steps. additional computation steps for hourly
        # time step.
        computation_steps = npx.int64(npx.round(vs.dt / (1 / 5), 0))  # based on hours
        loop_arr = update(
            loop_arr,
            at[:, :, 4], (vs.q_hof / computation_steps) * vs.maskCatch,
        )
        loop_arr = update_add(
            loop_arr,
            at[:, :, 6], (vs.dt / computation_steps) * vs.maskCatch,
            )
        # potential horizontal length of shrinkage crack wetting front
        loop_arr = update(
            loop_arr,
            at[:, :, 0], (((vs.ks * vs.wfs * loop_arr[:, :, 6] * 2) / vs.theta_d)**0.5) * vs.maskCatch,
        )
        # potential horizontal shrinkage crack infiltration
        loop_arr = update(
            loop_arr,
            at[:, :, 2], ((vs.z_sc_non_sat * vs.theta_d * settings.l_sc) * (loop_arr[:, :, 0] - loop_arr[:, :, 1]) * 1e-06) * vs.maskCatch,
        )
        loop_arr = update(
            loop_arr,
            at[:, :, 2], npx.where(loop_arr[:, :, 2] <= 0, 0, loop_arr[:, :, 2]) * vs.maskCatch,
        )
        # actual horizontal shrinkage crack infiltration
        loop_arr = update(
            loop_arr,
            at[:, :, 3], npx.where(loop_arr[:, :, 2] > loop_arr[:, :, 4], loop_arr[:, :, 4], loop_arr[:, :, 2]) * vs.maskCatch,
        )
        loop_arr = update(
            loop_arr,
            at[:, :, 3], npx.where(vs.z_sc_non_sat <= 0, 0, loop_arr[:, :, 3]) * vs.maskCatch,
        )
        loop_arr = update_add(
            loop_arr,
            at[:, :, 7], loop_arr[:, :, 3] * vs.maskCatch,
        )
        loop_arr = update_add(
            loop_arr,
            at[:, :, 5], loop_arr[:, :, 3] * vs.maskCatch,
        )
        # actual horizontal length of shrinkage crack wetting front
        loop_arr = update(
            loop_arr,
            at[:, :, 0], (loop_arr[:, :, 5] / settings.l_sc / 2) * vs.maskCatch,
        )
        # duration of shrinkage crack infiltration
        loop_arr = update(
            loop_arr,
            at[:, :, 6], ((loop_arr[:, :, 1]**2 * vs.theta_d) / (vs.ks * vs.wfs * 2)) * vs.maskCatch,
        )

        loop_arr = update(
            loop_arr,
            at[:, :, 1], loop_arr[:, :, 0] * vs.maskCatch,
        )

        return loop_arr

    loop_arr = for_loop(0, computation_steps, loop_body, loop_arr)

    vs.y_sc = update(
        vs.y_sc,
        at[:, :, vs.tau], loop_arr[:, :, 0] * vs.maskCatch,
    )

    vs.inf_sc = update(
        vs.inf_sc,
        at[:, :], loop_arr[:, :, 7] * vs.maskCatch,
    )

    vs.inf_sc_event_csum = update_add(
        vs.inf_sc_event_csum,
        at[:, :], vs.inf_sc * vs.maskCatch,
    )

    # potential hortonian overland flow after matrix and macropore
    # infiltration and shrinkage crack infiltration
    vs.q_hof = update(
        vs.q_hof,
        at[:, :], (vs.q_hof - vs.inf_sc) * vs.maskCatch,
    )
    vs.q_hof = update(
        vs.q_hof,
        at[:, :], npx.where(vs.q_hof < 0, 0, vs.q_hof) * vs.maskCatch,
    )
    vs.q_sur = update(
        vs.q_sur,
        at[:, :], vs.q_hof * vs.maskCatch,
    )

    return KernelOutput(inf_sc=vs.inf_sc, inf_sc_event_csum=vs.inf_sc_event_csum, y_sc=vs.y_sc, q_hof=vs.q_hof, q_sur=vs.q_sur, z_sc_non_sat=vs.z_sc_non_sat)


@roger_kernel
def calc_inf_rz(state):
    """
    Calculates infiltration into root zone
    """
    vs = state.variables

    # matrix infiltration into root zone
    vs.inf_mat_rz = update(
        vs.inf_mat_rz,
        at[:, :], vs.inf_mat * vs.maskCatch,
    )

    # macropore infiltration into root zone
    rz_share_mp = allocate(state.dimensions, ("x", "y"))
    rz_share_mp = update(
        rz_share_mp,
        at[:, :], npx.where(vs.lmpv_non_sat > 0, 1. - (vs.lmpv - vs.z_root[:, :, vs.tau]) / vs.lmpv_non_sat, 0) * vs.maskCatch,
    )
    rz_share_mp = update(
        rz_share_mp,
        at[:, :], npx.where(vs.lmpv <= vs.z_root[:, :, vs.tau], 1, rz_share_mp) * vs.maskCatch,
    )
    rz_share_mp = update(
        rz_share_mp,
        at[:, :], npx.where(vs.lmpv_non_sat <= vs.lmpv - vs.z_root[:, :, vs.tau], 0, rz_share_mp) * vs.maskCatch,
    )

    vs.inf_mp_rz = update(
        vs.inf_mp_rz,
        at[:, :], vs.inf_mp * rz_share_mp * vs.maskCatch,
    )

    # shrinkage crack infiltration into root zone
    vs.inf_sc_rz = update(
        vs.inf_sc_rz,
        at[:, :], vs.inf_sc * vs.maskCatch,
    )

    vs.inf_rz = update(
        vs.inf_rz,
        at[:, :], (vs.inf_mat_rz + vs.inf_mp_rz + vs.inf_sc_rz) * vs.maskCatch,
    )

    # update root zone storage after infiltration
    vs.S_fp_rz = update_add(
        vs.S_fp_rz,
        at[:, :], vs.inf_rz * vs.maskCatch,
    )

    # root zone fine pore excess fills root zone large pores
    mask = (vs.S_fp_rz > vs.S_ufc_rz)
    vs.S_lp_rz = update_add(
        vs.S_lp_rz,
        at[:, :], (vs.S_fp_rz - vs.S_ufc_rz) * mask * vs.maskCatch,
    )
    vs.S_fp_rz = update(
        vs.S_fp_rz,
        at[:, :], npx.where(mask, vs.S_ufc_rz, vs.S_fp_rz) * vs.maskCatch,
    )

    return KernelOutput(inf_mat_rz=vs.inf_mat_rz, inf_mp_rz=vs.inf_mp_rz, inf_sc_rz=vs.inf_sc_rz, inf_rz=vs.inf_rz, S_fp_rz=vs.S_fp_rz, S_lp_rz=vs.S_lp_rz)


@roger_kernel
def calc_inf_ss(state):
    """
    Calculates infiltration into subsoil
    """
    vs = state.variables

    # macropore infiltration into subsoil
    rz_share_mp = allocate(state.dimensions, ("x", "y"))
    rz_share_mp = update(
        rz_share_mp,
        at[:, :], npx.where(vs.lmpv_non_sat > 0, 1. - (vs.lmpv - vs.z_root[:, :, vs.tau]) / vs.lmpv_non_sat, 0) * vs.maskCatch,
    )
    rz_share_mp = update(
        rz_share_mp,
        at[:, :], npx.where(vs.lmpv <= vs.z_root[:, :, vs.tau], 1, rz_share_mp) * vs.maskCatch,
    )
    rz_share_mp = update(
        rz_share_mp,
        at[:, :], npx.where(vs.lmpv_non_sat <= vs.lmpv - vs.z_root[:, :, vs.tau], 0, rz_share_mp) * vs.maskCatch,
    )
    vs.inf_mp_ss = update(
        vs.inf_mp_ss,
        at[:, :], vs.inf_mp * (1 - rz_share_mp) * vs.maskCatch,
    )
    vs.inf_ss = update(
        vs.inf_ss,
        at[:, :], vs.inf_mp_ss * vs.maskCatch,
    )

    # update subsoil storage after infiltration
    vs.S_fp_ss = update_add(
        vs.S_fp_ss,
        at[:, :], vs.inf_ss * vs.maskCatch,
    )

    # subsoil fine pore excess fills subsoil large pores
    mask = (vs.S_fp_ss > vs.S_ufc_ss)
    vs.S_lp_ss = update_add(
        vs.S_lp_ss,
        at[:, :], (vs.S_fp_ss - vs.S_ufc_ss) * mask * vs.maskCatch,
    )
    vs.S_fp_ss = update(
        vs.S_fp_ss,
        at[:, :], npx.where(mask, vs.S_ufc_ss, vs.S_fp_ss) * vs.maskCatch,
    )

    return KernelOutput(inf_mp_ss=vs.inf_mp_ss, inf_ss=vs.inf_ss, S_fp_ss=vs.S_fp_ss, S_lp_ss=vs.S_lp_ss)


@roger_kernel
def calc_inf(state):
    """
    Calculates infiltration
    """
    vs = state.variables

    vs.inf = update(
        vs.inf,
        at[:, :], (vs.inf_rz + vs.inf_ss) * vs.maskCatch,
    )

    return KernelOutput(inf=vs.inf)


@roger_kernel
def calc_z_wf_fc(state):
    """
    Calculates wetting front depth to reach field capacity at time t.
    """
    vs = state.variables

    z_wf_fc = allocate(state.dimensions, ("x", "y"))
    z_wf_fc = update(
        z_wf_fc,
        at[:, :], npx.where(vs.theta_d_fp > 0, vs.inf_mat_event_csum / vs.theta_d_fp, vs.z_wf[:, :, vs.tau]) * vs.maskCatch,
    )
    z_wf_fc = update(
        z_wf_fc,
        at[:, :], npx.where(z_wf_fc > vs.z_soil, vs.z_soil, z_wf_fc) * vs.maskCatch,
    )

    return z_wf_fc


@roger_kernel
def calc_theta_d(state):
    """
    Calculates soil moisture deficit.
    """
    vs = state.variables

    theta_d = allocate(state.dimensions, ("x", "y"))

    mask1 = (vs.z_root[:, :, vs.tau] > 0)
    theta_d = update(
        theta_d,
        at[:, :], (vs.theta_sat - vs.theta_rz[:, :, vs.tau]) * (1 - vs.sealing/100) * mask1 * vs.maskCatch,
    )
    theta_d = update(
        theta_d,
        at[:, :], npx.where(vs.z_soil <= 0, 0.01, theta_d) * vs.maskCatch,
    )
    theta_d = update(
        theta_d,
        at[:, :], npx.where(theta_d <= 0, 0.01, theta_d) * vs.maskCatch,
    )

    return theta_d


@roger_kernel
def calc_theta_d_rel(state):
    """
    Calculates relative soil moisture deficit.
    """
    vs = state.variables

    theta_d_rel = allocate(state.dimensions, ("x", "y"))

    mask1 = (vs.z_root[:, :, vs.tau] > 0)
    theta_d_rel = update(
        theta_d_rel,
        at[:, :], ((vs.theta_sat - vs.theta_rz[:, :, vs.tau]) / (vs.theta_sat - vs.theta_pwp)) * (1 - vs.sealing/100) * mask1 * vs.maskCatch,
    )
    theta_d_rel = update(
        theta_d_rel,
        at[:, :], npx.where(vs.z_soil <= 0, 0.01, theta_d_rel) * vs.maskCatch,
    )
    theta_d_rel = update(
        theta_d_rel,
        at[:, :], npx.where(theta_d_rel <= 0, 0.01, theta_d_rel) * vs.maskCatch,
    )

    return theta_d_rel


def _calc_theta_d_rel(state):
    """
    Calculates relative soil moisture deficit.
    """
    vs = state.variables

    theta_d_rel = allocate(state.dimensions, ("x", "y"))

    mask1 = (vs.z_soil > 0)
    theta_d_rel = update(
        theta_d_rel,
        at[:, :], (vs.theta_d / (vs.theta_sat - vs.theta_pwp)) * (1 - vs.sealing/100) * mask1 * vs.maskCatch,
    )
    theta_d_rel = update(
        theta_d_rel,
        at[:, :], npx.where(vs.z_soil <= 0, 0.01, theta_d_rel) * vs.maskCatch,
    )
    theta_d_rel = update(
        theta_d_rel,
        at[:, :], npx.where(theta_d_rel <= 0, 0.01, theta_d_rel) * vs.maskCatch,
    )

    return theta_d_rel


@roger_kernel
def calc_theta_d_fp(state):
    """
    Calculates soil moisture deficit in fine pores.
    """
    vs = state.variables

    theta_d_fp = allocate(state.dimensions, ("x", "y"))

    mask1 = (vs.z_soil > 0)
    theta_d_fp = update(
        theta_d_fp,
        at[:, :], (vs.theta_fc - vs.theta_rz[:, :, vs.tau]) * (1 - vs.sealing/100) * mask1 * vs.maskCatch,
    )
    theta_d_fp = update(
        theta_d_fp,
        at[:, :], npx.where(vs.z_soil <= 0, 0.01, theta_d_fp) * vs.maskCatch,
    )
    theta_d_fp = update(
        theta_d_fp,
        at[:, :], npx.where(theta_d_fp <= 0, 0.01, theta_d_fp) * vs.maskCatch,
    )

    return theta_d_fp


@roger_kernel
def calc_sat_itt(state):
    """
    Calculates iteration when soil matrix is saturated while infiltration.
    """
    vs = state.variables

    sat_itt = allocate(state.dimensions, ("x", "y"), dtype=int)

    def loop_body(i, sat_itt):
        # mask1 = ((vs.ks * vs.DT_event[i] * vs.theta_d * vs.wfs) / vs.prec_event[:, :, i] - (vs.ks * vs.DT_event[i]) == vs.prec_event_csum[:, :, i+1]) & (sat_itt <= 0)
        mask2 = ((vs.prec_event[:, :, i] * (1 / vs.DT_event[i]) - vs.ks) * vs.prec_event_csum[:, :, i+1] >= vs.ks * vs.theta_d * vs.wfs) & (sat_itt <= 0)
        # sat_itt = update(
        #     sat_itt,
        #     at[:, :], npx.where(mask1, i, sat_itt),
        # )
        sat_itt = update(
            sat_itt,
            at[:, :], npx.where(mask2, i, sat_itt),
        )

        return sat_itt

    sat_itt = for_loop(0, npx.int64(vs.event_end-vs.event_start), loop_body, sat_itt)

    mask1 = npx.all(vs.prec_event * (1 / vs.DT_event[npx.newaxis, npx.newaxis, :]) < vs.ks[:, :, npx.newaxis] * vs.DT_event[npx.newaxis, npx.newaxis, :], axis=-1)
    sat_itt = update(
        sat_itt,
        at[:, :], npx.where(mask1, npx.argmax(vs.prec_event_csum, axis=-1) - 1, sat_itt),
    )

    return sat_itt


@roger_kernel
def calc_pi_gr(state, sat_itt):
    """
    Calculates threshold of precipitation intensity
    """
    vs = state.variables

    pi_gr = allocate(state.dimensions, ("x", "y"))

    def loop_body(i, pi_gr):
        mask1 = (sat_itt == i)
        pi_gr = update(
            pi_gr,
            at[:, :], npx.where(mask1, vs.ks * (((vs.theta_d * vs.wfs)/vs.prec_event_csum[:, :, i]) + 1), pi_gr) * vs.maskCatch,
        )

        return pi_gr

    pi_gr = for_loop(0, npx.int64(npx.max(sat_itt)+1), loop_body, pi_gr)

    return pi_gr


@roger_kernel
def calc_pi_m(state, sat_itt):
    """
    Calculates infiltration sum at saturation
    """
    vs = state.variables

    pi_m = allocate(state.dimensions, ("x", "y"))

    def loop_body(i, pi_m):
        mask = (sat_itt == i)
        pi_m = update(
            pi_m,
            at[:, :], npx.where(mask, vs.prec_event[:, :, i] * (1 / vs.DT_event[i]), pi_m) * vs.maskCatch,
        )

        return pi_m

    pi_m = for_loop(0, npx.int64(npx.max(sat_itt)+1), loop_body, pi_m)

    return pi_m


@roger_kernel
def calc_sat_time(state, pi_m, pi_gr, sat_itt):
    """
    Calculates time to reach soil matrix saturation during rainfall event.
    """
    vs = state.variables

    tsum = allocate(state.dimensions, ("x", "y"))

    def loop_body(i, tsum):
        mask1 = (sat_itt == i) & (vs.pi_m > vs.pi_gr)
        mask2 = ((vs.prec_event[:, :, i] * (1 / vs.DT_event[i]) - vs.ks) * vs.prec_event_csum[:, :, i+1] > vs.ks * vs.theta_d * vs.wfs) & (sat_itt == i) & (vs.pi_m < vs.pi_gr)
        tsum = update(
            tsum,
            at[:, :], npx.where(mask1, vs.t_event_csum[i], tsum),
        )
        tsum = update(
            tsum,
            at[:, :], npx.where(mask2, vs.t_event_csum[i] + ((vs.ks * vs.theta_d * vs.wfs) / (vs.pi_m * (vs.pi_m * - vs.ks))) - (vs.DT_event[i] / vs.pi_m) * vs.prec_event_csum[:, :, i], tsum),
        )

        return tsum

    tsum = for_loop(0, npx.int64(npx.max(sat_itt)+1), loop_body, tsum)

    sat_time = allocate(state.dimensions, ("x", "y"))
    sat_time = update(
        sat_time,
        at[:, :], tsum * vs.maskCatch,
    )

    return sat_time


@roger_kernel
def calc_Fs(state, pi_m):
    """
    Calculates infiltration rate at saturation.
    """
    vs = state.variables

    Fs = allocate(state.dimensions, ("x", "y"))

    Fs = update(
        Fs,
        at[:, :], ((vs.ks * vs.theta_d * vs.wfs) / (pi_m - vs.ks)) * vs.maskCatch,
    )
    Fs = update(
        Fs,
        at[:, :], npx.where(pi_m <= vs.ks, pi_m, Fs) * vs.maskCatch,
    )

    return Fs


@roger_kernel
def calc_depth_shrinkage_cracks(state):
    """
    Calculates depth of shrinkage cracks.
    """
    vs = state.variables

    vs.z_sc = update(
        vs.z_sc,
        at[:, :], npx.where(vs.theta_rz[:, :, vs.tau] < vs.theta_4, vs.z_sc_max, npx.where((vs.theta_rz[:, :, vs.tau] >= vs.theta_4) & (vs.theta_rz[:, :, vs.tau] < vs.theta_27), (vs.theta_rz[:, :, vs.tau] - vs.theta_4) / (vs.theta_27 - vs.theta_4), 0) * vs.z_sc_max) * vs.maskCatch,
    )
    vs.z_sc = update(
        vs.z_sc,
        at[:, :], npx.where(vs.theta_rz[:, :, vs.tau] < vs.theta_4, vs.z_sc_max, vs.z_sc[:, :]) * vs.maskCatch,
    )
    vs.z_sc = update(
        vs.z_sc,
        at[:, :], npx.where(vs.theta_rz[:, :, vs.tau] > vs.theta_27, 0, vs.z_sc[:, :]) * vs.maskCatch,
    )
    vs.z_sc = update(
        vs.z_sc,
        at[:, :], ((1. - vs.sealing/100.) * vs.z_sc[:, :]) * vs.maskCatch,
    )

    vs.z_sc = update(
        vs.z_sc,
        at[:, :], npx.where(vs.z_sc[:, :] > vs.z_root[:, :, vs.tau], vs.z_root[:, :, vs.tau], vs.z_sc[:, :]) * vs.maskCatch,
    )

    vs.z_sc = update(
        vs.z_sc,
        at[:, :], npx.where((vs.lu_id == 13), 0, vs.z_sc[:, :]) * vs.maskCatch,
    )

    return KernelOutput(z_sc=vs.z_sc)


@roger_kernel
def set_event_vars(state):
    """
    Set event-based variables at the beginning of an event
    """
    vs = state.variables

    vs.no_wf = update(
        vs.no_wf,
        at[:, :], 1 * vs.maskCatch,
    )

    # wetting front depth (in mm)
    vs.z_wf = update(
        vs.z_wf,
        at[:, :, :], 0,
    )
    # first wetting front
    vs.z_wf_t0 = update(
        vs.z_wf_t0,
        at[:, :, :], 0,
    )
    # second wetting front
    vs.z_wf_t1 = update(
        vs.z_wf_t1,
        at[:, :, :], 0,
    )
    # wetting front depth to reach field capacity
    vs.z_wf_fc = update(
        vs.z_wf_fc,
        at[:, :], 0,
    )
    # accumulated infiltration during an event (in mm)
    # matrix
    vs.inf_mat_event_csum = update(
        vs.inf_mat_event_csum,
        at[:, :], 0,
    )

    # accumulated potential infiltration during an event (in mm)
    # matrix
    vs.inf_mat_pot_event_csum = update(
        vs.inf_mat_pot_event_csum,
        at[:, :], 0,
    )

    # macropores
    vs.inf_mp_event_csum = update(
        vs.inf_mp_event_csum,
        at[:, :], 0,
    )
    # radial length of shrinkage crack wetting front
    vs.y_mp = update(
        vs.y_mp,
        at[:, :, :], 0,
    )
    # shrinkage cracks
    vs.inf_sc_event_csum = update(
        vs.inf_sc_event_csum,
        at[:, :], 0,
    )
    # radial length of shrinkage crack wetting front
    vs.y_sc = update(
        vs.y_sc,
        at[:, :, :], 0,
    )

    # soil moisture deficit
    theta_d = calc_theta_d(state)
    vs.theta_d = update(
        vs.theta_d,
        at[:, :], theta_d * vs.maskCatch,
    )
    theta_d_rel = calc_theta_d_rel(state)
    vs.theta_d_rel = update(
        vs.theta_d_rel,
        at[:, :], theta_d_rel * vs.maskCatch,
    )
    vs.theta_d_t0 = update(
        vs.theta_d_t0,
        at[:, :], theta_d * vs.maskCatch,
    )
    vs.theta_d_rel_t0 = update(
        vs.theta_d_rel_t0,
        at[:, :], theta_d_rel * vs.maskCatch,
    )
    theta_d_fp = calc_theta_d_fp(state)
    vs.theta_d_fp = update(
        vs.theta_d_fp,
        at[:, :], theta_d_fp * vs.maskCatch,
    )
    # accumulated precipitation wihin event (in mm)
    vs.prec_event_csum = update(
        vs.prec_event_csum,
        at[:, :, :], 0,
    )
    vs.prec_event_csum = update(
        vs.prec_event_csum,
        at[:, :, 1:], npx.cumsum(vs.prec_event, axis=-1),
    )
    # accumulated time during an event (in mm)
    vs.t_event_sum = update(
        vs.t_event_sum,
        at[:, :, :], 0,
    )
    vs.t_event_csum = update(
        vs.t_event_csum,
        at[:], 0,
    )
    vs.t_event_csum = update(
        vs.t_event_csum,
        at[1:], npx.cumsum(vs.DT_event),
    )

    # number of time step of matrix saturation
    sat_itt = calc_sat_itt(state)
    vs.sat_itt = update(
        vs.sat_itt,
        at[:, :], sat_itt * vs.maskCatch,
    )
    # threshold intensity (mm/h)
    pi_gr = calc_pi_gr(state, sat_itt)
    vs.pi_gr = update(
        vs.pi_gr,
        at[:, :], pi_gr * vs.maskCatch,
    )
    # precipitation intensity at saturation (mm/h)
    pi_m = calc_pi_m(state, sat_itt)
    vs.pi_m = update(
        vs.pi_m,
        at[:, :], pi_m * vs.maskCatch,
    )
    # saturation time (hours)
    t_sat = calc_sat_time(state, pi_m, pi_gr, sat_itt)
    vs.t_sat = update(
        vs.t_sat,
        at[:, :], npx.round(t_sat, 2) * vs.maskCatch,
    )
    # infiltration at saturation (mm)
    Fs = calc_Fs(state, pi_m)
    vs.Fs = update(
        vs.Fs,
        at[:, :], Fs * vs.maskCatch,
    )
    # reset accumulated soil evaporation deficit
    vs.de = update(
        vs.de,
        at[:, :], 0,
    )

    # depth of shrinkage cracks
    vs.update(calc_depth_shrinkage_cracks(state))

    return KernelOutput(no_wf=vs.no_wf, z_wf=vs.z_wf,
                        z_wf_t0=vs.z_wf_t0,
                        z_wf_t1=vs.z_wf_t1, z_wf_fc=vs.z_wf_fc,
                        inf_mat_event_csum=vs.inf_mat_event_csum,
                        inf_mat_pot_event_csum=vs.inf_mat_pot_event_csum,
                        inf_mp_event_csum=vs.inf_mp_event_csum,
                        y_mp=vs.y_mp,
                        inf_sc_event_csum=vs.inf_sc_event_csum,
                        y_sc=vs.y_sc,
                        theta_d=vs.theta_d,
                        theta_d_rel=vs.theta_d_rel,
                        theta_d_t0=vs.theta_d_t0,
                        theta_d_fp=vs.theta_d_fp,
                        prec_event_csum=vs.prec_event_csum,
                        t_event_sum=vs.t_event_sum,
                        t_event_csum=vs.t_event_csum,
                        sat_itt=vs.sat_itt,
                        pi_gr=vs.pi_gr,
                        pi_m=vs.pi_m,
                        t_sat=vs.t_sat,
                        Fs=vs.Fs,
                        de=vs.de,
                        z_sc=vs.z_sc
                        )


@roger_kernel
def set_event_vars_start_rainfall_pause(state):
    """
    Set event-based variables at the beginning of an rainfall pause
    """
    vs = state.variables

    z_wf_fc = calc_z_wf_fc(state)

    vs.z_wf_fc = update(
        vs.z_wf_fc,
        at[:, :], z_wf_fc * vs.maskCatch,
    )

    return KernelOutput(z_wf_fc=vs.z_wf_fc)


@roger_kernel
def set_event_vars_end_rainfall_pause(state):
    """
    Set event-based variables at the end of an rainfall pause
    """
    vs = state.variables

    vs.no_wf = update(
        vs.no_wf,
        at[:, :], 2 * vs.maskCatch,
    )

    theta_d = calc_theta_d(state)
    vs.theta_d = update(
        vs.theta_d,
        at[:, :], theta_d * vs.maskCatch,
    )
    theta_d_rel = calc_theta_d_rel(state)
    vs.theta_d_rel = update(
        vs.theta_d_rel,
        at[:, :], theta_d_rel * vs.maskCatch,
    )
    # second wetting front
    vs.z_wf_t1 = update(
        vs.z_wf_t1,
        at[:, :, :], 0,
    )

    # accumulated event precipitation
    vs.prec_event_csum = update(
        vs.prec_event_csum,
        at[:, :, :], 0,
    )
    vs.prec_event_csum = update(
        vs.prec_event_csum,
        at[:, :, 1:], npx.cumsum(vs.prec_event, axis=-1),
    )
    # accumulated time during an event (in mm)
    vs.t_event_sum = update(
        vs.t_event_sum,
        at[:, :, :], 0,
    )
    vs.t_event_csum = update(
        vs.t_event_csum,
        at[:], 0,
    )
    vs.t_event_csum = update(
        vs.t_event_csum,
        at[1:], npx.cumsum(vs.DT_event),
    )

    # number of time step of matrix saturation
    sat_itt = calc_sat_itt(state)
    vs.sat_itt = update(
        vs.sat_itt,
        at[:, :], sat_itt * vs.maskCatch,
    )
    # threshold intensity (mm/h)
    pi_gr = calc_pi_gr(state, sat_itt)
    vs.pi_gr = update(
        vs.pi_gr,
        at[:, :], pi_gr * vs.maskCatch,
    )
    # precipitation intensity at saturation (mm/h)
    pi_m = calc_pi_m(state, sat_itt)
    vs.pi_m = update(
        vs.pi_m,
        at[:, :], pi_m * vs.maskCatch,
    )
    # saturation time (hours)
    t_sat = calc_sat_time(state, pi_m, pi_gr, sat_itt)
    vs.t_sat = update(
        vs.t_sat,
        at[:, :], t_sat * vs.maskCatch,
    )
    # infiltration at saturation (mm)
    Fs = calc_Fs(state, pi_m)
    vs.Fs = update(
        vs.Fs,
        at[:, :], Fs * vs.maskCatch,
    )

    return KernelOutput(no_wf=vs.no_wf,
                        z_wf_t1=vs.z_wf_t1,
                        theta_d=vs.theta_d,
                        theta_d_rel=vs.theta_d_rel,
                        prec_event_csum=vs.prec_event_csum,
                        t_event_sum=vs.t_event_sum,
                        t_event_csum=vs.t_event_csum,
                        sat_itt=vs.sat_itt,
                        pi_gr=vs.pi_gr,
                        pi_m=vs.pi_m,
                        t_sat=vs.t_sat,
                        Fs=vs.Fs,
                        de=vs.de
                        )


@roger_kernel
def reset_event_vars(state):
    """
    Reset event-based variables at the end of an event
    """
    vs = state.variables

    # wetting front depth (in mm)
    vs.z_wf = update(
        vs.z_wf,
        at[:, :, :], 0,
    )
    # first wetting front
    vs.z_wf_t0 = update(
        vs.z_wf_t0,
        at[:, :, :], 0,
    )
    # second wetting front
    vs.z_wf_t1 = update(
        vs.z_wf_t1,
        at[:, :, :], 0,
    )
    # radial length of shrinkage crack wetting front
    vs.y_mp = update(
        vs.y_mp,
        at[:, :, vs.tau], 0,
    )
    # radial length of shrinkage crack wetting front
    vs.y_sc = update(
        vs.y_sc,
        at[:, :, :], 0,
    )
    # soil moisture deficit
    theta_d = calc_theta_d(state)
    vs.theta_d = update(
        vs.theta_d,
        at[:, :], theta_d * vs.maskCatch,
    )
    vs.theta_d_t0 = update(
        vs.theta_d_t0,
        at[:, :], theta_d * vs.maskCatch,
    )
    vs.sat_itt = update(
        vs.sat_itt,
        at[:, :], 0,
    )
    vs.pi_gr = update(
        vs.pi_gr,
        at[:, :], 0,
    )
    vs.pi_m = update(
        vs.pi_m,
        at[:, :], 0,
    )
    vs.t_sat = update(
        vs.t_sat,
        at[:, :], 0,
    )
    vs.Fs = update(
        vs.Fs,
        at[:, :], 0,
    )
    vs.z_sc = update(
        vs.z_sc,
        at[:, :], 0,
    )

    return KernelOutput(z_wf=vs.z_wf,
                        z_wf_t0=vs.z_wf_t0,
                        z_wf_t1=vs.z_wf_t1,
                        y_mp=vs.y_mp,
                        y_sc=vs.y_sc,
                        theta_d=vs.theta_d,
                        theta_d_t0=vs.theta_d_t0,
                        sat_itt=vs.sat_itt,
                        pi_gr=vs.pi_gr,
                        pi_m=vs.pi_m,
                        t_sat=vs.t_sat,
                        Fs=vs.Fs,
                        z_sc=vs.z_sc)


@roger_routine
def calculate_infiltration(state):
    """
    Calculates infiltration
    """
    vs = state.variables
    settings = state.settings

    arr_itt = allocate(state.dimensions, ("x", "y"))
    arr_itt = update(
        arr_itt,
        at[:, :], vs.itt,
    )
    cond1 = ((vs.EVENT_ID[:, :, vs.itt-1] == 0) & (vs.EVENT_ID[:, :, vs.itt] >= 1) & (arr_itt >= 1))
    cond2 = ((vs.PREC[:, :, vs.itt] == 0) & (vs.PREC[:, :, vs.itt - 1] != 0) & (vs.EVENT_ID[:, :, vs.itt - 1] >= 1) & (arr_itt >= 1))
    cond3 = ((vs.PREC[:, :, vs.itt] != 0) & (vs.PREC[:, :, vs.itt - 1] == 0) & (vs.EVENT_ID[:, :, vs.itt - 1] == vs.EVENT_ID[:, :, vs.itt]) & (arr_itt >= 1))
    cond4 = ((vs.EVENT_ID[:, :, vs.itt-1] >= 1) & (vs.EVENT_ID[:, :, vs.itt] == 0) & (arr_itt >= 1))
    cond5 = (vs.EVENT_ID[:, :, vs.itt] >= 1)
    if cond1.any():
        # number of event
        vs.event_id = npx.max(vs.EVENT_ID[:, :, vs.itt])
        # iteration at event start
        vs.event_start = vs.itt
        vs.event_restart = vs.itt
        # iteration at event end
        vs.event_end = vs.itt + settings.nittevent
        if (vs.event_end >= settings.nitt):
            vs.event_end = settings.nitt
        vs.DT_event = update(
            vs.DT_event,
            at[:], 24,
        )
        vs.DT_event = update(
            vs.DT_event,
            at[0:vs.event_end-vs.event_start], vs.DT[vs.event_start:vs.event_end],
        )
        vs.prec_event = update(
            vs.prec_event,
            at[:, :, :], 0,
        )
        vs.prec_event = update(
            vs.prec_event,
            at[:, :, 0:vs.event_end-vs.event_start], npx.where(vs.EVENT_ID == vs.event_id, vs.PREC, 0)[:, :, vs.event_start:vs.event_end],
        )
        # time step during event (10mins or 1 hour)
        vs.dt_event = npx.min(vs.DT_event)
        vs.update(set_event_vars(state))
    if cond2.any():
        vs.event_restart = vs.itt
        vs.update(set_event_vars_start_rainfall_pause(state))
    if cond3.any():
        vs.event_restart = vs.itt
        vs.DT_event = update(
            vs.DT_event,
            at[vs.itt-vs.event_start:vs.event_end-vs.event_start], 0,
        )
        vs.prec_event = update(
            vs.prec_event,
            at[:, :, 0:vs.itt-vs.event_start], 0,
        )
        vs.update(set_event_vars_end_rainfall_pause(state))
    if cond5.any():
        vs.t_event_sum = update_add(
            vs.t_event_sum,
            at[:, :, vs.tau], vs.dt,
        )

    vs.update(calc_inf_mat(state))
    vs.update(calc_inf_mp(state))
    vs.update(calc_inf_sc(state))
    vs.update(calc_inf_rz(state))
    vs.update(calc_inf_ss(state))
    vs.update(calc_inf(state))

    if cond4.any():
        vs.event_id = 0
        vs.update(reset_event_vars(state))


@roger_kernel
def calculate_infiltration_rz_transport_kernel(state):
    """
    Calculates transport of infiltration
    """
    vs = state.variables

    vs.sa_rz = update_add(
        vs.sa_rz,
        at[:, :, vs.tau, 0], vs.inf_mat_rz * vs.maskCatch,
    )

    vs.sa_rz = update_add(
        vs.sa_rz,
        at[:, :, vs.tau, 0], vs.inf_pf_rz * vs.maskCatch,
    )

    return KernelOutput(sa_rz=vs.sa_rz)


@roger_kernel
def calculate_infiltration_rz_transport_iso_kernel(state):
    """
    Calculates isotope transport of infiltration
    """
    vs = state.variables

    # isotope ratio of infiltration
    vs.C_inf_mat_rz = update(
        vs.C_inf_mat_rz,
        at[:, :], npx.where(vs.inf_mat_rz > 0, vs.C_in, npx.NaN) * vs.maskCatch,
    )
    vs.C_inf_pf_rz = update(
        vs.C_inf_pf_rz,
        at[:, :], npx.where(vs.inf_pf_rz > 0, vs.C_in, npx.NaN) * vs.maskCatch,
    )
    vs.sa_rz = update_add(
        vs.sa_rz,
        at[:, :, vs.tau, 0], vs.inf_mat_rz + vs.inf_pf_rz * vs.maskCatch,
    )
    vs.msa_rz = update(
        vs.msa_rz,
        at[:, :, vs.tau, 0], npx.where(vs.inf_mat_rz + vs.inf_pf_rz > 0, vs.C_in, npx.NaN) * vs.maskCatch,
    )

    return KernelOutput(sa_rz=vs.sa_rz, msa_rz=vs.msa_rz, C_inf_mat_rz=vs.C_inf_mat_rz, C_inf_pf_rz=vs.C_inf_pf_rz)


@roger_kernel
def calculate_infiltration_rz_transport_anion_kernel(state):
    """
    Calculates isotope transport of infiltration
    """
    vs = state.variables

    # solute concentration of infiltration
    vs.C_inf_mat_rz = update(
        vs.C_inf_mat_rz,
        at[:, :], npx.where(vs.inf_mat_rz > 0, vs.C_in, 0) * vs.maskCatch,
    )
    vs.C_inf_pf_rz = update(
        vs.C_inf_pf_rz,
        at[:, :], npx.where(vs.inf_pf_rz > 0, vs.C_in, 0) * vs.maskCatch,
    )
    # solute mass of infiltration
    vs.M_inf_mat_rz = update(
        vs.M_inf_mat_rz,
        at[:, :], vs.C_inf_mat_rz * vs.inf_mat_rz * vs.maskCatch,
    )
    vs.M_inf_pf_rz = update(
        vs.C_inf_pf_rz,
        at[:, :], vs.C_inf_pf_rz * vs.inf_pf_rz * vs.maskCatch,
    )
    vs.sa_rz = update_add(
        vs.sa_rz,
        at[:, :, vs.tau, 0], vs.inf_mat_rz + vs.inf_pf_rz * vs.maskCatch,
    )
    vs.msa_rz = update_add(
        vs.msa_rz,
        at[:, :, vs.tau, 0], vs.M_inf_mat_rz + vs.M_inf_pf_rz * vs.maskCatch,
    )

    return KernelOutput(sa_rz=vs.sa_rz, msa_rz=vs.msa_rz, C_inf_mat_rz=vs.C_inf_mat_rz, C_inf_pf_rz=vs.C_inf_pf_rz, M_inf_mat_rz=vs.M_inf_mat_rz, M_inf_pf_rz=vs.M_inf_pf_rz)


@roger_kernel
def calculate_infiltration_ss_transport_kernel(state):
    """
    Calculates travel time of transpiration
    """
    vs = state.variables

    vs.sa_ss = update_add(
        vs.sa_ss,
        at[:, :, vs.tau, 0], vs.inf_mat_ss * vs.maskCatch,
    )

    return KernelOutput(sa_ss=vs.sa_ss)


@roger_kernel
def calculate_infiltration_ss_transport_iso_kernel(state):
    """
    Calculates isotope transport of infiltration
    """
    vs = state.variables

    # isotope ratio of infiltration
    vs.C_inf_pf_ss = update(
        vs.C_inf_pf_ss,
        at[:, :], npx.where(vs.inf_pf_ss > 0, vs.C_in, npx.NaN) * vs.maskCatch,
    )

    vs.sa_ss = update_add(
        vs.sa_ss,
        at[:, :, vs.tau, 0], vs.inf_pf_ss * vs.maskCatch,
    )

    vs.msa_ss = update(
        vs.msa_ss,
        at[:, :, vs.tau, 0], npx.where(vs.inf_pf_ss > 0, vs.C_in, npx.NaN) * vs.maskCatch,
    )

    return KernelOutput(sa_ss=vs.sa_ss, msa_ss=vs.msa_ss, C_inf_pf_ss=vs.C_inf_pf_ss)


@roger_kernel
def calculate_infiltration_ss_transport_anion_kernel(state):
    """
    Calculates isotope transport of infiltration
    """
    vs = state.variables

    # solute concentration of infiltration
    vs.C_inf_pf_ss = update(
        vs.C_inf_pf_ss,
        at[:, :], npx.where(vs.inf_pf_ss > 0, vs.C_in, 0) * vs.maskCatch,
    )
    # solute mass of infiltration
    vs.M_inf_pf_ss = update(
        vs.C_inf_pf_ss,
        at[:, :], vs.C_inf_pf_ss * vs.inf_pf_ss * vs.maskCatch,
    )
    vs.sa_ss = update_add(
        vs.sa_ss,
        at[:, :, vs.tau, 0], vs.inf_pf_ss * vs.maskCatch,
    )
    vs.msa_ss = update_add(
        vs.msa_ss,
        at[:, :, vs.tau, 0], vs.M_inf_pf_ss * vs.maskCatch,
    )

    return KernelOutput(sa_ss=vs.sa_ss, msa_ss=vs.msa_ss, C_inf_pf_ss=vs.C_inf_pf_ss, M_inf_pf_ss=vs.M_inf_pf_ss)


@roger_routine
def calculate_infiltration_rz_transport(state):
    """
    Calculates infiltration transport
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate):
        vs.update(calculate_infiltration_rz_transport_kernel(state))

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(calculate_infiltration_rz_transport_iso_kernel(state))

    if settings.enable_offline_transport and (settings.enable_chloride | settings.enable_bromide | settings.enable_nitrate):
        vs.update(calculate_infiltration_rz_transport_anion_kernel(state))


@roger_routine
def calculate_infiltration_ss_transport(state):
    """
    Calculates infiltration transport
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate):
        vs.update(calculate_infiltration_ss_transport_kernel(state))

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(calculate_infiltration_ss_transport_iso_kernel(state))

    if settings.enable_offline_transport and (settings.enable_chloride | settings.enable_bromide | settings.enable_nitrate):
        vs.update(calculate_infiltration_ss_transport_anion_kernel(state))
