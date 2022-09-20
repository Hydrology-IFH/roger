from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, scan, at


@roger_kernel
def calc_green_ampt_params(state):
    """
    Calculates matrix infiltration parameters
    """
    vs = state.variables

    # threshold intensity (mm/h)
    pi_gr = calc_pi_gr(state)
    vs.pi_gr = update(
        vs.pi_gr,
        at[2:-2, 2:-2], pi_gr[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    # precipitation intensity at saturation (mm/h)
    pi_m = calc_pi_m(state)
    vs.pi_m = update(
        vs.pi_m,
        at[2:-2, 2:-2], pi_m[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    # saturation time (hours)
    t_sat = calc_sat_time(state, pi_m, pi_gr)
    vs.t_sat = update(
        vs.t_sat,
        at[2:-2, 2:-2], t_sat[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    # infiltration at saturation (mm)
    Fs = calc_Fs(state, pi_m)
    vs.Fs = update(
        vs.Fs,
        at[2:-2, 2:-2], Fs[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(pi_gr=vs.pi_gr,
                        pi_m=vs.pi_m,
                        t_sat=vs.t_sat,
                        Fs=vs.Fs,
                        )


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

    mask1 = (vs.pi_m <= vs.prec_event_csum) & (vs.t_event_csum > vs.t_sat) & (vs.t_sat > 0)
    mask2 = (vs.pi_m > vs.prec_event_csum) & (vs.t_event_csum > vs.t_sat) & (vs.t_sat > 0)
    mask3 = (vs.t_sat > vs.t_event_csum - vs.dt) & (vs.t_sat < vs.t_event_csum)
    mask4 = (vs.pi_m > vs.prec_event_csum) & (vs.t_sat <= 0)
    a = update(
        a,
        at[2:-2, 2:-2], vs.ks[2:-2, 2:-2] * (vs.t_event_csum[2:-2, 2:-2] - vs.t_sat[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    b = update(
        b,
        at[2:-2, 2:-2], vs.Fs[2:-2, 2:-2] + 2 * vs.theta_d[2:-2, 2:-2] * vs.wfs[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    l1 = update(
        l1,
        at[2:-2, 2:-2], npx.where(vs.z0[2:-2, 2:-2, vs.tau] > vs.ks[2:-2, 2:-2] * vs.dt, (vs.ks[2:-2, 2:-2] * vs.dt * vs.wfs[2:-2, 2:-2] * vs.theta_d[2:-2, 2:-2]) / (vs.z0[2:-2, 2:-2, vs.tau] - vs.ks[2:-2, 2:-2] * vs.dt), (vs.ks[2:-2, 2:-2] * vs.dt * vs.wfs[2:-2, 2:-2] * vs.theta_d[2:-2, 2:-2]) / (vs.ks[2:-2, 2:-2] * vs.dt)) * vs.maskCatch[2:-2, 2:-2],
    )

    # first wetting front
    # potential matrix infiltration rate after saturation is reached
    vs.inf_mat_pot = update(
        vs.inf_mat_pot,
        at[2:-2, 2:-2], 0,
    )
    vs.inf_mat_pot = update(
        vs.inf_mat_pot,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], (vs.ks[2:-2, 2:-2]*vs.dt/2) * (1 + (1 + 2*b[2:-2, 2:-2]/a[2:-2, 2:-2]) / (1 + (4*b[2:-2, 2:-2]/a[2:-2, 2:-2]) + (4*vs.Fs_t0[2:-2, 2:-2]**2 / a[2:-2, 2:-2]**2))**0.5) * ((100 - vs.sealing[2:-2, 2:-2]) / 100), vs.inf_mat_pot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    # potential matrix infiltration rate before saturation is reached
    vs.inf_mat_pot = update(
        vs.inf_mat_pot,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], vs.ks[2:-2, 2:-2] * vs.dt * (1 + ((vs.wfs[2:-2, 2:-2] * vs.theta_d[2:-2, 2:-2]) / l1[2:-2, 2:-2])) * ((100 - vs.sealing[2:-2, 2:-2]) / 100), vs.inf_mat_pot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    # potential matrix infiltration rate when saturation is reached within
    # time step
    inf_mat_pot_rec = update(
        inf_mat_pot_rec,
        at[2:-2, 2:-2], npx.where(mask3[2:-2, 2:-2], (vs.ks[2:-2, 2:-2]*vs.dt/2) * (1 + (1 + 2*b[2:-2, 2:-2]/a[2:-2, 2:-2]) / (1 + (4*b[2:-2, 2:-2]/a[2:-2, 2:-2]) + (4*vs.Fs_t0[2:-2, 2:-2]**2 / a[2:-2, 2:-2]**2))**0.5), inf_mat_pot_rec[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    inf_mat_pot_sat = update(
        inf_mat_pot_sat,
        at[2:-2, 2:-2], npx.where(mask3[2:-2, 2:-2], vs.z0[2:-2, 2:-2, vs.tau] * (vs.t_sat[2:-2, 2:-2] - (vs.t_event_csum[2:-2, 2:-2] - vs.dt)), inf_mat_pot_sat[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.inf_mat_pot = update(
        vs.inf_mat_pot,
        at[2:-2, 2:-2], npx.where(mask3[2:-2, 2:-2], inf_mat_pot_sat[2:-2, 2:-2] + inf_mat_pot_rec[2:-2, 2:-2] * ((100 - vs.sealing[2:-2, 2:-2]) / 100), vs.inf_mat_pot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.inf_mat_pot = update(
        vs.inf_mat_pot,
        at[2:-2, 2:-2], npx.where(mask4[2:-2, 2:-2], vs.pi_gr[2:-2, 2:-2] * ((100 - vs.sealing[2:-2, 2:-2]) / 100), vs.inf_mat_pot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    inf_mat_pot = update(
        inf_mat_pot,
        at[2:-2, 2:-2], vs.inf_mat_pot[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # matrix infiltration
    mask7 = (vs.z0[:, :, vs.tau] < vs.inf_mat_pot)
    mask8 = (vs.z0[:, :, vs.tau] >= vs.inf_mat_pot)
    vs.inf_mat = update(
        vs.inf_mat,
        at[2:-2, 2:-2], npx.where(mask7[2:-2, 2:-2], vs.z0[2:-2, 2:-2, vs.tau], vs.inf_mat[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.inf_mat = update(
        vs.inf_mat,
        at[2:-2, 2:-2], npx.where(mask8[2:-2, 2:-2], vs.inf_mat_pot[2:-2, 2:-2], vs.inf_mat[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.inf_mat = update(
        vs.inf_mat,
        at[2:-2, 2:-2], npx.where(vs.inf_mat[2:-2, 2:-2] < 0, 0, vs.inf_mat[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # matrix infiltration
    mask9 = (vs.z0[:, :, vs.tau] < vs.inf_mat_pot)
    mask10 = (vs.z0[:, :, vs.tau] >= vs.inf_mat_pot)
    vs.inf_mat = update(
        vs.inf_mat,
        at[2:-2, 2:-2], npx.where(mask9[2:-2, 2:-2], vs.z0[2:-2, 2:-2, vs.tau], vs.inf_mat[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.inf_mat = update(
        vs.inf_mat,
        at[2:-2, 2:-2], npx.where(mask10[2:-2, 2:-2], vs.inf_mat_pot[2:-2, 2:-2], vs.inf_mat[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.inf_mat = update(
        vs.inf_mat,
        at[2:-2, 2:-2], npx.where(vs.inf_mat[2:-2, 2:-2] < 0, 0, vs.inf_mat[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # update cumulated infiltration while event
    vs.inf_mat_event_csum = update_add(
        vs.inf_mat_event_csum,
        at[2:-2, 2:-2], vs.inf_mat[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.inf_mat_pot_event_csum = update_add(
        vs.inf_mat_pot_event_csum,
        at[2:-2, 2:-2], vs.inf_mat_pot[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # wetting front depth
    # change in wetting front depth
    dz_wf = allocate(state.dimensions, ("x", "y"))
    mask11 = (vs.no_wf == 1)
    mask12 = (vs.no_wf == 2)
    dz_wf = update(
        dz_wf,
        at[2:-2, 2:-2], (vs.inf_mat[2:-2, 2:-2] / vs.theta_d_t0[2:-2, 2:-2]) * mask11[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    dz_wf = update(
        dz_wf,
        at[2:-2, 2:-2], npx.where(mask12[2:-2, 2:-2], vs.inf_mat[2:-2, 2:-2] / vs.theta_d[2:-2, 2:-2], dz_wf[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.z_wf_t0 = update_add(
        vs.z_wf_t0,
        at[2:-2, 2:-2, vs.tau], npx.where(npx.isfinite(dz_wf[2:-2, 2:-2]), dz_wf[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_wf_t1 = update_add(
        vs.z_wf_t1,
        at[2:-2, 2:-2, vs.tau], npx.where(npx.isfinite(dz_wf[2:-2, 2:-2]), dz_wf[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )
    mask13 = (vs.z_wf_t0[:, :, vs.tau] > vs.z_soil)
    vs.z_wf_t0 = update(
        vs.z_wf_t0,
        at[2:-2, 2:-2, vs.tau], npx.where(mask13[2:-2, 2:-2], vs.z_soil[2:-2, 2:-2], vs.z_wf_t0[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    mask19 = (vs.z_wf_t1[:, :, vs.tau] > vs.z_soil)
    vs.z_wf_t1 = update(
        vs.z_wf_t1,
        at[2:-2, 2:-2, vs.tau], npx.where(mask19[2:-2, 2:-2], vs.z_soil[2:-2, 2:-2], vs.z_wf_t1[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    # hortonian overland flow after matrix infiltration
    vs.z0 = update_add(
        vs.z0,
        at[2:-2, 2:-2, vs.tau], - vs.inf_mat[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z0 = update(
        vs.z0,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z0[2:-2, 2:-2, vs.tau] < 0, 0, vs.z0[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    # change in potential wetting front depth while rainfall pause
    dz_wf_t0 = allocate(state.dimensions, ("x", "y"))
    dz_wf_t0 = update(
        dz_wf_t0,
        at[2:-2, 2:-2], npx.where((vs.z_wf_fc[2:-2, 2:-2] > 0) & (vs.rain_ground[2:-2, 2:-2] <= 0) & (vs.no_wf[2:-2, 2:-2] == 1), vs.inf_mat_pot[2:-2, 2:-2] / vs.theta_d_t0[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )
    # wetting front moves downwards
    vs.z_wf_t0 = update_add(
        vs.z_wf_t0,
        at[2:-2, 2:-2, vs.tau], npx.where(npx.isfinite(dz_wf_t0[2:-2, 2:-2]), dz_wf_t0[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )
    mask17 = (vs.z_wf_t0[:, :, vs.tau] > vs.z_wf_fc) & (vs.z_wf_fc > 0)
    vs.z_wf_t0 = update(
        vs.z_wf_t0,
        at[2:-2, 2:-2, vs.tau], npx.where(mask17[2:-2, 2:-2], vs.z_wf_fc[2:-2, 2:-2], vs.z_wf_t0[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_wf_t0 = update(
        vs.z_wf_t0,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_wf_t0[2:-2, 2:-2, vs.tau] > vs.z_soil[2:-2, 2:-2], vs.z_soil[2:-2, 2:-2], vs.z_wf_t0[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    dz_wf_t1 = allocate(state.dimensions, ("x", "y"))
    dz_wf_t1 = update(
        dz_wf_t1,
        at[2:-2, 2:-2], npx.where((vs.z_wf_fc[2:-2, 2:-2] > 0) & (vs.rain_ground[2:-2, 2:-2] <= 0) & (vs.no_wf[2:-2, 2:-2] == 2), vs.inf_mat_pot[2:-2, 2:-2] / vs.theta_d[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )
    # wetting front moves downwards
    vs.z_wf_t1 = update_add(
        vs.z_wf_t1,
        at[2:-2, 2:-2, vs.tau], npx.where(npx.isfinite(dz_wf_t1[2:-2, 2:-2]), dz_wf_t1[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )
    mask18 = (vs.z_wf_t1[:, :, vs.tau] > vs.z_wf_fc) & (vs.z_wf_fc > 0)
    vs.z_wf_t1 = update(
        vs.z_wf_t1,
        at[2:-2, 2:-2, vs.tau], npx.where(mask18[2:-2, 2:-2], vs.z_wf_fc[2:-2, 2:-2], vs.z_wf_t1[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_wf_t1 = update(
        vs.z_wf_t1,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_wf_t1[2:-2, 2:-2, vs.tau] > vs.z_soil[2:-2, 2:-2], vs.z_soil[2:-2, 2:-2], vs.z_wf_t1[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    # update wetting front depth and soil moisture deficit
    mask14 = (vs.z_wf_t0[:, :, vs.tau] >= vs.z_wf_t1[:, :, vs.tau]) & (vs.z_wf_t1[:, :, vs.tau] <= 0)
    mask15 = (vs.z_wf_t0[:, :, vs.tau] > vs.z_wf_t1[:, :, vs.tau]) & (vs.z_wf_t1[:, :, vs.tau] > 0)
    mask20 = (vs.z_wf_t0[:, :, vs.tau] <= vs.z_wf_t1[:, :, vs.tau]) & (vs.z_wf_t1[:, :, vs.tau] > 0)
    vs.z_wf = update(
        vs.z_wf,
        at[2:-2, 2:-2, vs.tau], npx.where(mask14[2:-2, 2:-2], vs.z_wf_t0[2:-2, 2:-2, vs.tau], vs.z_wf[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.theta_d = update(
        vs.theta_d,
        at[2:-2, 2:-2], npx.where(mask14[2:-2, 2:-2], vs.theta_d_t0[2:-2, 2:-2], vs.theta_d[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.theta_d_rel = update(
        vs.theta_d_rel,
        at[2:-2, 2:-2], npx.where(mask14[2:-2, 2:-2], vs.theta_d_rel_t0[2:-2, 2:-2], vs.theta_d_rel[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_wf = update(
        vs.z_wf,
        at[2:-2, 2:-2, vs.taum1], npx.where(mask15[2:-2, 2:-2], 0, vs.z_wf[2:-2, 2:-2, vs.taum1]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_wf = update(
        vs.z_wf,
        at[2:-2, 2:-2, vs.tau], npx.where(mask15[2:-2, 2:-2], vs.z_wf_t1[2:-2, 2:-2, vs.tau], vs.z_wf[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.no_wf = update(
        vs.no_wf,
        at[2:-2, 2:-2], npx.where(mask20[2:-2, 2:-2], 1, vs.no_wf[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_wf = update(
        vs.z_wf,
        at[2:-2, 2:-2, vs.tau], npx.where(mask20[2:-2, 2:-2], vs.z_wf_t0[2:-2, 2:-2, vs.tau], vs.z_wf[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.theta_d = update(
        vs.theta_d,
        at[2:-2, 2:-2], npx.where(mask20[2:-2, 2:-2], vs.theta_d_t0[2:-2, 2:-2], vs.theta_d[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.theta_d_rel = update(
        vs.theta_d_rel,
        at[2:-2, 2:-2], npx.where(mask20[2:-2, 2:-2], vs.theta_d_rel_t0[2:-2, 2:-2], vs.theta_d_rel[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    mask16 = (vs.z_wf[:, :, vs.tau] > vs.z_soil)
    vs.z_wf = update(
        vs.z_wf,
        at[2:-2, 2:-2, vs.tau], npx.where(mask16[2:-2, 2:-2], vs.z_soil[2:-2, 2:-2], vs.z_wf[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    mask17 = (vs.theta_d_t1 <= 0)
    vs.theta_d = update(
        vs.theta_d,
        at[2:-2, 2:-2], npx.where(mask17[2:-2, 2:-2], vs.theta_d_t0[2:-2, 2:-2], vs.theta_d[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(inf_mat_pot=vs.inf_mat_pot, inf_mat=vs.inf_mat, z0=vs.z0, z_wf=vs.z_wf, z_wf_t0=vs.z_wf_t0, z_wf_t1=vs.z_wf_t1, theta_d_rel=vs.theta_d_rel, theta_d=vs.theta_d)


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
        at[2:-2, 2:-2], npx.where(vs.no_wf[2:-2, 2:-2] == 1, 0, vs.z_wf_t0[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    z_wf = update(
        z_wf,
        at[2:-2, 2:-2], npx.where(vs.no_wf[2:-2, 2:-2] == 2, 0, vs.z_wf_t1[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    z_wf_m1 = update(
        z_wf,
        at[2:-2, 2:-2], npx.where(vs.no_wf[2:-2, 2:-2] == 1, 0, vs.z_wf_t0[2:-2, 2:-2, vs.taum1]) * vs.maskCatch[2:-2, 2:-2],
    )
    z_wf_m1 = update(
        z_wf,
        at[2:-2, 2:-2], npx.where(vs.no_wf[2:-2, 2:-2] == 2, 0, vs.z_wf_t1[2:-2, 2:-2, vs.taum1]) * vs.maskCatch[2:-2, 2:-2],
    )

    # length of non saturated vertical macropores at beginning of time step (in mm)
    vs.lmpv_non_sat = update(
        vs.lmpv_non_sat,
        at[2:-2, 2:-2], vs.lmpv[2:-2, 2:-2] - z_wf[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.lmpv_non_sat = update(
        vs.lmpv_non_sat,
        at[2:-2, 2:-2], npx.where(vs.lmpv_non_sat[2:-2, 2:-2] < 0, 0, vs.lmpv_non_sat[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # delta of wetting front depth (in mm)
    dz_wf = update(
        dz_wf,
        at[2:-2, 2:-2], z_wf[2:-2, 2:-2] - z_wf_m1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    mask1 = (z_wf >= vs.lmpv)
    dz_wf = update(
        dz_wf,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], vs.lmpv_non_sat[2:-2, 2:-2], dz_wf[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    dz_wf = update(
        dz_wf,
        at[2:-2, 2:-2], npx.where(vs.lmpv_non_sat[2:-2, 2:-2] <= 0, 0, dz_wf[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    dz_wf = update(
        dz_wf,
        at[2:-2, 2:-2], npx.where(dz_wf[2:-2, 2:-2] <= 0, 0, dz_wf[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # length of non-saturated macropores at beginning of time step (in mm)
    vs.lmpv_non_sat = update(
        vs.lmpv_non_sat,
        at[2:-2, 2:-2], vs.lmpv[2:-2, 2:-2] - vs.z_wf[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.lmpv_non_sat = update(
        vs.lmpv_non_sat,
        at[2:-2, 2:-2], npx.where(vs.lmpv_non_sat[2:-2, 2:-2] < 0, 0, vs.lmpv_non_sat[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # determine computation steps. additional computation steps for hourly
    # time step.
    computation_steps = npx.int64(npx.round(vs.dt / (1 / 5), 0))  # based on hours

    vs.lmpv_non_sat = update(
        vs.lmpv_non_sat,
        at[2:-2, 2:-2], npx.where(computation_steps == 1, vs.lmpv_non_sat[2:-2, 2:-2] + dz_wf[2:-2, 2:-2] / 1.39, vs.lmpv_non_sat[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
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
    # 10 = z0_di
    # 11 = inf_mp
    # 12 = inf_mp_pot
    # 13 = inf_mp_event_csum
    # 14 = t
    # 15 = y
    loop_arr = allocate(state.dimensions, ("x", "y", 16))

    loop_arr = update(
        loop_arr,
        at[2:-2, 2:-2, 2], (settings.r_mp / 2) * vs.maskCatch[2:-2, 2:-2],
    )
    loop_arr = update(
        loop_arr,
        at[2:-2, 2:-2, 4], vs.theta_d[2:-2, 2:-2] * settings.r_mp ** 2 * vs.maskCatch[2:-2, 2:-2],
    )
    loop_arr = update(
        loop_arr,
        at[2:-2, 2:-2, 13], vs.inf_mp_event_csum[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    loop_arr = update(
        loop_arr,
        at[2:-2, 2:-2, 3], vs.y_mp[2:-2, 2:-2, vs.taum1] * vs.maskCatch[2:-2, 2:-2],
    )

    def loop_body(carry, i):
        vs, settings, loop_arr = carry

        # determine computation steps. additional computation steps for hourly
        # time step.
        computation_steps = npx.int64(npx.round(vs.dt / (1 / 5), 0))  # based on hours
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 10], vs.z0[2:-2, 2:-2, vs.tau] * (vs.mp_drain_area[2:-2, 2:-2] / computation_steps) * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update_add(
            loop_arr,
            at[2:-2, 2:-2, 14], (vs.dt / computation_steps) * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 7], vs.ks[2:-2, 2:-2] * vs.wfs[2:-2, 2:-2] * loop_arr[2:-2, 2:-2, 14] * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 7], npx.where(npx.isnan(loop_arr[2:-2, 2:-2, 7]), 0, loop_arr[2:-2, 2:-2, 7]) * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 5], (6**0.5 * 2 * (loop_arr[2:-2, 2:-2, 7]*(6*loop_arr[2:-2, 2:-2, 7] - loop_arr[2:-2, 2:-2, 4]))**0.5) * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 5], npx.where(npx.isnan(loop_arr[2:-2, 2:-2, 5]), 0, loop_arr[2:-2, 2:-2, 5]) * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 6], (settings.r_mp*vs.theta_d[2:-2, 2:-2]**2) * (12*loop_arr[2:-2, 2:-2, 7] - loop_arr[2:-2, 2:-2, 4] + loop_arr[2:-2, 2:-2, 5]) * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 6], npx.where(npx.isnan(loop_arr[2:-2, 2:-2, 6]), 0, loop_arr[2:-2, 2:-2, 6]) * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 6], npx.where(loop_arr[2:-2, 2:-2, 6] <= 0, 0, loop_arr[2:-2, 2:-2, 6]) * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 0], ((loop_arr[2:-2, 2:-2, 6]**(1/3)) / vs.theta_d[2:-2, 2:-2]) * 0.5 * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 1], (loop_arr[2:-2, 2:-2, 4] / (loop_arr[2:-2, 2:-2, 6]**(1/3))) * 0.5 * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 15], (loop_arr[2:-2, 2:-2, 0] + loop_arr[2:-2, 2:-2, 1] + loop_arr[2:-2, 2:-2, 2]) * vs.maskCatch[2:-2, 2:-2],
        )
        # potential radial length of macropore wetting front
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 15], npx.where(loop_arr[2:-2, 2:-2, 15] < settings.r_mp, settings.r_mp, loop_arr[2:-2, 2:-2, 15]) * vs.maskCatch[2:-2, 2:-2],
        )
        # constrain the potential radial length of macropore wetting front
        # to the radial length at beginning of computation step
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 15], npx.where(loop_arr[2:-2, 2:-2, 15] < loop_arr[2:-2, 2:-2, 3], loop_arr[2:-2, 2:-2, 3], loop_arr[2:-2, 2:-2, 15]) * vs.maskCatch[2:-2, 2:-2],
        )
        # potential macropore infiltration (in mm/dt)
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 8], (settings.pi * (loop_arr[2:-2, 2:-2, 15]**2 - loop_arr[2:-2, 2:-2, 3]**2) * vs.lmpv_non_sat[2:-2, 2:-2] * vs.theta_d[2:-2, 2:-2] * vs.dmpv[2:-2, 2:-2] * 1e-06) * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update_add(
            loop_arr,
            at[2:-2, 2:-2, 12], loop_arr[2:-2, 2:-2, 8] * vs.maskCatch[2:-2, 2:-2],
        )
        # constrain macropore infiltration with matrix infiltration excess
        mask1 = (loop_arr[:, :, 8] > loop_arr[:, :, 10])
        # actual macropore infiltration
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 9], loop_arr[2:-2, 2:-2, 8] * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 9], npx.where(mask1[2:-2, 2:-2], loop_arr[2:-2, 2:-2, 10], loop_arr[2:-2, 2:-2, 9]) * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 9], npx.where(vs.lmpv_non_sat[2:-2, 2:-2] == 0, 0, loop_arr[2:-2, 2:-2, 9]) * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update_add(
            loop_arr,
            at[2:-2, 2:-2, 11], loop_arr[2:-2, 2:-2, 9] * vs.maskCatch[2:-2, 2:-2],
        )

        # calculate radial length of macropore wetting front
        loop_arr = update_add(
            loop_arr,
            at[2:-2, 2:-2, 13], loop_arr[2:-2, 2:-2, 9] * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 15], settings.r_mp + ((loop_arr[2:-2, 2:-2, 13] / (vs.dmpv[2:-2, 2:-2] * vs.theta_d[2:-2, 2:-2])) / settings.pi)**.5 * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 15], npx.where(loop_arr[2:-2, 2:-2, 15] < settings.r_mp, settings.r_mp, loop_arr[2:-2, 2:-2, 15]) * vs.maskCatch[2:-2, 2:-2],
        )

        # duration of macropore infiltration
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 14], vs.theta_d[2:-2, 2:-2] / (vs.ks[2:-2, 2:-2] * vs.wfs[2:-2, 2:-2] * settings.r_mp) * (loop_arr[2:-2, 2:-2, 15]**3 / 3. - loop_arr[2:-2, 2:-2, 15]**2 * settings.r_mp / 2. + settings.r_mp**3 / 6.) * vs.maskCatch[2:-2, 2:-2],
        )

        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 11], npx.where(loop_arr[2:-2, 2:-2, 11] < 0, 0, loop_arr[2:-2, 2:-2, 11]) * vs.maskCatch[2:-2, 2:-2],
        )

        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 3], loop_arr[2:-2, 2:-2, 15] * vs.maskCatch[2:-2, 2:-2],
        )

        carry = (vs, settings, loop_arr)

        return carry, None

    steps = npx.arange(0, computation_steps)
    carry = (vs, settings, loop_arr)
    res, _ = scan(loop_body, carry, steps)

    loop_arr = res[2]

    vs.y_mp = update(
        vs.y_mp,
        at[2:-2, 2:-2, vs.tau], loop_arr[2:-2, 2:-2, 15] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.y_mp = update(
        vs.y_mp,
        at[2:-2, 2:-2, vs.tau], npx.where(npx.isnan(vs.y_mp[2:-2, 2:-2, vs.tau]), 0, vs.y_mp[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.inf_mp = update(
        vs.inf_mp,
        at[2:-2, 2:-2], loop_arr[2:-2, 2:-2, 11] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.inf_mp = update(
        vs.inf_mp,
        at[2:-2, 2:-2], npx.where(npx.isnan(vs.inf_mp[2:-2, 2:-2]), 0, vs.inf_mp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # macropore infiltration into root zone
    rz_share_mp = allocate(state.dimensions, ("x", "y"))
    rz_share_mp = update(
        rz_share_mp,
        at[2:-2, 2:-2], npx.where(vs.lmpv_non_sat[2:-2, 2:-2] > 0, 1. - (vs.lmpv[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau]) / vs.lmpv_non_sat[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )
    rz_share_mp = update(
        rz_share_mp,
        at[2:-2, 2:-2], npx.where(vs.lmpv[2:-2, 2:-2] <= vs.z_root[2:-2, 2:-2, vs.tau], 1, rz_share_mp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    rz_share_mp = update(
        rz_share_mp,
        at[2:-2, 2:-2], npx.where(vs.lmpv_non_sat[2:-2, 2:-2] <= vs.lmpv[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau], 0, rz_share_mp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.inf_mp_rz = update(
        vs.inf_mp_rz,
        at[2:-2, 2:-2], vs.inf_mp[2:-2, 2:-2] * rz_share_mp[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.inf_mp_rz = update(
        vs.inf_mp_rz,
        at[2:-2, 2:-2], npx.where((vs.inf_mp_rz[2:-2, 2:-2] > (vs.S_ac_rz[2:-2, 2:-2] + vs.S_ufc_rz[2:-2, 2:-2]) - (vs.S_lp_rz[2:-2, 2:-2] + vs.S_fp_rz[2:-2, 2:-2])) & ((vs.S_ac_rz[2:-2, 2:-2] + vs.S_ufc_rz[2:-2, 2:-2]) - (vs.S_lp_rz[2:-2, 2:-2] + vs.S_fp_rz[2:-2, 2:-2]) > 0), (vs.S_ac_rz[2:-2, 2:-2] + vs.S_ufc_rz[2:-2, 2:-2]) - (vs.S_lp_rz[2:-2, 2:-2] + vs.S_fp_rz[2:-2, 2:-2]), vs.inf_mp_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # macropore infiltration into subsoil
    rz_share_mp = allocate(state.dimensions, ("x", "y"))
    rz_share_mp = update(
        rz_share_mp,
        at[2:-2, 2:-2], npx.where(vs.lmpv_non_sat[2:-2, 2:-2] > 0, 1. - (vs.lmpv[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau]) / vs.lmpv_non_sat[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )
    rz_share_mp = update(
        rz_share_mp,
        at[2:-2, 2:-2], npx.where(vs.lmpv[2:-2, 2:-2] <= vs.z_root[2:-2, 2:-2, vs.tau], 1, rz_share_mp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    rz_share_mp = update(
        rz_share_mp,
        at[2:-2, 2:-2], npx.where(vs.lmpv_non_sat[2:-2, 2:-2] <= vs.lmpv[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau], 0, rz_share_mp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.inf_mp_ss = update(
        vs.inf_mp_ss,
        at[2:-2, 2:-2], vs.inf_mp[2:-2, 2:-2] * (1 - rz_share_mp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.inf_mp_ss = update(
        vs.inf_mp_ss,
        at[2:-2, 2:-2], npx.where((vs.inf_mp_ss[2:-2, 2:-2] > (vs.S_ac_ss[2:-2, 2:-2] + vs.S_ufc_ss[2:-2, 2:-2]) - (vs.S_lp_ss[2:-2, 2:-2] + vs.S_fp_ss[2:-2, 2:-2])) & ((vs.S_ac_ss[2:-2, 2:-2] + vs.S_ufc_ss[2:-2, 2:-2]) - (vs.S_lp_ss[2:-2, 2:-2] + vs.S_fp_ss[2:-2, 2:-2]) > 0), (vs.S_ac_ss[2:-2, 2:-2] + vs.S_ufc_ss[2:-2, 2:-2]) - (vs.S_lp_ss[2:-2, 2:-2] + vs.S_fp_ss[2:-2, 2:-2]), vs.inf_mp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.inf_ss = update(
        vs.inf_ss,
        at[2:-2, 2:-2], vs.inf_mp_ss[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # update subsoil storage after macropore infiltration
    vs.S_fp_ss = update_add(
        vs.S_fp_ss,
        at[2:-2, 2:-2], vs.inf_ss[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # subsoil fine pore excess fills subsoil large pores
    mask = (vs.S_fp_ss > vs.S_ufc_ss)
    vs.S_lp_ss = update_add(
        vs.S_lp_ss,
        at[2:-2, 2:-2], (vs.S_fp_ss[2:-2, 2:-2] - vs.S_ufc_ss[2:-2, 2:-2]) * mask[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_fp_ss = update(
        vs.S_fp_ss,
        at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.S_ufc_ss[2:-2, 2:-2], vs.S_fp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.inf_mp = update(
        vs.inf_mp,
        at[2:-2, 2:-2], vs.inf_mp_rz[2:-2, 2:-2] + vs.inf_mp_ss[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.inf_mp_event_csum = update_add(
        vs.inf_mp_event_csum,
        at[2:-2, 2:-2], vs.inf_mp[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # potential hortonian overland flow after matrix and macropore
    # infiltration
    vs.z0 = update_add(
        vs.z0,
        at[2:-2, 2:-2, vs.tau], - vs.inf_mp[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z0 = update(
        vs.z0,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z0[2:-2, 2:-2, vs.tau] < 0, 0, vs.z0[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    # lower boundary condition of vertical macropores
    if settings.enable_macropore_lower_boundary_condition:
        npx.where((vs.z0[2:-2, 2:-2, vs.tau] > vs.ks[2:-2, 2:-2] * vs.dt * vs.mp_drain_area[2:-2, 2:-2]), vs.ks[2:-2, 2:-2] * vs.dt * vs.mp_drain_area[2:-2, 2:-2], vs.z0[2:-2, 2:-2, vs.tau])
        mask_lbc = (vs.z_wf[2:-2, 2:-2, vs.tau] > vs.lmpv[2:-2, 2:-2])
        vs.inf_mp = update_add(
            vs.inf_mp,
            at[2:-2, 2:-2], npx.where(mask_lbc, npx.where((vs.z0[2:-2, 2:-2, vs.tau] > vs.ks[2:-2, 2:-2] * vs.dt * vs.mp_drain_area[2:-2, 2:-2]), vs.ks[2:-2, 2:-2] * vs.dt * vs.mp_drain_area[2:-2, 2:-2], vs.z0[2:-2, 2:-2, vs.tau]), 0) * vs.maskCatch[2:-2, 2:-2],
        )
        # limit macropore infiltration to pores which are not filled
        vs.inf_mp = update(
            vs.inf_mp,
            at[2:-2, 2:-2], npx.where((vs.z_wf[2:-2, 2:-2, vs.tau] > vs.z_root[2:-2, 2:-2, vs.tau]) & (vs.inf_mp[2:-2, 2:-2] > vs.S_ac_ss[2:-2, 2:-2] - vs.S_lp_ss[2:-2, 2:-2]) & (vs.S_ac_ss[2:-2, 2:-2] - vs.S_lp_ss[2:-2, 2:-2] > 0), vs.S_ac_ss[2:-2, 2:-2] - vs.S_lp_ss[2:-2, 2:-2], vs.inf_mp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.z0 = update_add(
            vs.z0,
            at[2:-2, 2:-2, vs.tau], -npx.where(mask_lbc, npx.where((vs.z0[2:-2, 2:-2, vs.tau] > vs.ks[2:-2, 2:-2] * vs.dt * vs.mp_drain_area[2:-2, 2:-2]), vs.ks[2:-2, 2:-2] * vs.dt * vs.mp_drain_area[2:-2, 2:-2], vs.z0[2:-2, 2:-2, vs.tau]), 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.z0 = update(
            vs.z0,
            at[2:-2, 2:-2, vs.tau], npx.where(vs.z0[2:-2, 2:-2, vs.tau] < 0, 0, vs.z0[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
        )

    return KernelOutput(inf_mp=vs.inf_mp, inf_mp_event_csum=vs.inf_mp_event_csum, y_mp=vs.y_mp, z0=vs.z0, inf_mp_ss=vs.inf_mp_ss, inf_ss=vs.inf_ss, S_fp_ss=vs.S_fp_ss, S_lp_ss=vs.S_lp_ss, inf_mp_rz=vs.inf_mp_rz)


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
        at[2:-2, 2:-2], npx.where(vs.no_wf[2:-2, 2:-2] == 1, 0, vs.z_wf_t0[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    z_wf = update(
        z_wf,
        at[2:-2, 2:-2], npx.where(vs.no_wf[2:-2, 2:-2] == 2, 0, vs.z_wf_t1[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    z_wf_m1 = update(
        z_wf,
        at[2:-2, 2:-2], npx.where(vs.no_wf[2:-2, 2:-2] == 1, 0, vs.z_wf_t0[2:-2, 2:-2, vs.taum1]) * vs.maskCatch[2:-2, 2:-2],
    )
    z_wf_m1 = update(
        z_wf,
        at[2:-2, 2:-2], npx.where(vs.no_wf[2:-2, 2:-2] == 2, 0, vs.z_wf_t1[2:-2, 2:-2, vs.taum1]) * vs.maskCatch[2:-2, 2:-2],
    )

    # length of non saturated shrinkage crack at beginning of time step (in mm)
    vs.z_sc_non_sat = update(
        vs.z_sc_non_sat,
        at[2:-2, 2:-2], vs.z_sc[2:-2, 2:-2] - z_wf[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sc_non_sat = update(
        vs.z_sc_non_sat,
        at[2:-2, 2:-2], npx.where(vs.z_sc_non_sat[2:-2, 2:-2] < 0, 0, vs.z_sc_non_sat[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # delta of wetting front depth (in mm)
    dz_wf = update(
        dz_wf,
        at[2:-2, 2:-2], z_wf[2:-2, 2:-2] - z_wf_m1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    mask1 = (z_wf >= vs.z_sc)
    dz_wf = update(
        dz_wf,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], vs.z_sc_non_sat[2:-2, 2:-2], dz_wf[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    dz_wf = update(
        dz_wf,
        at[2:-2, 2:-2], npx.where(vs.z_sc_non_sat[2:-2, 2:-2] <= 0, 0, dz_wf[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    dz_wf = update(
        dz_wf,
        at[2:-2, 2:-2], npx.where(dz_wf[2:-2, 2:-2] <= 0, 0, dz_wf[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # length of non-saturated macropores at beginning of time step (in mm)
    vs.z_sc_non_sat = update(
        vs.z_sc_non_sat,
        at[2:-2, 2:-2], vs.z_sc[2:-2, 2:-2] - vs.z_wf[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sc_non_sat = update(
        vs.z_sc_non_sat,
        at[2:-2, 2:-2], npx.where(vs.z_sc_non_sat[2:-2, 2:-2] < 0, 0, vs.z_sc_non_sat[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # determine computation steps. additional computation steps for hourly
    # time step.
    computation_steps = npx.int64(npx.round(vs.dt / (1 / 5), 0))  # based on hours

    vs.z_sc_non_sat = update(
        vs.z_sc_non_sat,
        at[2:-2, 2:-2], npx.where(computation_steps == 1, vs.z_sc_non_sat[2:-2, 2:-2] + dz_wf[2:-2, 2:-2] / 1.39, vs.z_sc_non_sat[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # variables to calculate y(t)
    # 0 = y
    # 1 = ym1
    # 2 = inf_sc_pot_di
    # 3 = inf_sc_di
    # 4 = z0_di
    # 5 = inf_sc_event_csum
    # 6 = t
    # 7 = inf_sc
    loop_arr = allocate(state.dimensions, ("x", "y", 8))

    loop_arr = update(
        loop_arr,
        at[2:-2, 2:-2, 5], vs.inf_sc_event_csum[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    loop_arr = update(
        loop_arr,
        at[2:-2, 2:-2, 1], vs.y_sc[2:-2, 2:-2, vs.taum1] * vs.maskCatch[2:-2, 2:-2],
    )

    def loop_body(carry, i):
        vs, settings, loop_arr = carry
        # determine computation steps. additional computation steps for hourly
        # time step.
        computation_steps = npx.int64(npx.round(vs.dt / (1 / 5), 0))  # based on hours
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 4], (vs.z0[2:-2, 2:-2, vs.tau] / computation_steps) * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update_add(
            loop_arr,
            at[2:-2, 2:-2, 6], (vs.dt / computation_steps) * vs.maskCatch[2:-2, 2:-2],
            )
        # potential horizontal length of shrinkage crack wetting front
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 0], (((vs.ks[2:-2, 2:-2] * vs.wfs[2:-2, 2:-2] * loop_arr[2:-2, 2:-2, 6] * 2) / vs.theta_d[2:-2, 2:-2])**0.5) * vs.maskCatch[2:-2, 2:-2],
        )
        # potential horizontal shrinkage crack infiltration
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 2], ((vs.z_sc_non_sat[2:-2, 2:-2] * vs.theta_d[2:-2, 2:-2] * settings.l_sc) * (loop_arr[2:-2, 2:-2, 0] - loop_arr[2:-2, 2:-2, 1]) * 1e-06) * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 2], npx.where(loop_arr[2:-2, 2:-2, 2] <= 0, 0, loop_arr[2:-2, 2:-2, 2]) * vs.maskCatch[2:-2, 2:-2],
        )
        # actual horizontal shrinkage crack infiltration
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 3], npx.where(loop_arr[2:-2, 2:-2, 2] > loop_arr[2:-2, 2:-2, 4], loop_arr[2:-2, 2:-2, 4], loop_arr[2:-2, 2:-2, 2]) * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 3], npx.where(vs.z_sc_non_sat[2:-2, 2:-2] <= 0, 0, loop_arr[2:-2, 2:-2, 3]) * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update_add(
            loop_arr,
            at[2:-2, 2:-2, 7], loop_arr[2:-2, 2:-2, 3] * vs.maskCatch[2:-2, 2:-2],
        )
        loop_arr = update_add(
            loop_arr,
            at[2:-2, 2:-2, 5], loop_arr[2:-2, 2:-2, 3] * vs.maskCatch[2:-2, 2:-2],
        )
        # actual horizontal length of shrinkage crack wetting front
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 0], (loop_arr[2:-2, 2:-2, 5] / settings.l_sc / 2) * vs.maskCatch[2:-2, 2:-2],
        )
        # duration of shrinkage crack infiltration
        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 6], ((loop_arr[2:-2, 2:-2, 1]**2 * vs.theta_d[2:-2, 2:-2]) / (vs.ks[2:-2, 2:-2] * vs.wfs[2:-2, 2:-2] * 2)) * vs.maskCatch[2:-2, 2:-2],
        )

        loop_arr = update(
            loop_arr,
            at[2:-2, 2:-2, 1], loop_arr[2:-2, 2:-2, 0] * vs.maskCatch[2:-2, 2:-2],
        )

        carry = (vs, settings, loop_arr)

        return carry, None

    steps = npx.arange(0, computation_steps)
    carry = (vs, settings, loop_arr)
    res, _ = scan(loop_body, carry, steps)
    loop_arr = res[2]

    vs.y_sc = update(
        vs.y_sc,
        at[2:-2, 2:-2, vs.tau], loop_arr[2:-2, 2:-2, 0] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.inf_sc = update(
        vs.inf_sc,
        at[2:-2, 2:-2], loop_arr[2:-2, 2:-2, 7] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.inf_sc_event_csum = update_add(
        vs.inf_sc_event_csum,
        at[2:-2, 2:-2], vs.inf_sc[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # potential hortonian overland flow after matrix and macropore
    # infiltration and shrinkage crack infiltration
    vs.z0 = update_add(
        vs.z0,
        at[2:-2, 2:-2, vs.tau], - vs.inf_sc[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z0 = update(
        vs.z0,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z0[2:-2, 2:-2, vs.tau] < 0, 0, vs.z0[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(inf_sc=vs.inf_sc, inf_sc_event_csum=vs.inf_sc_event_csum, y_sc=vs.y_sc, z0=vs.z0, z_sc_non_sat=vs.z_sc_non_sat)


@roger_kernel
def calc_inf_rz(state):
    """
    Calculates infiltration into root zone
    """
    vs = state.variables

    # matrix infiltration into root zone
    vs.inf_mat_rz = update(
        vs.inf_mat_rz,
        at[2:-2, 2:-2], vs.inf_mat[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # shrinkage crack infiltration into root zone
    vs.inf_sc_rz = update(
        vs.inf_sc_rz,
        at[2:-2, 2:-2], vs.inf_sc[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.inf_rz = update(
        vs.inf_rz,
        at[2:-2, 2:-2], (vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_mp_rz[2:-2, 2:-2] + vs.inf_sc_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # update root zone storage after infiltration
    vs.S_fp_rz = update_add(
        vs.S_fp_rz,
        at[2:-2, 2:-2], vs.inf_rz[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # root zone fine pore excess fills root zone large pores
    mask = (vs.S_fp_rz > vs.S_ufc_rz)
    vs.S_lp_rz = update_add(
        vs.S_lp_rz,
        at[2:-2, 2:-2], (vs.S_fp_rz[2:-2, 2:-2] - vs.S_ufc_rz[2:-2, 2:-2]) * mask[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_fp_rz = update(
        vs.S_fp_rz,
        at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.S_ufc_rz[2:-2, 2:-2], vs.S_fp_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(inf_mat_rz=vs.inf_mat_rz, inf_mp_rz=vs.inf_mp_rz, inf_sc_rz=vs.inf_sc_rz, inf_rz=vs.inf_rz, S_fp_rz=vs.S_fp_rz, S_lp_rz=vs.S_lp_rz)


@roger_kernel
def calc_surface_runoff(state):
    """
    Calculates surface runoff
    """
    vs = state.variables

    vs.q_hof = update(
        vs.q_hof,
        at[2:-2, 2:-2], npx.where((vs.z0[2:-2, 2:-2, vs.tau] > 0) & (vs.S_rz[2:-2, 2:-2, vs.tau] < vs.S_sat_rz[2:-2, 2:-2]), vs.z0[2:-2, 2:-2, vs.tau], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.z0 = update_add(
        vs.z0,
        at[2:-2, 2:-2, vs.tau], -vs.q_hof[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sof = update(
        vs.q_sof,
        at[2:-2, 2:-2], npx.where((vs.S_lp_rz[2:-2, 2:-2] + vs.S_fp_rz[2:-2, 2:-2]) > (vs.S_ac_rz[2:-2, 2:-2] + vs.S_ufc_rz[2:-2, 2:-2]), (vs.S_lp_rz[2:-2, 2:-2] + vs.S_fp_rz[2:-2, 2:-2]) - (vs.S_ac_rz[2:-2, 2:-2] + vs.S_ufc_rz[2:-2, 2:-2]), 0) * vs.maskCatch[2:-2, 2:-2],
    )

    mask = (vs.q_sof > 0)
    vs.S_fp_rz = update(
        vs.S_fp_rz,
        at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.S_ufc_rz[2:-2, 2:-2], vs.S_fp_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_lp_rz = update(
        vs.S_lp_rz,
        at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.S_ac_rz[2:-2, 2:-2], vs.S_lp_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sur = update(
        vs.q_sur,
        at[2:-2, 2:-2], vs.q_hof[2:-2, 2:-2] + vs.q_sof[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(q_hof=vs.q_hof, q_sof=vs.q_sof, q_sur=vs.q_sur, z0=vs.z0)


@roger_kernel
def calc_inf(state):
    """
    Calculates infiltration
    """
    vs = state.variables

    vs.inf = update(
        vs.inf,
        at[2:-2, 2:-2], (vs.inf_rz[2:-2, 2:-2] + vs.inf_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
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
        at[2:-2, 2:-2], npx.where(vs.theta_d_fp[2:-2, 2:-2] > 0, vs.inf_mat_event_csum[2:-2, 2:-2] / vs.theta_d_fp[2:-2, 2:-2], vs.z_wf[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    z_wf_fc = update(
        z_wf_fc,
        at[2:-2, 2:-2], npx.where(z_wf_fc[2:-2, 2:-2] > vs.z_soil[2:-2, 2:-2], vs.z_soil[2:-2, 2:-2], z_wf_fc[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
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
        at[2:-2, 2:-2], (vs.theta_sat[2:-2, 2:-2] - vs.theta_rz[2:-2, 2:-2, vs.tau]) * (1 - vs.sealing[2:-2, 2:-2]/100) * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    theta_d = update(
        theta_d,
        at[2:-2, 2:-2], npx.where(vs.z_soil[2:-2, 2:-2] <= 0, 0.01, theta_d[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    theta_d = update(
        theta_d,
        at[2:-2, 2:-2], npx.where(theta_d[2:-2, 2:-2] <= 0, 0.01, theta_d[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
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
        at[2:-2, 2:-2], ((vs.theta_sat[2:-2, 2:-2] - vs.theta_rz[2:-2, 2:-2, vs.tau]) / (vs.theta_sat[2:-2, 2:-2] - vs.theta_pwp[2:-2, 2:-2])) * (1 - vs.sealing[2:-2, 2:-2]/100) * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    theta_d_rel = update(
        theta_d_rel,
        at[2:-2, 2:-2], npx.where(vs.z_soil[2:-2, 2:-2] <= 0, 0.01, theta_d_rel[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    theta_d_rel = update(
        theta_d_rel,
        at[2:-2, 2:-2], npx.where(theta_d_rel[2:-2, 2:-2] <= 0, 0.01, theta_d_rel[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
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
        at[2:-2, 2:-2], (vs.theta_d[2:-2, 2:-2] / (vs.theta_sat[2:-2, 2:-2] - vs.theta_pwp[2:-2, 2:-2])) * (1 - vs.sealing[2:-2, 2:-2]/100) * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    theta_d_rel = update(
        theta_d_rel,
        at[2:-2, 2:-2], npx.where(vs.z_soil[2:-2, 2:-2] <= 0, 0.01, theta_d_rel[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    theta_d_rel = update(
        theta_d_rel,
        at[2:-2, 2:-2], npx.where(theta_d_rel[2:-2, 2:-2] <= 0, 0.01, theta_d_rel[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
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
        at[2:-2, 2:-2], (vs.theta_fc[2:-2, 2:-2] - vs.theta_rz[2:-2, 2:-2, vs.tau]) * (1 - vs.sealing[2:-2, 2:-2]/100) * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    theta_d_fp = update(
        theta_d_fp,
        at[2:-2, 2:-2], npx.where(vs.z_soil[2:-2, 2:-2] <= 0, 0.01, theta_d_fp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    theta_d_fp = update(
        theta_d_fp,
        at[2:-2, 2:-2], npx.where(theta_d_fp[2:-2, 2:-2] <= 0, 0.01, theta_d_fp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return theta_d_fp


@roger_kernel
def calc_pi_gr(state):
    """
    Calculates threshold of precipitation intensity
    """
    vs = state.variables

    pi_gr = allocate(state.dimensions, ("x", "y"), dtype=int)

    pi_gr = update(
        pi_gr,
        at[2:-2, 2:-2], vs.ks[2:-2, 2:-2] * (((vs.theta_d[2:-2, 2:-2] * vs.wfs[2:-2, 2:-2])/(vs.prec_event_csum[2:-2, 2:-2] + 1)) + 1),
    )

    return pi_gr


@roger_kernel
def calc_pi_m(state):
    """
    Calculates infiltration sum at saturation
    """
    vs = state.variables

    pi_m = allocate(state.dimensions, ("x", "y"))

    pi_m = update(
        pi_m,
        at[2:-2, 2:-2], vs.ks[2:-2, 2:-2] * vs.theta_d[2:-2, 2:-2] * vs.wfs[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return pi_m


@roger_kernel
def calc_sat_time(state, pi_m, pi_gr):
    """
    Calculates time to reach soil matrix saturation during rainfall event.
    """
    vs = state.variables

    mask1 = (vs.pi_m <= vs.prec_event_csum) & (vs.pi_m > vs.pi_gr) & (vs.t_sat == 0)
    mask2 = ((vs.prec[:, :, vs.tau] * (1 / vs.dt) - vs.ks) * vs.prec_event_csum > vs.ks * vs.theta_d * vs.wfs) & (vs.pi_m <= vs.prec_event_csum) & (vs.pi_m <= vs.pi_gr) & (vs.t_sat == 0)
    vs.t_sat = update(
        vs.t_sat,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], vs.t_event_csum[2:-2, 2:-2] - vs.dt, vs.t_sat[2:-2, 2:-2]),
    )
    vs.t_sat = update(
        vs.t_sat,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], vs.t_event_csum[2:-2, 2:-2] + ((vs.ks[2:-2, 2:-2] * vs.theta_d[2:-2, 2:-2] * vs.wfs[2:-2, 2:-2]) / (vs.pi_m[2:-2, 2:-2] * (vs.pi_m[2:-2, 2:-2] * - vs.ks[2:-2, 2:-2]))) - (vs.dt / vs.pi_m[2:-2, 2:-2]) * vs.prec_event_csum[2:-2, 2:-2], vs.t_sat[2:-2, 2:-2]),
    )

    return vs.t_sat


@roger_kernel
def calc_Fs(state, pi_m):
    """
    Calculates infiltration rate at saturation.
    """
    vs = state.variables

    Fs = allocate(state.dimensions, ("x", "y"))

    Fs = update(
        Fs,
        at[2:-2, 2:-2], ((vs.ks[2:-2, 2:-2] * vs.theta_d[2:-2, 2:-2] * vs.wfs[2:-2, 2:-2]) / (pi_m[2:-2, 2:-2] - vs.ks[2:-2, 2:-2])) * vs.maskCatch[2:-2, 2:-2],
    )
    Fs = update(
        Fs,
        at[2:-2, 2:-2], npx.where(pi_m[2:-2, 2:-2] <= vs.ks[2:-2, 2:-2], pi_m[2:-2, 2:-2], Fs[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
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
        at[2:-2, 2:-2], npx.where(vs.theta_rz[2:-2, 2:-2, vs.tau] < vs.theta_4[2:-2, 2:-2], vs.z_sc_max[2:-2, 2:-2], npx.where((vs.theta_rz[2:-2, 2:-2, vs.tau] >= vs.theta_4[2:-2, 2:-2]) & (vs.theta_rz[2:-2, 2:-2, vs.tau] < vs.theta_27[2:-2, 2:-2]), (vs.theta_rz[2:-2, 2:-2, vs.tau] - vs.theta_4[2:-2, 2:-2]) / (vs.theta_27[2:-2, 2:-2] - vs.theta_4[2:-2, 2:-2]), 0) * vs.z_sc_max[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sc = update(
        vs.z_sc,
        at[2:-2, 2:-2], npx.where(vs.theta_rz[2:-2, 2:-2, vs.tau] < vs.theta_4[2:-2, 2:-2], vs.z_sc_max[2:-2, 2:-2], vs.z_sc[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sc = update(
        vs.z_sc,
        at[2:-2, 2:-2], npx.where(vs.theta_rz[2:-2, 2:-2, vs.tau] > vs.theta_27[2:-2, 2:-2], 0, vs.z_sc[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sc = update(
        vs.z_sc,
        at[2:-2, 2:-2], ((1. - vs.sealing[2:-2, 2:-2]/100.) * vs.z_sc[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.z_sc = update(
        vs.z_sc,
        at[2:-2, 2:-2], npx.where(vs.z_sc[2:-2, 2:-2] > vs.z_root[2:-2, 2:-2, vs.tau], vs.z_root[2:-2, 2:-2, vs.tau], vs.z_sc[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.z_sc = update(
        vs.z_sc,
        at[2:-2, 2:-2], npx.where((vs.lu_id[2:-2, 2:-2] == 13), 0, vs.z_sc[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
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
        at[2:-2, 2:-2], 1 * vs.maskCatch[2:-2, 2:-2],
    )

    # wetting front depth (in mm)
    vs.z_wf = update(
        vs.z_wf,
        at[2:-2, 2:-2, :], 0,
    )
    # first wetting front
    vs.z_wf_t0 = update(
        vs.z_wf_t0,
        at[2:-2, 2:-2, :], 0,
    )
    # second wetting front
    vs.z_wf_t1 = update(
        vs.z_wf_t1,
        at[2:-2, 2:-2, :], 0,
    )
    # wetting front depth to reach field capacity
    vs.z_wf_fc = update(
        vs.z_wf_fc,
        at[2:-2, 2:-2], 0,
    )
    # accumulated infiltration during an event (in mm)
    # matrix
    vs.inf_mat_event_csum = update(
        vs.inf_mat_event_csum,
        at[2:-2, 2:-2], 0,
    )

    # accumulated potential infiltration during an event (in mm)
    # matrix
    vs.inf_mat_pot_event_csum = update(
        vs.inf_mat_pot_event_csum,
        at[2:-2, 2:-2], 0,
    )

    # macropores
    vs.inf_mp_event_csum = update(
        vs.inf_mp_event_csum,
        at[2:-2, 2:-2], 0,
    )
    # radial length of shrinkage crack wetting front
    vs.y_mp = update(
        vs.y_mp,
        at[2:-2, 2:-2, :], 0,
    )
    # shrinkage cracks
    vs.inf_sc_event_csum = update(
        vs.inf_sc_event_csum,
        at[2:-2, 2:-2], 0,
    )
    # radial length of shrinkage crack wetting front
    vs.y_sc = update(
        vs.y_sc,
        at[2:-2, 2:-2, :], 0,
    )

    # soil moisture deficit
    theta_d = calc_theta_d(state)
    vs.theta_d = update(
        vs.theta_d,
        at[2:-2, 2:-2], theta_d[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    theta_d_rel = calc_theta_d_rel(state)
    vs.theta_d_rel = update(
        vs.theta_d_rel,
        at[2:-2, 2:-2], theta_d_rel[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.theta_d_t0 = update(
        vs.theta_d_t0,
        at[2:-2, 2:-2], theta_d[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.theta_d_rel_t0 = update(
        vs.theta_d_rel_t0,
        at[2:-2, 2:-2], theta_d_rel[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    theta_d_fp = calc_theta_d_fp(state)
    vs.theta_d_fp = update(
        vs.theta_d_fp,
        at[2:-2, 2:-2], theta_d_fp[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    # accumulated precipitation wihin event (in mm)
    vs.prec_event_csum = update(
        vs.prec_event_csum,
        at[2:-2, 2:-2], 0,
    )
    # accumulated time during an event (in mm)
    vs.t_event_csum = update(
        vs.t_event_csum,
        at[2:-2, 2:-2], 0,
    )

    # reset accumulated soil evaporation deficit
    vs.de = update(
        vs.de,
        at[2:-2, 2:-2], 0,
    )

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
                        t_event_csum=vs.t_event_csum,
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
    mask = (vs.prec[:, :, vs.tau] == 0) & (vs.prec[:, :, vs.taum1] != 0)

    vs.z_wf_fc = update(
        vs.z_wf_fc,
        at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], z_wf_fc[2:-2, 2:-2], vs.z_wf_fc[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(z_wf_fc=vs.z_wf_fc)


@roger_kernel
def set_event_vars_end_rainfall_pause(state):
    """
    Set event-based variables at the end of an rainfall pause
    """
    vs = state.variables

    mask = (vs.prec[:, :, vs.tau] != 0) & (vs.prec[:, :, vs.taum1] == 0)

    vs.no_wf = update(
        vs.no_wf,
        at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], 2, vs.no_wf[2:-2, 2:-2]),
    )

    theta_d = calc_theta_d(state)
    vs.theta_d = update(
        vs.theta_d,
        at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], theta_d[2:-2, 2:-2], vs.theta_d[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    theta_d_rel = calc_theta_d_rel(state)
    vs.theta_d_rel = update(
        vs.theta_d_rel,
        at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], theta_d_rel[2:-2, 2:-2], vs.theta_d_rel[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    # second wetting front
    vs.z_wf_t1 = update(
        vs.z_wf_t1,
        at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2, npx.newaxis], 0, vs.z_wf_t1[2:-2, 2:-2, :]),
    )

    # accumulated event precipitation
    vs.prec_event_csum = update(
        vs.prec_event_csum,
        at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], 0, vs.prec_event_csum[2:-2, 2:-2]),
    )
    # accumulated time during an event (in mm)
    vs.t_event_csum = update(
        vs.t_event_csum,
        at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], 0, vs.t_event_csum[2:-2, 2:-2]),
    )

    return KernelOutput(no_wf=vs.no_wf,
                        z_wf_t1=vs.z_wf_t1,
                        theta_d=vs.theta_d,
                        theta_d_rel=vs.theta_d_rel,
                        prec_event_csum=vs.prec_event_csum,
                        t_event_csum=vs.t_event_csum,
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
        at[2:-2, 2:-2, :], 0,
    )
    # first wetting front
    vs.z_wf_t0 = update(
        vs.z_wf_t0,
        at[2:-2, 2:-2, :], 0,
    )
    # second wetting front
    vs.z_wf_t1 = update(
        vs.z_wf_t1,
        at[2:-2, 2:-2, :], 0,
    )
    # radial length of shrinkage crack wetting front
    vs.y_mp = update(
        vs.y_mp,
        at[2:-2, 2:-2, vs.tau], 0,
    )
    # radial length of shrinkage crack wetting front
    vs.y_sc = update(
        vs.y_sc,
        at[2:-2, 2:-2, :], 0,
    )
    # soil moisture deficit
    theta_d = calc_theta_d(state)
    vs.theta_d = update(
        vs.theta_d,
        at[2:-2, 2:-2], theta_d[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.theta_d_t0 = update(
        vs.theta_d_t0,
        at[2:-2, 2:-2], theta_d[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.pi_gr = update(
        vs.pi_gr,
        at[2:-2, 2:-2], 0,
    )
    vs.pi_m = update(
        vs.pi_m,
        at[2:-2, 2:-2], 0,
    )
    vs.t_sat = update(
        vs.t_sat,
        at[2:-2, 2:-2], 0,
    )
    vs.Fs = update(
        vs.Fs,
        at[2:-2, 2:-2], 0,
    )
    vs.z_sc = update(
        vs.z_sc,
        at[2:-2, 2:-2], 0,
    )

    return KernelOutput(z_wf=vs.z_wf,
                        z_wf_t0=vs.z_wf_t0,
                        z_wf_t1=vs.z_wf_t1,
                        y_mp=vs.y_mp,
                        y_sc=vs.y_sc,
                        theta_d=vs.theta_d,
                        theta_d_t0=vs.theta_d_t0,
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

    cond1 = ((vs.event_id[vs.taum1] == 0) & (vs.event_id[vs.tau] >= 1))
    cond2 = ((vs.prec[2:-2, 2:-2, vs.tau] == 0).any() & (vs.prec[2:-2, 2:-2, vs.taum1] != 0).any() & (vs.event_id[vs.taum1] >= 1))
    cond3 = ((vs.prec[2:-2, 2:-2, vs.tau] != 0).any() & (vs.prec[2:-2, 2:-2, vs.taum1] == 0).any() & (vs.event_id[vs.taum1] == vs.event_id[vs.tau]))
    cond4 = ((vs.event_id[vs.taum1] >= 1) & (vs.event_id[vs.tau] == 0))
    cond5 = (vs.event_id[vs.tau] >= 1)
    if cond1.any():
        vs.update(calc_depth_shrinkage_cracks(state))
        vs.update(set_event_vars(state))
    if cond2.any():
        vs.update(set_event_vars_start_rainfall_pause(state))
    if cond3.any():
        vs.update(set_event_vars_end_rainfall_pause(state))
    if cond5.any():
        vs.t_event_csum = update_add(
            vs.t_event_csum,
            at[2:-2, 2:-2], vs.dt,
        )

    vs.update(calc_green_ampt_params(state))
    vs.update(calc_inf_mat(state))
    vs.update(calc_inf_mp(state))
    vs.update(calc_inf_sc(state))
    vs.update(calc_inf_rz(state))
    vs.update(calc_inf(state))
    vs.update(calc_surface_runoff(state))

    if cond4.any():
        vs.update(reset_event_vars(state))


@roger_kernel
def calculate_infiltration_rz_transport_kernel(state):
    """
    Calculates transport of infiltration
    """
    vs = state.variables

    vs.sa_rz = update_add(
        vs.sa_rz,
        at[2:-2, 2:-2, vs.tau, 0], vs.inf_mat_rz[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.sa_rz = update_add(
        vs.sa_rz,
        at[2:-2, 2:-2, vs.tau, 0], vs.inf_pf_rz[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
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
        at[2:-2, 2:-2], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_inf_pf_rz = update(
        vs.C_inf_pf_rz,
        at[2:-2, 2:-2], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
    )
    # travel time distribution
    vs.tt_inf_mat_rz = update(
        vs.tt_inf_mat_rz,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.tt_inf_pf_rz = update(
        vs.tt_inf_pf_rz,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    # isotope travel time distribution
    vs.mtt_inf_mat_rz = update(
        vs.mtt_inf_mat_rz,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.mtt_inf_pf_rz = update(
        vs.mtt_inf_pf_rz,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    # update isotope StorAge
    vs.msa_rz = update(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, :], npx.where(vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.sa_rz[2:-2, 2:-2, vs.tau, :] > 0, vs.msa_rz[2:-2, 2:-2, vs.tau, :] * (vs.sa_rz[2:-2, 2:-2, vs.tau, :] / (vs.tt_inf_mat_rz[2:-2, 2:-2, :] * vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] + vs.sa_rz[2:-2, 2:-2, vs.tau, :])) + vs.mtt_inf_mat_rz[2:-2, 2:-2, :] * ((vs.tt_inf_mat_rz[2:-2, 2:-2, :] * vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis]) / (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] + vs.sa_rz[2:-2, 2:-2, vs.tau, :])), vs.msa_rz[2:-2, 2:-2, vs.tau, :]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.msa_rz = update(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, :], npx.where(vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] + vs.sa_rz[2:-2, 2:-2, vs.tau, :] > 0, vs.msa_rz[2:-2, 2:-2, vs.tau, :] * (vs.sa_rz[2:-2, 2:-2, vs.tau, :] / (vs.tt_inf_pf_rz[2:-2, 2:-2, :] * vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] + vs.sa_rz[2:-2, 2:-2, vs.tau, :])) + vs.mtt_inf_pf_rz[2:-2, 2:-2, :] * ((vs.tt_inf_pf_rz[2:-2, 2:-2, :] * vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis]) / (vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] + vs.sa_rz[2:-2, 2:-2, vs.tau, :])), vs.msa_rz[2:-2, 2:-2, vs.tau, :]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    # update StorAge
    vs.sa_rz = update_add(
        vs.sa_rz,
        at[2:-2, 2:-2, vs.tau, 0], vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(sa_rz=vs.sa_rz, msa_rz=vs.msa_rz, C_inf_mat_rz=vs.C_inf_mat_rz, C_inf_pf_rz=vs.C_inf_pf_rz, tt_inf_mat_rz=vs.tt_inf_mat_rz, mtt_inf_mat_rz=vs.mtt_inf_mat_rz, tt_inf_pf_rz=vs.tt_inf_pf_rz, mtt_inf_pf_rz=vs.mtt_inf_pf_rz)


@roger_kernel
def calculate_infiltration_rz_transport_anion_kernel(state):
    """
    Calculates isotope transport of infiltration
    """
    vs = state.variables

    # solute concentration of infiltration
    vs.C_inf_mat_rz = update(
        vs.C_inf_mat_rz,
        at[2:-2, 2:-2], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_inf_pf_rz = update(
        vs.C_inf_pf_rz,
        at[2:-2, 2:-2], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )
    # travel time distribution
    vs.tt_inf_mat_rz = update(
        vs.tt_inf_mat_rz,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.tt_inf_pf_rz = update(
        vs.tt_inf_pf_rz,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    # solute travel time distribution
    vs.mtt_inf_mat_rz = update(
        vs.mtt_inf_mat_rz,
        at[2:-2, 2:-2, 0], vs.inf_mat_rz[2:-2, 2:-2] * vs.C_in[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.mtt_inf_pf_rz = update(
        vs.mtt_inf_pf_rz,
        at[2:-2, 2:-2, 0], vs.inf_pf_rz[2:-2, 2:-2] * vs.C_in[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    # solute mass of infiltration
    vs.M_inf_mat_rz = update(
        vs.M_inf_mat_rz,
        at[2:-2, 2:-2], vs.C_inf_mat_rz[2:-2, 2:-2] * vs.inf_mat_rz[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.M_inf_pf_rz = update(
        vs.C_inf_pf_rz,
        at[2:-2, 2:-2], vs.C_inf_pf_rz[2:-2, 2:-2] * vs.inf_pf_rz[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.sa_rz = update_add(
        vs.sa_rz,
        at[2:-2, 2:-2, vs.tau, 0], vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.msa_rz = update_add(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, 0], vs.M_inf_mat_rz[2:-2, 2:-2] + vs.M_inf_pf_rz[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(sa_rz=vs.sa_rz, msa_rz=vs.msa_rz, C_inf_mat_rz=vs.C_inf_mat_rz, C_inf_pf_rz=vs.C_inf_pf_rz, M_inf_mat_rz=vs.M_inf_mat_rz, M_inf_pf_rz=vs.M_inf_pf_rz, tt_inf_mat_rz=vs.tt_inf_mat_rz, mtt_inf_mat_rz=vs.mtt_inf_mat_rz, tt_inf_pf_rz=vs.tt_inf_pf_rz, mtt_inf_pf_rz=vs.mtt_inf_pf_rz)


@roger_kernel
def calculate_infiltration_ss_transport_kernel(state):
    """
    Calculates travel time of transpiration
    """
    vs = state.variables

    vs.sa_ss = update_add(
        vs.sa_ss,
        at[2:-2, 2:-2, vs.tau, 0], vs.inf_mat_ss[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
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
        at[2:-2, 2:-2], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
    )
    # travel time distribution
    vs.tt_inf_pf_ss = update(
        vs.tt_inf_pf_ss,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    # isotope travel time distribution
    vs.mtt_inf_pf_ss = update(
        vs.mtt_inf_pf_ss,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )
    # update isotope StorAge
    vs.msa_ss = update(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.tau, :], npx.where(vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] + vs.sa_ss[2:-2, 2:-2, vs.tau, :] > 0, vs.msa_ss[2:-2, 2:-2, vs.tau, :] * (vs.sa_ss[2:-2, 2:-2, vs.tau, :] / (vs.tt_inf_pf_ss[2:-2, 2:-2, :] * vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] + vs.sa_ss[2:-2, 2:-2, vs.tau, :])) + vs.mtt_inf_pf_ss[2:-2, 2:-2, :] * ((vs.tt_inf_pf_ss[2:-2, 2:-2, :] * vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis]) / (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] + vs.sa_ss[2:-2, 2:-2, vs.tau, :])), vs.msa_ss[2:-2, 2:-2, vs.tau, :]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    # update StorAge
    vs.sa_ss = update_add(
        vs.sa_ss,
        at[2:-2, 2:-2, vs.tau, 0], vs.inf_pf_ss[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(sa_ss=vs.sa_ss, msa_ss=vs.msa_ss, C_inf_pf_ss=vs.C_inf_pf_ss, tt_inf_pf_ss=vs.tt_inf_pf_ss, mtt_inf_pf_ss=vs.mtt_inf_pf_ss)


@roger_kernel
def calculate_infiltration_ss_transport_anion_kernel(state):
    """
    Calculates isotope transport of infiltration
    """
    vs = state.variables

    # solute concentration of infiltration
    vs.C_inf_pf_ss = update(
        vs.C_inf_pf_ss,
        at[2:-2, 2:-2], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )
    # travel time distribution
    vs.tt_inf_pf_ss = update(
        vs.tt_inf_pf_ss,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    # isotope travel time distribution
    vs.mtt_inf_pf_ss = update(
        vs.mtt_inf_pf_ss,
        at[2:-2, 2:-2, 0], vs.inf_pf_ss[2:-2, 2:-2] * vs.C_in[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    # solute mass of infiltration
    vs.M_inf_pf_ss = update(
        vs.C_inf_pf_ss,
        at[2:-2, 2:-2], vs.C_inf_pf_ss[2:-2, 2:-2] * vs.inf_pf_ss[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    # update StorAge
    vs.sa_ss = update_add(
        vs.sa_ss,
        at[2:-2, 2:-2, vs.tau, 0], vs.inf_pf_ss[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.msa_ss = update_add(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.tau, 0], vs.M_inf_pf_ss[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(sa_ss=vs.sa_ss, msa_ss=vs.msa_ss, C_inf_pf_ss=vs.C_inf_pf_ss, M_inf_pf_ss=vs.M_inf_pf_ss, tt_inf_pf_ss=vs.tt_inf_pf_ss, mtt_inf_pf_ss=vs.mtt_inf_pf_ss)


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
