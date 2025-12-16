from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at
from roger.core.utilities import linear_regression


@roger_kernel
def _calc_theta_d_rel(state, theta):
    """
    Calculates relative soil moisture deficit.
    """
    vs = state.variables

    theta_d_rel = allocate(state.dimensions, ("x", "y"))
    theta_d_rel = update(
        theta_d_rel,
        at[2:-2, 2:-2], ((vs.theta_sat[2:-2, 2:-2] - theta[2:-2, 2:-2]) / (vs.theta_sat[2:-2, 2:-2] - vs.theta_pwp[2:-2, 2:-2])) * vs.maskCatch[2:-2, 2:-2],
    )

    return theta_d_rel


@roger_kernel
def calc_theta_d_rel(state):
    """
    Calculates film flow event duration
    """
    vs = state.variables
    settings = state.settings

    itt_event = allocate(state.dimensions, ("x", "y", "timesteps_event_ff"))
    itt_event = update(
        itt_event,
        at[2:-2, 2:-2, :], npx.arange(0, settings.nittevent_ff, 1)[npx.newaxis, npx.newaxis, :],
    )

    vs.theta_d_rel_rz_ff = update(
        vs.theta_d_rel_rz_ff,
        at[:, :, vs.event_no_ff - 1], _calc_theta_d_rel(state, vs.theta_rz[:, :, vs.tau]),
    )
    vs.theta_d_rel_ss_ff = update(
        vs.theta_d_rel_ss_ff,
        at[:, :, vs.event_no_ff - 1], _calc_theta_d_rel(state, vs.theta_ss[:, :, vs.tau]),
    )

    return KernelOutput(
        theta_d_rel_rz_ff=vs.theta_d_rel_rz_ff,
        theta_d_rel_ss_ff=vs.theta_d_rel_ss_ff,
        )


@roger_kernel
def calc_t_end(state):
    """
    Calculates film flow event duration
    """
    vs = state.variables
    settings = state.settings

    itt_event = allocate(state.dimensions, ("x", "y", "timesteps_event_ff"), dtype=int)
    itt_event = update(
        itt_event,
        at[2:-2, 2:-2, :], npx.arange(0, settings.nittevent_ff, 1)[npx.newaxis, npx.newaxis, :],
    )

    vs.t_end_ff = update(
        vs.t_end_ff,
        at[2:-2, 2:-2, vs.event_no_ff - 1], npx.min(npx.where(itt_event[2:-2, 2:-2, :] > vs.ts_ff[2:-2, 2:-2, vs.event_no_ff - 1, npx.newaxis], npx.where(vs.rain_int_ff[2:-2, 2:-2, vs.event_no_ff - 1, npx.newaxis] * ((vs.ti_ff[2:-2, 2:-2, vs.event_no_ff - 1, npx.newaxis] - vs.ts_ff[2:-2, 2:-2, vs.event_no_ff - 1, npx.newaxis]) / (itt_event[2:-2, 2:-2, :] - vs.ts_ff[2:-2, 2:-2, vs.event_no_ff - 1, npx.newaxis]))**(3/2) <= vs.rain_int_ff[2:-2, 2:-2, vs.event_no_ff - 1, npx.newaxis] * settings.ff_tc, itt_event[2:-2, 2:-2, :], settings.nittevent_ff), settings.nittevent_ff), axis=-1),
    )

    return KernelOutput(
        t_end_ff=vs.t_end_ff,
        )


@roger_kernel
def calc_volume_flux_density(state):
    """
    Calculates volume flux density of film flow
    """
    vs = state.variables
    settings = state.settings

    itt_event = allocate(state.dimensions, ("x", "y", "timesteps_event_ff"))
    itt_event = update(
        itt_event,
        at[2:-2, 2:-2, :], npx.arange(0, settings.nittevent_ff, 1)[npx.newaxis, npx.newaxis, :],
    )

    # define lower and upper quartile of rainfall
    idx_rain_25 = allocate(state.dimensions, ("x", "y"))
    idx_rain_75 = allocate(state.dimensions, ("x", "y"))
    idx_rain_25 = update(
        idx_rain_25,
        at[2:-2, 2:-2], npx.max(npx.where((vs.rain_event_csum[2:-2, 2:-2, :] <= 0.25 * vs.rain_event_sum[2:-2, 2:-2, npx.newaxis]) & (vs.rain_event_csum[2:-2, 2:-2, :] > 0), itt_event[2:-2, 2:-2, :], 0), axis=-1),
    )
    idx_rain_75 = update(
        idx_rain_75,
        at[2:-2, 2:-2], npx.min(npx.where(vs.rain_event_csum[2:-2, 2:-2, :] >= 0.75 * vs.rain_event_sum[2:-2, 2:-2, npx.newaxis], itt_event[2:-2, 2:-2, :], settings.nittevent_ff), axis=-1),
    )

    idx_reg = allocate(state.dimensions, ("x", "y", "timesteps_event_ff"))
    rain_reg = allocate(state.dimensions, ("x", "y", "timesteps_event_ff"))
    rain_csum_reg = allocate(state.dimensions, ("x", "y", "timesteps_event_ff"))
    rain_init = allocate(state.dimensions, ("x", "y"))
    params_reg = allocate(state.dimensions, ("x", "y", 2))
    idx_reg = update(
        idx_reg,
        at[2:-2, 2:-2, :], npx.where((itt_event[2:-2, 2:-2, :] >= idx_rain_25[2:-2, 2:-2, npx.newaxis]) & (itt_event[2:-2, 2:-2, :] <= idx_rain_75[2:-2, 2:-2, npx.newaxis]), itt_event[2:-2, 2:-2, :], 0),
    )
    rain_init = update(
        rain_init,
        at[2:-2, 2:-2], npx.max(npx.where(vs.rain_event_csum[2:-2, 2:-2, :] <= 0.25 * vs.rain_event_sum[2:-2, 2:-2, npx.newaxis], vs.rain_event_csum[2:-2, 2:-2, :], 0), axis=-1),
    )
    rain_reg = update(
        rain_reg,
        at[2:-2, 2:-2, :], npx.where((itt_event[2:-2, 2:-2, :] >= idx_rain_25[2:-2, 2:-2, npx.newaxis]) & (itt_event[2:-2, 2:-2, :] <= idx_rain_75[2:-2, 2:-2, npx.newaxis]), vs.rain_event[2:-2, 2:-2, :], 0),
    )
    rain_csum_reg = update(
        rain_csum_reg,
        at[2:-2, 2:-2, :], npx.cumsum(rain_reg[2:-2, 2:-2, :], axis=-1) + rain_init[2:-2, 2:-2, npx.newaxis],
    )
    # linear regression to determine volume flux density
    params_reg = update(
        params_reg,
        at[2:-2, 2:-2, :], linear_regression(idx_reg[2:-2, 2:-2, :], rain_csum_reg[2:-2, 2:-2, :], params_reg[2:-2, 2:-2, :]),
    )

    vs.qs_ff = update(
        vs.qs_ff,
        at[2:-2, 2:-2, vs.event_no_ff - 1], params_reg[2:-2, 2:-2, 0]/600/1000,
    )

    vs.tb_ff = update(
        vs.tb_ff,
        at[2:-2, 2:-2, vs.event_no_ff - 1], npx.where(-params_reg[2:-2, 2:-2, 1]/params_reg[2:-2, 2:-2, 0] >= 0, -params_reg[2:-2, 2:-2, 1]/params_reg[2:-2, 2:-2, 0], 0),
    )

    vs.ts_ff = update(
        vs.ts_ff,
        at[2:-2, 2:-2, vs.event_no_ff - 1], vs.tb_ff[2:-2, 2:-2, vs.event_no_ff - 1] + (vs.rain_event_sum[2:-2, 2:-2] - params_reg[2:-2, 2:-2, 1])/params_reg[2:-2, 2:-2, 0],
    )

    return KernelOutput(
        qs_ff=vs.qs_ff,
        tb_ff=vs.tb_ff,
        ts_ff=vs.ts_ff,
        )


@roger_kernel
def calc_rain_pulse(state):
    """
    Calculates rainfall intensity of input pulse
    """
    vs = state.variables
    settings = state.settings

    itt_event = allocate(state.dimensions, ("x", "y", "timesteps_event_ff"))
    itt_event = update(
        itt_event,
        at[2:-2, 2:-2, :], npx.arange(0, settings.nittevent_ff, 1)[npx.newaxis, npx.newaxis, :],
    )

    ts = allocate(state.dimensions, ("x", "y"))
    tb = allocate(state.dimensions, ("x", "y"))
    ts = update(
        ts,
        at[2:-2, 2:-2], (vs.ts_ff[2:-2, 2:-2, vs.event_no_ff - 1] - vs.ts_ff[2:-2, 2:-2, vs.event_no_ff - 1] % 1) + 1,
    )
    tb = update(
        tb,
        at[2:-2, 2:-2], npx.where(vs.tb_ff[2:-2, 2:-2, vs.event_no_ff - 1] - vs.tb_ff[2:-2, 2:-2, vs.event_no_ff - 1] % 1 > 0, vs.tb_ff[2:-2, 2:-2, vs.event_no_ff - 1] - vs.tb_ff[2:-2, 2:-2, vs.event_no_ff - 1] % 1, 0)
    )

    vs.rain_int_ff = update(
        vs.rain_int_ff,
        at[2:-2, 2:-2, vs.event_no_ff - 1], (vs.qs_ff[2:-2, 2:-2, vs.event_no_ff - 1] * 600 * 1000 * (vs.ts_ff[2:-2, 2:-2, vs.event_no_ff - 1] - vs.tb_ff[2:-2, 2:-2, vs.event_no_ff - 1])) / (ts[2:-2, 2:-2] - tb[2:-2, 2:-2]),
    )

    vs.rain_event_ff = update(
        vs.rain_event_ff,
        at[2:-2, 2:-2, :], npx.where((itt_event[2:-2, 2:-2, :] >= tb[2:-2, 2:-2, npx.newaxis]) & (itt_event[2:-2, 2:-2, :] <= ts[2:-2, 2:-2, npx.newaxis]), vs.rain_int_ff[2:-2, 2:-2, vs.event_no_ff - 1, npx.newaxis], 0),
    )

    return KernelOutput(
        rain_int_ff=vs.rain_int_ff,
        rain_event_ff=vs.rain_event_ff,
        )


@roger_kernel
def calc_velocity(state):
    """
    Calculates film flow velocity
    """
    vs = state.variables

    vs.v_wf = update(
        vs.v_wf,
        at[2:-2, 2:-2, vs.event_no_ff - 1], vs.a_ff[2:-2, 2:-2] * vs.qs_ff[2:-2, 2:-2, vs.event_no_ff - 1]**(2/3) * 600 * 1000,
    )

    vs.v_perc = update(
        vs.v_perc,
        at[2:-2, 2:-2, vs.event_no_ff - 1], vs.v_wf[2:-2, 2:-2, vs.event_no_ff - 1] * 3,
    )

    return KernelOutput(
        v_wf=vs.v_wf,
        v_perc=vs.v_perc,
        )


@roger_kernel
def calc_intersection(state):
    """
    Calculates intersection time and intersection depth between wetting
    front and percolation front.
    """
    vs = state.variables

    vs.ti_ff = update(
        vs.ti_ff,
        at[2:-2, 2:-2, vs.event_no_ff - 1], vs.tb_ff[2:-2, 2:-2, vs.event_no_ff - 1] + 0.5 * (3 * (vs.ts_ff[2:-2, 2:-2, vs.event_no_ff - 1] - vs.tb_ff[2:-2, 2:-2, vs.event_no_ff - 1])),
    )

    vs.zi_ff = update(
        vs.zi_ff,
        at[2:-2, 2:-2, vs.event_no_ff - 1], ((3 * vs.v_wf[2:-2, 2:-2, vs.event_no_ff - 1]) / 2) * (vs.ts_ff[2:-2, 2:-2, vs.event_no_ff - 1] - vs.tb_ff[2:-2, 2:-2, vs.event_no_ff - 1]),
    )

    return KernelOutput(
        ti_ff=vs.ti_ff,
        zi_ff=vs.zi_ff,
        )


@roger_kernel
def calc_intersection_at_soil_depth(state):
    """
    Calculates intersection time of wetting front and percolation front at
    soil depth.
    """
    vs = state.variables

    vs.tw_ff = update(
        vs.tw_ff,
        at[2:-2, 2:-2, vs.event_no_ff - 1], vs.tb_ff[2:-2, 2:-2, vs.event_no_ff - 1] + vs.z_soil[2:-2, 2:-2] / vs.v_wf[2:-2, 2:-2, vs.event_no_ff - 1],
    )

    vs.tp_ff = update(
        vs.tp_ff,
        at[2:-2, 2:-2, vs.event_no_ff - 1], vs.ts_ff[2:-2, 2:-2, vs.event_no_ff - 1] + vs.z_soil[2:-2, 2:-2] / vs.v_perc[2:-2, 2:-2, vs.event_no_ff - 1],
    )

    return KernelOutput(
        tw_ff=vs.tw_ff,
        tp_ff=vs.tp_ff,
        )


@roger_kernel
def calc_infiltration(state):
    """
    Calculates infiltration
    """
    vs = state.variables

    vs.rain_ff = update(
        vs.rain_ff,
        at[2:-2, 2:-2], vs.rain_event_ff[2:-2, 2:-2, vs.itt_event_ff[vs.event_no_ff - 1]],
    )
    vs.prec = update_add(
        vs.prec,
        at[2:-2, 2:-2, vs.tau], vs.rain_ff[2:-2, 2:-2],
    )
    vs.S_f = update_add(
        vs.S_f,
        at[2:-2, 2:-2, vs.event_no_ff - 1], vs.rain_ff[2:-2, 2:-2],
    )

    return KernelOutput(
        prec=vs.prec,
        rain_ff=vs.rain_ff,
        S_f=vs.S_f,
        )


@roger_kernel
def calc_wetting_front_depth(state):
    """
    Calculates wetting front depth
    """
    vs = state.variables

    vs.z_wf_ff = update(
        vs.z_wf_ff,
        at[2:-2, 2:-2, :, vs.tau], npx.where((vs.itt_event_ff[npx.newaxis, npx.newaxis, :] > vs.tb_ff[2:-2, 2:-2, :]) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] < vs.ti_ff[2:-2, 2:-2, :]) & (vs.S_f[2:-2, 2:-2, :] > 0), vs.v_wf[2:-2, 2:-2, :] * (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] - vs.tb_ff[2:-2, 2:-2, :]), vs.z_wf_ff[2:-2, 2:-2, :, vs.tau]),
    )

    vs.z_wf_ff = update(
        vs.z_wf_ff,
        at[2:-2, 2:-2, :, vs.tau], npx.where((vs.itt_event_ff[npx.newaxis, npx.newaxis, :] > vs.ti_ff[2:-2, 2:-2, :]) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] < vs.t_end_ff[2:-2, 2:-2, :]) & (vs.S_f[2:-2, 2:-2, :] > 0), vs.v_perc[2:-2, 2:-2, :] * (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] - vs.ts_ff[2:-2, 2:-2, :])**(1/3) * ((vs.ts_ff[2:-2, 2:-2, :] - vs.tb_ff[2:-2, 2:-2, :]) / 2)**(2/3), vs.z_wf_ff[2:-2, 2:-2, :, vs.tau]),
    )

    vs.z_wf_ff = update(
        vs.z_wf_ff,
        at[2:-2, 2:-2, :, :], npx.where(vs.itt_event_ff[npx.newaxis, npx.newaxis, :, npx.newaxis] >= vs.t_end_ff[2:-2, 2:-2, :, npx.newaxis], 0, vs.z_wf_ff[2:-2, 2:-2, :]),
    )

    vs.z_wf = update(
        vs.z_wf,
        at[2:-2, 2:-2, vs.tau], npx.max(vs.z_wf_ff[2:-2, 2:-2, :, vs.tau], axis=2),
    )

    return KernelOutput(
        z_wf_ff=vs.z_wf_ff,
        z_wf=vs.z_wf,
        )


@roger_kernel
def calc_percolation_front_depth(state):
    """
    Calculates percolation front depth
    """
    vs = state.variables

    vs.z_pf_ff = update(
        vs.z_pf_ff,
        at[2:-2, 2:-2, :, vs.tau], npx.where((vs.itt_event_ff[npx.newaxis, npx.newaxis, :] > vs.ts_ff[2:-2, 2:-2, :]) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] <= vs.ti_ff[2:-2, 2:-2, :]) & (vs.S_f[2:-2, 2:-2, :] > 0), vs.v_perc[2:-2, 2:-2, :] * (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] - vs.ts_ff[2:-2, 2:-2, :]), vs.z_pf_ff[2:-2, 2:-2, :, vs.tau]),
    )

    vs.z_pf_ff = update(
        vs.z_pf_ff,
        at[2:-2, 2:-2, :, vs.tau], npx.where(vs.z_pf_ff[2:-2, 2:-2, :, vs.tau] > vs.z_soil[2:-2, 2:-2, npx.newaxis], vs.z_soil[2:-2, 2:-2, npx.newaxis], vs.z_pf_ff[2:-2, 2:-2, :, vs.tau]),
    )

    vs.z_pf_ff = update(
        vs.z_pf_ff,
        at[2:-2, 2:-2, :, :], npx.where(vs.itt_event_ff[npx.newaxis, npx.newaxis, :, npx.newaxis] >= vs.t_end_ff[2:-2, 2:-2, :, npx.newaxis], 0, vs.z_pf_ff[2:-2, 2:-2, :]),
    )

    vs.z_pf = update(
        vs.z_pf,
        at[2:-2, 2:-2, vs.tau], npx.max(vs.z_pf_ff[2:-2, 2:-2, :, vs.tau], axis=2),
    )

    return KernelOutput(
        z_pf_ff=vs.z_pf_ff,
        z_pf=vs.z_pf,
        )


@roger_kernel
def calc_abstraction(state):
    """
    Calculates abstraction from film flow
    """
    vs = state.variables

    vs.ff_abs_rz = update(
        vs.ff_abs_rz,
        at[2:-2, 2:-2, :], npx.where(((vs.z_wf_ff[2:-2, 2:-2, :, vs.tau] - vs.z_wf_ff[2:-2, 2:-2, :, vs.taum1]) > 0) & (vs.z_wf_ff[2:-2, 2:-2, :, vs.tau] < vs.z_root[2:-2, 2:-2, vs.tau, npx.newaxis]), vs.theta_d_rel_rz_ff[2:-2, 2:-2, :] * vs.wfs[2:-2, 2:-2, npx.newaxis] * vs.ks[2:-2, 2:-2, npx.newaxis] * vs.dt * ((vs.wfs[2:-2, 2:-2, npx.newaxis] + (vs.z_wf_ff[2:-2, 2:-2, :, vs.tau] - vs.z_wf_ff[2:-2, 2:-2, :, vs.taum1])) / (vs.z_wf_ff[2:-2, 2:-2, :, vs.tau] - vs.z_wf_ff[2:-2, 2:-2, :, vs.taum1])) * vs.c_ff[2:-2, 2:-2, npx.newaxis], 0),
    )
    vs.ff_abs_rz = update(
        vs.ff_abs_rz,
        at[2:-2, 2:-2, :], npx.where(vs.ff_abs_rz[2:-2, 2:-2, :] >= vs.S_f[2:-2, 2:-2, :], vs.S_f[2:-2, 2:-2, :], vs.ff_abs_rz[2:-2, 2:-2, :]),
    )
    vs.S_f = update_add(
        vs.S_f,
        at[2:-2, 2:-2, :], npx.where(vs.ff_abs_rz[2:-2, 2:-2, :] > 0, -vs.ff_abs_rz[2:-2, 2:-2, :], 0),
    )

    vs.ff_abs_ss = update(
        vs.ff_abs_ss,
        at[2:-2, 2:-2, :], npx.where(((vs.z_wf_ff[2:-2, 2:-2, :, vs.tau] - vs.z_wf_ff[2:-2, 2:-2, :, vs.taum1]) > 0) & (vs.z_wf_ff[2:-2, 2:-2, :, vs.tau] >= vs.z_root[2:-2, 2:-2, vs.tau, npx.newaxis]) & (vs.z_wf_ff[2:-2, 2:-2, :, vs.tau] <= vs.z_soil[2:-2, 2:-2, npx.newaxis]), vs.theta_d_rel_ss_ff[2:-2, 2:-2, :] * vs.wfs[2:-2, 2:-2, npx.newaxis] * vs.ks[2:-2, 2:-2, npx.newaxis] * vs.dt * ((vs.wfs[2:-2, 2:-2, npx.newaxis] + (vs.z_wf_ff[2:-2, 2:-2, :, vs.tau] - vs.z_wf_ff[2:-2, 2:-2, :, vs.taum1])) / (vs.z_wf_ff[2:-2, 2:-2, :, vs.tau] - vs.z_wf_ff[2:-2, 2:-2, :, vs.taum1])) * vs.c_ff[2:-2, 2:-2, npx.newaxis], 0),
    )
    vs.ff_abs_ss = update(
        vs.ff_abs_ss,
        at[2:-2, 2:-2, :], npx.where(vs.ff_abs_ss[2:-2, 2:-2, :] >= vs.S_f[2:-2, 2:-2, :], vs.S_f[2:-2, 2:-2, :], vs.ff_abs_ss[2:-2, 2:-2, :]),
    )
    vs.ff_abs_ss = update(
        vs.ff_abs_ss,
        at[2:-2, 2:-2, :], npx.where(vs.ff_abs_ss[2:-2, 2:-2, :] >= vs.S_f[2:-2, 2:-2, :], vs.S_f[2:-2, 2:-2, :], vs.ff_abs_ss[2:-2, 2:-2, :]),
    )
    vs.S_f = update_add(
        vs.S_f,
        at[2:-2, 2:-2, :], npx.where(vs.ff_abs_ss[2:-2, 2:-2, :] > 0, -vs.ff_abs_ss[2:-2, 2:-2, :], 0),
    )

    # abstraction of the residual film at the end of the event
    vs.ff_abs_rz = update(
        vs.ff_abs_rz,
        at[2:-2, 2:-2, :], npx.where(vs.itt_event_ff[npx.newaxis, npx.newaxis, :] >= vs.t_end_ff[2:-2, 2:-2, :], vs.S_f_rz[2:-2, 2:-2, :], vs.ff_abs_rz[2:-2, 2:-2, :]),
    )
    vs.ff_abs_ss = update(
        vs.ff_abs_ss,
        at[2:-2, 2:-2, :], npx.where(vs.itt_event_ff[npx.newaxis, npx.newaxis, :] >= vs.t_end_ff[2:-2, 2:-2, :], vs.S_f_ss[2:-2, 2:-2, :], vs.ff_abs_ss[2:-2, 2:-2, :]),
    )
    vs.S_f = update(
        vs.S_f,
        at[2:-2, 2:-2, :], npx.where(vs.itt_event_ff[npx.newaxis, npx.newaxis, :] >= vs.t_end_ff[2:-2, 2:-2, :], 0, vs.S_f[2:-2, 2:-2, :]),
    )

    vs.ff_abs = update(
        vs.ff_abs,
        at[2:-2, 2:-2, :], vs.ff_abs_rz[2:-2, 2:-2, :] + vs.ff_abs_ss[2:-2, 2:-2, :],
    )

    # update root zone storage after abtraction from film flow
    vs.S_fp_rz = update_add(
        vs.S_fp_rz,
        at[2:-2, 2:-2], npx.sum(vs.ff_abs_rz[2:-2, 2:-2, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
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

    # update subsoil storage after abtraction from film flow
    vs.S_fp_ss = update_add(
        vs.S_fp_ss,
        at[2:-2, 2:-2], npx.sum(vs.ff_abs_ss[2:-2, 2:-2], axis=-1) * vs.maskCatch[2:-2, 2:-2],
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

    return KernelOutput(
        ff_abs_rz=vs.ff_abs_rz,
        ff_abs_ss=vs.ff_abs_ss,
        ff_abs=vs.ff_abs,
        S_f=vs.S_f,
        S_fp_rz=vs.S_fp_rz,
        S_lp_rz=vs.S_lp_rz,
        S_fp_ss=vs.S_fp_ss,
        S_lp_ss=vs.S_lp_ss,
        )


@roger_kernel
def calc_drainage(state):
    """
    Calculates film flow drainage
    """
    vs = state.variables

    ff_drain_pot = allocate(state.dimensions, ("x", "y", "events_ff"))

    ff_drain_pot = update(
        ff_drain_pot,
        at[2:-2, 2:-2, :], npx.where((vs.tp_ff[2:-2, 2:-2, :] < vs.ti_ff[2:-2, 2:-2, :]) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] >= vs.tw_ff[2:-2, 2:-2, :]) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] <= vs.tp_ff[2:-2, 2:-2, :]), vs.rain_int_ff[2:-2, 2:-2, :], 0),
    )
    ff_drain_pot = update(
        ff_drain_pot,
        at[2:-2, 2:-2, :], npx.where((vs.tp_ff[2:-2, 2:-2, :] < vs.ti_ff[2:-2, 2:-2, :]) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] > vs.tp_ff[2:-2, 2:-2, :]) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] < vs.ti_ff[2:-2, 2:-2, :]), vs.rain_int_ff[2:-2, 2:-2, :] * (vs.tp_ff[2:-2, 2:-2, :] - vs.ts_ff[2:-2, 2:-2, :]) / (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] - vs.ts_ff[2:-2, 2:-2, :])**(3/2), ff_drain_pot[2:-2, 2:-2, :]),
    )
    ff_drain_pot = update(
        ff_drain_pot,
        at[2:-2, 2:-2, :], npx.where((vs.tp_ff[2:-2, 2:-2, :] < vs.ti_ff[2:-2, 2:-2, :]) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] > vs.ti_ff[2:-2, 2:-2, :]) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] < vs.t_end_ff[2:-2, 2:-2, :]), (vs.S_f[2:-2, 2:-2, :] / 2) * (vs.tw_ff[2:-2, 2:-2, :] - vs.ts_ff[2:-2, 2:-2, :])**(1/2) * (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] - vs.ts_ff[2:-2, 2:-2, :])**(-3/2), ff_drain_pot[2:-2, 2:-2, :]),
    )
    ff_drain_pot = update(
        ff_drain_pot,
        at[2:-2, 2:-2, :], npx.where((vs.tp_ff[2:-2, 2:-2, :] >= vs.ti_ff[2:-2, 2:-2, :]) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] >= vs.tw_ff[2:-2, 2:-2, :]) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] <= vs.ti_ff[2:-2, 2:-2, :]), vs.rain_int_ff[2:-2, 2:-2, :], ff_drain_pot[2:-2, 2:-2, :]),
    )
    ff_drain_pot = update(
        ff_drain_pot,
        at[2:-2, 2:-2, :], npx.where((vs.tp_ff[2:-2, 2:-2, :] >= vs.ti_ff[2:-2, 2:-2, :]) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] > vs.ti_ff[2:-2, 2:-2, :]) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] < vs.t_end_ff[2:-2, 2:-2, :]), (vs.S_f[2:-2, 2:-2, :] / 2) * (vs.tw_ff[2:-2, 2:-2, :] - vs.ts_ff[2:-2, 2:-2, :])**(1/2) * (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] - vs.ts_ff[2:-2, 2:-2, :])**(-3/2), ff_drain_pot[2:-2, 2:-2, :]),
    )
    ff_drain_pot = update(
        ff_drain_pot,
        at[2:-2, 2:-2, :], npx.where((vs.tw_ff[2:-2, 2:-2, :] < vs.ts_ff[2:-2, 2:-2, :]) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] > vs.ti_ff[2:-2, 2:-2, :]) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] < vs.t_end_ff[2:-2, 2:-2, :]), (vs.S_f[2:-2, 2:-2, :] / 2) * (vs.tw_ff[2:-2, 2:-2, :] - vs.tb_ff[2:-2, 2:-2, :])**(1/2) * (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] - vs.tb_ff[2:-2, 2:-2, :])**(-3/2), ff_drain_pot[2:-2, 2:-2, :]),
    )
    ff_drain_pot = update(
        ff_drain_pot,
        at[2:-2, 2:-2, :], npx.where(vs.z_wf_ff[2:-2, 2:-2, :, vs.tau] < vs.z_soil[2:-2, 2:-2, npx.newaxis], 0, ff_drain_pot[2:-2, 2:-2, :]),
    )
    vs.ff_drain = update(
        vs.ff_drain,
        at[2:-2, 2:-2], npx.nansum(npx.where(vs.S_f[2:-2, 2:-2, :] < ff_drain_pot[2:-2, 2:-2, :], vs.S_f[2:-2, 2:-2, :], ff_drain_pot[2:-2, 2:-2, :]), axis=-1),
    )
    vs.ff_drain = update(
        vs.ff_drain,
        at[2:-2, 2:-2], npx.where(vs.ff_drain[2:-2, 2:-2] < 0, 0, vs.ff_drain[2:-2, 2:-2]),
    )
    vs.S_f = update_add(
        vs.S_f,
        at[2:-2, 2:-2, :], -npx.where(vs.S_f[2:-2, 2:-2, :] < ff_drain_pot[2:-2, 2:-2, :], vs.S_f[2:-2, 2:-2, :], ff_drain_pot[2:-2, 2:-2, :]),
    )

    return KernelOutput(
        ff_drain=vs.ff_drain,
        S_f=vs.S_f,
        )


@roger_kernel
def update_film_volume(state):
    """
    Calculates film flow drainage
    """
    vs = state.variables

    vs.S_f_rz = update(
        vs.S_f_rz,
        at[2:-2, 2:-2, :], npx.where(vs.z_wf_ff[2:-2, 2:-2, :, vs.tau] <= vs.z_root[2:-2, 2:-2, vs.tau, npx.newaxis], vs.S_f[2:-2, 2:-2, :], (vs.z_root[2:-2, 2:-2, vs.tau, npx.newaxis]/vs.z_wf_ff[2:-2, 2:-2, :, vs.tau]) * vs.S_f[2:-2, 2:-2, :]),
    )
    vs.S_f_ss = update(
        vs.S_f_ss,
        at[2:-2, 2:-2, :], npx.where(vs.z_wf_ff[2:-2, 2:-2, :, vs.tau] > vs.z_root[2:-2, 2:-2, vs.tau, npx.newaxis], ((vs.z_wf_ff[2:-2, 2:-2, :, vs.tau] - vs.z_root[2:-2, 2:-2, vs.tau, npx.newaxis]) / vs.z_wf_ff[2:-2, 2:-2, :, vs.tau]) * vs.S_f[2:-2, 2:-2, :], 0),
    )

    return KernelOutput(
        S_f_rz=vs.S_f_rz,
        S_f_ss=vs.S_f_ss,
        )


@roger_routine
def calculate_film_flow(state):
    """
    Calculates film flow
    """
    vs = state.variables
    settings = state.settings

    vs.itt_event_ff = update(
        vs.itt_event_ff,
        at[:], npx.where(vs.itt - vs.event_start_ff < settings.nittevent_ff, vs.itt - vs.event_start_ff, settings.nittevent_ff - 1),
    )

    cond1 = ((vs.event_id[vs.taum1] == 0) & (vs.event_id[vs.tau] >= 1))
    if cond1.any():
        vs.z_wf_ff = update(
            vs.z_wf_ff,
            at[2:-2, 2:-2, vs.event_no_ff - 1, :], 0,
        )
        vs.z_pf_ff = update(
            vs.z_pf_ff,
            at[2:-2, 2:-2, vs.event_no_ff - 1, :], 0,
        )
        #TODO: hortonian overland flow

        vs.update(calc_theta_d_rel(state))
        vs.update(calc_volume_flux_density(state))
        vs.update(calc_velocity(state))
        vs.update(calc_intersection(state))
        vs.update(calc_rain_pulse(state))
        vs.update(calc_t_end(state))
        vs.update(calc_intersection_at_soil_depth(state))

    vs.update(calc_infiltration(state))
    vs.update(calc_wetting_front_depth(state))
    vs.update(calc_percolation_front_depth(state))
    vs.update(calc_abstraction(state))
    vs.update(calc_drainage(state))
    vs.update(update_film_volume(state))
