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
        at[:, :], ((vs.theta_sat - theta) / (vs.theta_sat - vs.theta_pwp)) * vs.maskCatch,
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
        at[:, :, :], npx.arange(0, settings.nittevent_ff, 1)[npx.newaxis, npx.newaxis, :],
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
        at[:, :, :], npx.arange(0, settings.nittevent_ff, 1)[npx.newaxis, npx.newaxis, :],
    )

    vs.t_end_ff = update(
        vs.t_end_ff,
        at[:, :, vs.event_no_ff - 1], npx.min(npx.where(itt_event > vs.ts_ff[:, :, vs.event_no_ff - 1, npx.newaxis], npx.where(vs.rain_int_ff[:, :, vs.event_no_ff - 1, npx.newaxis] * ((vs.ti_ff[:, :, vs.event_no_ff - 1, npx.newaxis] - vs.ts_ff[:, :, vs.event_no_ff - 1, npx.newaxis]) / (itt_event - vs.ts_ff[:, :, vs.event_no_ff - 1, npx.newaxis]))**(3/2) <= vs.rain_int_ff[:, :, vs.event_no_ff - 1, npx.newaxis] * settings.ff_tc, itt_event, settings.nittevent_ff), settings.nittevent_ff), axis=-1),
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
        at[:, :, :], npx.arange(0, settings.nittevent_ff, 1)[npx.newaxis, npx.newaxis, :],
    )

    # define lower and upper quartile of rainfall
    idx_rain_25 = allocate(state.dimensions, ("x", "y"))
    idx_rain_75 = allocate(state.dimensions, ("x", "y"))
    idx_rain_25 = update(
        idx_rain_25,
        at[:, :], npx.max(npx.where(vs.rain_event_csum <= 0.25 * vs.rain_event_sum[:, :, npx.newaxis], itt_event, 0), axis=-1),
    )
    idx_rain_75 = update(
        idx_rain_75,
        at[:, :], npx.min(npx.where(vs.rain_event_csum >= 0.75 * vs.rain_event_sum[:, :, npx.newaxis], itt_event, settings.nittevent_ff), axis=-1),
    )

    idx_reg = allocate(state.dimensions, ("x", "y", "timesteps_event_ff"))
    rain_reg = allocate(state.dimensions, ("x", "y", "timesteps_event_ff"))
    rain_csum_reg = allocate(state.dimensions, ("x", "y", "timesteps_event_ff"))
    rain_init = allocate(state.dimensions, ("x", "y"))
    params_reg = allocate(state.dimensions, ("x", "y", 2))
    idx_reg = update(
        idx_reg,
        at[:, :, :], npx.where((itt_event >= idx_rain_25[:, :, npx.newaxis]) & (itt_event <= idx_rain_75[:, :, npx.newaxis]), itt_event, 0),
    )
    rain_init = update(
        rain_init,
        at[:, :], npx.max(npx.where(vs.rain_event_csum <= 0.25 * vs.rain_event_sum[:, :, npx.newaxis], vs.rain_event_csum, 0), axis=-1),
    )
    rain_reg = update(
        rain_reg,
        at[:, :, :], npx.where((itt_event >= idx_rain_25[:, :, npx.newaxis]) & (itt_event <= idx_rain_75[:, :, npx.newaxis]), vs.rain_event, 0),
    )
    rain_csum_reg = update(
        rain_csum_reg,
        at[:, :, :], npx.cumsum(rain_reg, axis=-1) + rain_init[:, :, npx.newaxis],
    )
    # linear regression to determine volume flux density
    params_reg = update(
        params_reg,
        at[:, :, :], linear_regression(idx_reg, rain_csum_reg, params_reg),
    )

    vs.qs_ff = update(
        vs.qs_ff,
        at[:, :, vs.event_no_ff - 1], params_reg[:, :, 0]/600/1000,
    )

    vs.tb_ff = update(
        vs.tb_ff,
        at[:, :, vs.event_no_ff - 1], npx.where(-params_reg[:, :, 1]/params_reg[:, :, 0] >= 0, -params_reg[:, :, 1]/params_reg[:, :, 0], 0),
    )

    vs.ts_ff = update(
        vs.ts_ff,
        at[:, :, vs.event_no_ff - 1], vs.tb_ff[:, :, vs.event_no_ff - 1] + (vs.rain_event_sum - params_reg[:, :, 1])/params_reg[:, :, 0],
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
        at[:, :, :], npx.arange(0, settings.nittevent_ff, 1)[npx.newaxis, npx.newaxis, :],
    )

    ts = allocate(state.dimensions, ("x", "y"))
    tb = allocate(state.dimensions, ("x", "y"))
    ts = update(
        ts,
        at[:, :], (vs.ts_ff[:, :, vs.event_no_ff - 1] - vs.ts_ff[:, :, vs.event_no_ff - 1] % 1) + 1,
    )
    tb = update(
        tb,
        at[:, :], npx.where(vs.tb_ff[:, :, vs.event_no_ff - 1] - vs.tb_ff[:, :, vs.event_no_ff - 1] % 1 > 0, vs.tb_ff[:, :, vs.event_no_ff - 1] - vs.tb_ff[:, :, vs.event_no_ff - 1] % 1, 0)
    )

    vs.rain_int_ff = update(
        vs.rain_int_ff,
        at[:, :, vs.event_no_ff - 1], (vs.qs_ff[:, :, vs.event_no_ff - 1] * 600 * 1000 * (vs.ts_ff[:, :, vs.event_no_ff - 1] - vs.tb_ff[:, :, vs.event_no_ff - 1])) / (ts - tb),
    )

    vs.rain_event_ff = update(
        vs.rain_event_ff,
        at[:, :, :], npx.where((itt_event >= tb[:, :, npx.newaxis]) & (itt_event <= ts[:, :, npx.newaxis]), vs.rain_int_ff[:, :, vs.event_no_ff - 1, npx.newaxis], 0),
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
        at[:, :, vs.event_no_ff - 1], vs.a_ff * vs.qs_ff[:, :, vs.event_no_ff - 1]**(2/3) * 600 * 1000,
    )

    vs.v_perc = update(
        vs.v_perc,
        at[:, :, vs.event_no_ff - 1], vs.v_wf[:, :, vs.event_no_ff - 1] * 3,
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
        at[:, :, vs.event_no_ff - 1], vs.tb_ff[:, :, vs.event_no_ff - 1] + 0.5 * (3 * (vs.ts_ff[:, :, vs.event_no_ff - 1] - vs.tb_ff[:, :, vs.event_no_ff - 1])),
    )

    vs.zi_ff = update(
        vs.zi_ff,
        at[:, :, vs.event_no_ff - 1], ((3 * vs.v_wf[:, :, vs.event_no_ff - 1]) / 2) * (vs.ts_ff[:, :, vs.event_no_ff - 1] - vs.tb_ff[:, :, vs.event_no_ff - 1]),
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
        at[:, :, vs.event_no_ff - 1], vs.tb_ff[:, :, vs.event_no_ff - 1] + vs.z_soil / vs.v_wf[:, :, vs.event_no_ff - 1],
    )

    vs.tp_ff = update(
        vs.tp_ff,
        at[:, :, vs.event_no_ff - 1], vs.ts_ff[:, :, vs.event_no_ff - 1] + vs.z_soil / vs.v_perc[:, :, vs.event_no_ff - 1],
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
        at[:, :], vs.rain_event_ff[:, :, vs.itt_event_ff[vs.event_no_ff - 1]],
    )
    vs.prec = update_add(
        vs.prec,
        at[:, :], vs.rain_ff,
    )
    vs.S_f = update_add(
        vs.S_f,
        at[:, :, vs.event_no_ff - 1], vs.rain_ff,
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
        at[:, :, :, vs.tau], npx.where((vs.itt_event_ff[npx.newaxis, npx.newaxis, :] > vs.tb_ff) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] < vs.ti_ff) & (vs.S_f > 0), vs.v_wf * (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] - vs.tb_ff), vs.z_wf_ff[:, :, :, vs.tau]),
    )

    vs.z_wf_ff = update(
        vs.z_wf_ff,
        at[:, :, :, vs.tau], npx.where((vs.itt_event_ff[npx.newaxis, npx.newaxis, :] > vs.ti_ff) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] < vs.t_end_ff) & (vs.S_f > 0), vs.v_perc * (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] - vs.ts_ff)**(1/3) * ((vs.ts_ff - vs.tb_ff) / 2)**(2/3), vs.z_wf_ff[:, :, :, vs.tau]),
    )

    vs.z_wf_ff = update(
        vs.z_wf_ff,
        at[:, :, :, :], npx.where(vs.itt_event_ff[npx.newaxis, npx.newaxis, :, npx.newaxis] >= vs.t_end_ff[:, :, :, npx.newaxis], 0, vs.z_wf_ff),
    )

    vs.z_wf = update(
        vs.z_wf,
        at[:, :, vs.tau], npx.max(vs.z_wf_ff[:, :, :, vs.tau], axis=2),
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
        at[:, :, :, vs.tau], npx.where((vs.itt_event_ff[npx.newaxis, npx.newaxis, :] > vs.ts_ff) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] <= vs.ti_ff) & (vs.S_f > 0), vs.v_perc * (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] - vs.ts_ff), vs.z_pf_ff[:, :, :, vs.tau]),
    )

    vs.z_pf_ff = update(
        vs.z_pf_ff,
        at[:, :, :, vs.tau], npx.where(vs.z_pf_ff[:, :, :, vs.tau] > vs.z_soil[:, :, npx.newaxis], vs.z_soil[:, :, npx.newaxis], vs.z_pf_ff[:, :, :, vs.tau]),
    )

    vs.z_pf_ff = update(
        vs.z_pf_ff,
        at[:, :, :, :], npx.where(vs.itt_event_ff[npx.newaxis, npx.newaxis, :, npx.newaxis] >= vs.t_end_ff[:, :, :, npx.newaxis], 0, vs.z_pf_ff),
    )

    vs.z_pf = update(
        vs.z_pf,
        at[:, :, vs.tau], npx.max(vs.z_pf_ff[:, :, :, vs.tau], axis=2),
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
        at[:, :, :], npx.where(((vs.z_wf_ff[:, :, :, vs.tau] - vs.z_wf_ff[:, :, :, vs.taum1]) > 0) & (vs.z_wf_ff[:, :, :, vs.tau] < vs.z_root[:, :, vs.tau, npx.newaxis]), vs.theta_d_rel_rz_ff * vs.wfs[:, :, npx.newaxis] * vs.ks[:, :, npx.newaxis] * vs.dt * ((vs.wfs[:, :, npx.newaxis] + (vs.z_wf_ff[:, :, :, vs.tau] - vs.z_wf_ff[:, :, :, vs.taum1])) / (vs.z_wf_ff[:, :, :, vs.tau] - vs.z_wf_ff[:, :, :, vs.taum1])) * vs.c_ff[:, :, npx.newaxis], 0),
    )
    vs.ff_abs_rz = update(
        vs.ff_abs_rz,
        at[:, :, :], npx.where(vs.ff_abs_rz >= vs.S_f, vs.S_f, vs.ff_abs_rz),
    )
    vs.S_f = update_add(
        vs.S_f,
        at[:, :, :], npx.where(vs.ff_abs_rz > 0, -vs.ff_abs_rz, 0),
    )

    vs.ff_abs_ss = update(
        vs.ff_abs_ss,
        at[:, :, :], npx.where(((vs.z_wf_ff[:, :, :, vs.tau] - vs.z_wf_ff[:, :, :, vs.taum1]) > 0) & (vs.z_wf_ff[:, :, :, vs.tau] >= vs.z_root[:, :, vs.tau, npx.newaxis]) & (vs.z_wf_ff[:, :, :, vs.tau] <= vs.z_soil[:, :, npx.newaxis]), vs.theta_d_rel_ss_ff * vs.wfs[:, :, npx.newaxis] * vs.ks[:, :, npx.newaxis] * vs.dt * ((vs.wfs[:, :, npx.newaxis] + (vs.z_wf_ff[:, :, :, vs.tau] - vs.z_wf_ff[:, :, :, vs.taum1])) / (vs.z_wf_ff[:, :, :, vs.tau] - vs.z_wf_ff[:, :, :, vs.taum1])) * vs.c_ff[:, :, npx.newaxis], 0),
    )
    vs.ff_abs_ss = update(
        vs.ff_abs_ss,
        at[:, :, :], npx.where(vs.ff_abs_ss >= vs.S_f, vs.S_f, vs.ff_abs_ss),
    )
    vs.ff_abs_ss = update(
        vs.ff_abs_ss,
        at[:, :, :], npx.where(vs.ff_abs_ss >= vs.S_f, vs.S_f, vs.ff_abs_ss),
    )
    vs.S_f = update_add(
        vs.S_f,
        at[:, :, :], npx.where(vs.ff_abs_ss > 0, -vs.ff_abs_ss, 0),
    )

    # abstraction of the residual film at the end of the event
    vs.ff_abs_rz = update(
        vs.ff_abs_rz,
        at[:, :, :], npx.where(vs.itt_event_ff[npx.newaxis, npx.newaxis, :] >= vs.t_end_ff, vs.S_f_rz, vs.ff_abs_rz),
    )
    vs.ff_abs_ss = update(
        vs.ff_abs_ss,
        at[:, :, :], npx.where(vs.itt_event_ff[npx.newaxis, npx.newaxis, :] >= vs.t_end_ff, vs.S_f_ss, vs.ff_abs_ss),
    )
    vs.S_f = update(
        vs.S_f,
        at[:, :, :], npx.where(vs.itt_event_ff[npx.newaxis, npx.newaxis, :] >= vs.t_end_ff, 0, vs.S_f),
    )

    vs.ff_abs = update(
        vs.ff_abs,
        at[:, :, :], vs.ff_abs_rz + vs.ff_abs_ss,
    )

    # update root zone storage after abtraction from film flow
    vs.S_fp_rz = update_add(
        vs.S_fp_rz,
        at[:, :], npx.sum(vs.ff_abs_rz, axis=-1) * vs.maskCatch,
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

    # update subsoil storage after abtraction from film flow
    vs.S_fp_ss = update_add(
        vs.S_fp_ss,
        at[:, :], npx.sum(vs.ff_abs_ss, axis=-1) * vs.maskCatch,
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
        at[:, :, :], npx.where((vs.tp_ff < vs.ti_ff) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] >= vs.tw_ff) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] <= vs.tp_ff), vs.rain_int_ff, 0),
    )
    ff_drain_pot = update(
        ff_drain_pot,
        at[:, :, :], npx.where((vs.tp_ff < vs.ti_ff) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] > vs.tp_ff) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] < vs.ti_ff), vs.rain_int_ff * (vs.tp_ff - vs.ts_ff) / (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] - vs.ts_ff)**(3/2), ff_drain_pot),
    )
    ff_drain_pot = update(
        ff_drain_pot,
        at[:, :, :], npx.where((vs.tp_ff < vs.ti_ff) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] > vs.ti_ff) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] < vs.t_end_ff), (vs.S_f / 2) * (vs.tw_ff - vs.ts_ff)**(1/2) * (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] - vs.ts_ff)**(-3/2), ff_drain_pot),
    )
    ff_drain_pot = update(
        ff_drain_pot,
        at[:, :, :], npx.where((vs.tp_ff >= vs.ti_ff) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] >= vs.tw_ff) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] <= vs.ti_ff), vs.rain_int_ff, ff_drain_pot),
    )
    ff_drain_pot = update(
        ff_drain_pot,
        at[:, :, :], npx.where((vs.tp_ff >= vs.ti_ff) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] > vs.ti_ff) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] < vs.t_end_ff), (vs.S_f / 2) * (vs.tw_ff - vs.ts_ff)**(1/2) * (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] - vs.ts_ff)**(-3/2), ff_drain_pot),
    )
    ff_drain_pot = update(
        ff_drain_pot,
        at[:, :, :], npx.where((vs.tw_ff < vs.ts_ff) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] > vs.ti_ff) & (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] < vs.t_end_ff), (vs.S_f / 2) * (vs.tw_ff - vs.tb_ff)**(1/2) * (vs.itt_event_ff[npx.newaxis, npx.newaxis, :] - vs.tb_ff)**(-3/2), ff_drain_pot),
    )
    ff_drain_pot = update(
        ff_drain_pot,
        at[:, :, :], npx.where(vs.z_wf_ff[:, :, :, vs.tau] < vs.z_soil[:, :, npx.newaxis], 0, ff_drain_pot),
    )
    vs.ff_drain = update(
        vs.ff_drain,
        at[:, :], npx.nansum(npx.where(vs.S_f < ff_drain_pot, vs.S_f, ff_drain_pot), axis=-1),
    )
    vs.ff_drain = update(
        vs.ff_drain,
        at[:, :], npx.where(vs.ff_drain < 0, 0, vs.ff_drain),
    )
    vs.S_f = update_add(
        vs.S_f,
        at[:, :, :], -npx.where(vs.S_f < ff_drain_pot, vs.S_f, ff_drain_pot),
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
        at[:, :, :], npx.where(vs.z_wf_ff[:, :, :, vs.tau] <= vs.z_root[:, :, vs.tau, npx.newaxis], vs.S_f, (vs.z_root[:, :, vs.tau, npx.newaxis]/vs.z_wf_ff[:, :, :, vs.tau]) * vs.S_f),
    )
    vs.S_f_ss = update(
        vs.S_f_ss,
        at[:, :, :], npx.where(vs.z_wf_ff[:, :, :, vs.tau] > vs.z_root[:, :, vs.tau, npx.newaxis], ((vs.z_wf_ff[:, :, :, vs.tau] - vs.z_root[:, :, vs.tau, npx.newaxis]) / vs.z_wf_ff[:, :, :, vs.tau]) * vs.S_f, 0),
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
        at[:], npx.where(vs.itt - vs.event_start_ff <= settings.nittevent_ff, vs.itt - vs.event_start_ff, settings.nittevent_ff),
    )

    arr_itt = allocate(state.dimensions, ("x", "y"))
    arr_itt = update(
        arr_itt,
        at[:, :], vs.itt,
    )
    cond1 = ((vs.EVENT_ID_FF[:, :, vs.itt-1] == 0) & (vs.EVENT_ID_FF[:, :, vs.itt] >= 1) & (arr_itt >= 1))

    if cond1.any():
        vs.z_wf_ff = update(
            vs.z_wf_ff,
            at[:, :, vs.event_no_ff - 1, :], 0,
        )
        vs.z_pf_ff = update(
            vs.z_pf_ff,
            at[:, :, vs.event_no_ff - 1, :], 0,
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
