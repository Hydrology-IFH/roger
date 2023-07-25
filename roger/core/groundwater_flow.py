from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at


@roger_kernel
def _tt_z(z, kf, bdec):
    return kf * npx.exp(-z/bdec)


@roger_kernel
def calc_q_gw(state):
    """
    Calculates lateral groundwater flow
    """
    vs = state.variables
    settings = state.settings

    # Calculates transmissivity
    dz = allocate(state.dimensions, ("x", "y"))
    z = allocate(state.dimensions, ("x", "y", 1001))
    z = update(
        z,
        at[2:-2, 2:-2, :], npx.linspace(vs.z_gw[2:-2, 2:-2, vs.tau], vs.z_gw_tot[2:-2, 2:-2], num=1001, axis=-1) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    dz = update(
        dz,
        at[2:-2, 2:-2], (z[2:-2, 2:-2, 1] - z[2:-2, 2:-2, 0]) * vs.maskCatch[2:-2, 2:-2],
    )

    # calculates transmissivity
    vs.tt = update(
        vs.tt,
        at[2:-2, 2:-2], (npx.sum(_tt_z(z[2:-2, 2:-2, :], vs.kf[2:-2, 2:-2, npx.newaxis]/1000, vs.bdec[2:-2, 2:-2, npx.newaxis]), axis=-1) * dz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # TODO: calculate gradient

    # calculates lateral groundwater flow
    vs.q_gw = update(
        vs.q_gw,
        at[2:-2, 2:-2], (vs.tt[2:-2, 2:-2] * vs.dz_gw[2:-2, 2:-2] * settings.dx * vs.dt) * (1000/settings.dx**2) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_gw = update_add(
        vs.S_gw,
        at[2:-2, 2:-2, vs.tau], -vs.q_gw[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # TODO: add to neighboring cells

    return KernelOutput(q_gw=vs.q_gw, tt=vs.tt, S_gw=vs.S_gw)


@roger_kernel
def calc_q_bf(state):
    """
    Calculates baseflow
    """
    vs = state.variables
    settings = state.settings

    # TODO: update groundwater table

    mask1 = (vs.z_gw > vs.z_stream_tot)
    vs.q_bf = update(
        vs.q_bf,
        at[2:-2, 2:-2], (vs.kf[2:-2, 2:-2] * vs.dz_gw[2:-2, 2:-2] * settings.dx * vs.dt) * (1000/settings.dx**2) * mask1[2:-2, 2:-2] * vs.maskRiver[2:-2, 2:-2],
    )

    vs.S_gw = update_add(
        vs.S_gw,
        at[2:-2, 2:-2, vs.tau], -vs.q_bf[2:-2, 2:-2] * vs.maskRiver[2:-2, 2:-2],
    )

    return KernelOutput(q_bf=vs.q_bf, S_gw=vs.S_gw)


@roger_kernel
def calc_q_re(state):
    """
    Calculates groundwater recharge
    """
    vs = state.variables
    settings = state.settings


    mask1 = (vs.z_gw[:, :, vs.tau] > vs.z_soil)
    vs.S_vad_tot = update(
        vs.S_vad_tot,
        at[2:-2, 2:-2, vs.tau], npx.where(mask1[2:-2, 2:-2], (vs.z_gw[2:-2, 2:-2, vs.tau] - vs.z_soil[2:-2, 2:-2]) * vs.n0[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_vad = update_add(
        vs.S_vad,
        at[2:-2, 2:-2, vs.tau], vs.q_ss[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # linear reservoir
    k = allocate(state.dimensions, ("x", "y"))
    k = update(
        k,
        at[2:-2, 2:-2], (vs.kf[2:-2, 2:-2] / settings.kf_max) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.q_re = update(
        vs.q_re,
        at[2:-2, 2:-2], k[2:-2, 2:-2, npx.newaxis] * vs.S_vad[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_vad = update_add(
        vs.S_vad,
        at[2:-2, 2:-2, vs.tau], -vs.q_re[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # spill over of linear reservoir
    mask2 = (vs.S_vad[:, :, vs.tau] > vs.S_vad_tot[:, :, vs.tau])
    spillover = allocate(state.dimensions, ("x", "y"))
    spillover = update(
        spillover,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], (vs.S_vad[2:-2, 2:-2, vs.tau] - vs.S_vad_tot[2:-2, 2:-2]), 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.q_re = update_add(
        vs.q_re,
        at[2:-2, 2:-2], spillover[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_vad = update_add(
        vs.S_vad,
        at[2:-2, 2:-2, vs.tau], -vs.spillover[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(q_re=vs.q_re, S_vad=vs.S_vad, S_vad_tot=vs.S_vad_tot)


@roger_kernel
def calc_q_leak(state):
    """
    Calculates groundwater leakage
    """
    vs = state.variables
    settings = state.settings

    vs.q_leak = update(
        vs.q_leak ,
        at[2:-2, 2:-2], (vs.k_leak[2:-2, 2:-2] * settings.dx * vs.dt) * (1000/settings.dx**2) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_gw = update_add(
        vs.S_gw,
        at[2:-2, 2:-2, vs.tau], -vs.q_leak[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(q_leak=vs.q_leak, S_gw=vs.S_gw)


@roger_kernel
def update_storage_bc(state):
    """
    Updates groundwater storage with boundary conditions
    """
    pass


@roger_kernel
def calc_groundwater_runoff_routing(state):
    """
    Calculates groundwater runoff routing
    """
    pass


@roger_routine
def calculate_groundwater_recharge(state):
    """
    Calculates groundwater recharge
    """
    vs = state.variables

    vs.update(calc_q_re(state))

@roger_routine
def calculate_groundwater_flow(state):
    """
    Calculates groundwater recharge and lateral flow
    """
    vs = state.variables

    vs.update(calc_q_re(state))
    vs.update(calc_q_gw(state))
    vs.update(calc_q_leak(state))
