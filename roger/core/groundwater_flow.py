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
    z = npx.zeros((settings.nx, settings.ny, 1001), dtype=dz.dtype)
    z = update(
        z,
        at[:, :, :], npx.linspace(vs.z_gw, vs.z_gw_tot, num=1001) * vs.maskCatch,
    )

    dz = update(
        dz,
        at[:, :, :], (z[:, :, 1] - z[:, :, 0]) * vs.maskCatch,
    )

    vs.tt = update(
        vs.tt,
        at[:, :, :], (npx.sum(_tt_z(z, vs.kf[:, :, npx.newaxis], vs.bdec[:, :, npx.newaxis]), axis=2) * (dz/1001)) * vs.maskCatch,
    )

    # TODO: calculate gradient

    # calculates lateral groundwater flow
    vs.q_gw = update(
        vs.q_gw,
        at[:, :, :], (vs.tt * vs.beta * settings.dx * vs.dt) * (1000/settings.dx**2) * vs.maskCatch,
    )

    vs.S_gw = update_add(
        vs.S_gw,
        at[:, :, :], -vs.q_gw * vs.maskCatch,
    )

    # TODO: add to neighboring cells

    return KernelOutput(q_gw=vs.q_gw, tt=vs.tt)


@roger_kernel
def calc_q_bf(state):
    """
    Calculates baseflow
    """
    vs = state.variables
    settings = state.settings

    # TODO: update groundwater tabel

    mask1 = (vs.z_gw > vs.z_stream_tot)
    vs.q_bf = update(
        vs.q_bf,
        at[:, :, :], (vs.kf * vs.beta * settings.dx * vs.dt) * (1000/settings.dx**2) * mask1 * vs.maskRiver,
    )

    vs.S_gw = update_add(
        vs.S_gw,
        at[:, :, :], -vs.q_bf * vs.maskRiver,
    )

    return KernelOutput(q_bf=vs.q_bf, S_gw=vs.S_gw)


@roger_kernel
def calc_q_re(state):
    """
    Calculates groundwater recharge
    """
    vs = state.variables

    vs.S_gw = update_add(
        vs.S_gw,
        at[:, :, :], vs.q_re * vs.maskCatch,
    )

    return KernelOutput(q_re=vs.q_re, S_gw=vs.S_gw)


@roger_kernel
def calc_q_leak(state):
    """
    Calculates groundwater leakage
    """
    vs = state.variables
    settings = state.settings

    vs.q_leak = update(
        vs.q_leak ,
        at[:, :, :], (vs.k_leak * settings.dx * vs.dt) * (1000/settings.dx**2) * vs.maskCatch,
    )

    vs.S_gw = update_add(
        vs.S_gw,
        at[:, :, :], -vs.q_leak * vs.maskCatch,
    )

    return KernelOutput(q_leak=vs.q_leak, S_gw=vs.S_gw)


@roger_kernel
def update_storage_bc(state):
    """
    Updates groundwater storage with boundary conditions
    """
    vs = state.variables

    return KernelOutput(S_gw=vs.S_gw)


@roger_routine
def calculate_groundwater(state):
    """
    Calculates groundwater
    """
    vs = state.variables

    vs.update(update_storage_bc(state))
    vs.update(calc_q_re(state))
    vs.update(calc_q_gw(state))
    vs.update(calc_q_bf(state))
    vs.update(calc_q_leak(state))
