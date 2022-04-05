from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, at


@roger_kernel
def _ss_z(z, n0, bdec):
    return n0 * npx.exp(-z/bdec)


@roger_kernel
def calc_S_gw(state):
    """
    Calculates groundwater storage
    """
    vs = state.variables
    settings = state.settings

    # Calculates transmissivity
    dz = allocate(state.dimensions, ("x", "y"))
    z = npx.zeros((settings.nx, settings.ny, 1001), dtype=dz.float_type)
    z = update(
        z,
        at[:, :, :], npx.linspace(vs.z_gw, vs.z_gw_tot, num=1001) * vs.maskCatch,
    )

    dz = update(
        dz,
        at[:, :, :], (z[:, :, 1] - z[:, :, 0]) * vs.maskCatch,
    )

    vs.S_gw = update(
        vs.S_gw,
        at[:, :, vs.tau], (npx.sum(_ss_z(z, vs.n0[:, :, npx.newaxis], vs.bdec[:, :, npx.newaxis]), axis=2) * (dz/1001)) * vs.maskCatch,
    )

    return KernelOutput(S_gw=vs.S_gw, tt=vs.tt)


@roger_kernel
def calc_dS(state):
    """
    Calculates storage change
    """
    vs = state.variables

    vs.dS_gw = update(
        vs.dS_gw,
        at[:, :, vs.tau], (vs.S_gw[:, :, vs.tau] - vs.S_gw[:, :, vs.taum1]) * vs.maskCatch,
    )

    return KernelOutput(dS_gw=vs.dS_gw)


@roger_routine
def calculate_groundwater(state):
    """
    Calculates groundwater storage
    """
    vs = state.variables
    vs.update(calc_dS(state))
