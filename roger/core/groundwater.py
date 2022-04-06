from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, at


@roger_kernel
def _ss_z(z, n0, bdec):
    return n0 * npx.exp(-z/bdec)


def _z(z, z_gw_tot, n0, bdec, S_gw):
    return -bdec * n0 * npx.exp(-z_gw_tot/bdec) + bdec * n0 * npx.exp(-z_gw_tot/bdec) - (S_gw/1000)


def _zs(z_init, z_gw_tot, n0, bdec, S_gw):
    from scipy.optimize import fsolve
    import numpy as onp
    return onp.vectorize(fsolve(_z, x0=z_init, args=(z_gw_tot, n0, bdec, S_gw,)))


@roger_kernel
def calc_S_gw_from_z_gw(state):
    """
    Calculates groundwater storage
    """
    vs = state.variables

    # Calculates transmissivity
    dz = allocate(state.dimensions, ("x", "y"))
    z = allocate(state.dimensions, ("x", "y", 1001))
    z = update(
        z,
        at[:, :, :], npx.linspace(vs.z_gw[:, :, vs.tau], vs.z_gw_tot, num=1001) * vs.maskCatch,
    )

    dz = update(
        dz,
        at[:, :, :], (z[:, :, 1] - z[:, :, 0]) * vs.maskCatch,
    )

    vs.S_gw = update(
        vs.S_gw,
        at[:, :, vs.tau], (npx.sum(_ss_z(z, vs.n0[:, :, npx.newaxis], vs.bdec[:, :, npx.newaxis]), axis=2) * (dz/1001)) * vs.maskCatch,
    )

    return KernelOutput(S_gw=vs.S_gw)


@roger_kernel
def calc_z_gw(state):
    """
    Calculates groundwater head
    """
    vs = state.variables

    vs.z_gw = update(
        vs.z_gw,
        at[:, :, vs.tau], _zs(vs.z_gw[:, :, vs.taum1], vs.z_gw_tot, vs.n0, vs.bdec, vs.S_gw[:, :, vs.tau]) * vs.maskCatch,
    )

    return KernelOutput(z_gw=vs.z_gw)


@roger_routine
def calculate_groundwater(state):
    """
    Calculates groundwater storage
    """
    vs = state.variables
    vs.update(calc_z_gw(state))
