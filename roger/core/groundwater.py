from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core import numerics
from roger.core.operators import numpy as npx, update, at
import numpy as onp


@roger_kernel
def _ss_z(z, n0, bdec):
    return n0 * npx.exp(-z/bdec)


def _z(z, z_gw_tot, n0, bdec, S_gw):
    return -bdec * n0 * npx.exp(-z_gw_tot/bdec) + bdec * n0 * npx.exp(-z/bdec) - S_gw


def _zs(z_init, z_gw_tot, n0, bdec, S_gw):
    from scipy.optimize import fsolve
    return fsolve(_z, x0=z_init, args=(z_gw_tot, n0, bdec, S_gw,))


# vectorize solution of storativity equation
_zsv = onp.vectorize(_zs)


@roger_kernel
def calc_S_gw_from_z_gw(state):
    """
    Calculates groundwater storage
    """
    vs = state.variables

    # Calculates storativity
    dz = allocate(state.dimensions, ("x", "y"))
    z = allocate(state.dimensions, ("x", "y", 1001))
    z = update(
        z,
        at[2:-2, 2:-2, :], npx.linspace(vs.z_gw[2:-2, 2:-2, vs.tau], vs.z_gw_tot, num=1001, axis=-1) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    dz = update(
        dz,
        at[2:-2, 2:-2], (z[2:-2, 2:-2, 1] - z[2:-2, 2:-2, 0]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_gw = update(
        vs.S_gw,
        at[2:-2, 2:-2, vs.tau], (npx.sum(_ss_z(z, vs.n0[2:-2, 2:-2, npx.newaxis], vs.bdec[2:-2, 2:-2, npx.newaxis]), axis=-1) * dz[2:-2, 2:-2]) * 1000 * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(S_gw=vs.S_gw)


@roger_kernel
def calc_z_gw(state):
    """
    Calculates depth of groundwater table
    """
    vs = state.variables

    vs.z_gw = update(
        vs.z_gw,
        at[2:-2, 2:-2, vs.tau], _zsv(vs.z_gw[2:-2, 2:-2, vs.taum1], vs.z_gw_tot[2:-2, 2:-2], vs.n0[2:-2, 2:-2], vs.bdec[2:-2, 2:-2], vs.S_gw[2:-2, 2:-2, vs.tau]/1000) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(z_gw=vs.z_gw)


@roger_routine
def calculate_groundwater(state):
    """
    Calculates groundwater storage
    """
    vs = state.variables
    vs.update(calc_z_gw(state))


@roger_kernel
def calc_topo_kernel(state):
    vs = state.variables
    """
    Boundary mask
    """
    grid = allocate(state.dimensions, ("x", "y"))
    catch = allocate(state.dimensions, ("x", "y"))
    catch = update(catch, at[...], npx.where(vs.maskCatch, 1, 0))
    vs.maskBoundGw = update(vs.maskBoundGw, at[...], npx.where(grid - catch < 0, True, False))

    return KernelOutput(
        maskBoundGw=vs.maskBoundGw,
    )


@roger_kernel
def calc_parameters_groundwater_kernel(state):
    pass


@roger_routine
def calculate_parameters(state):
    vs = state.variables
    settings = state.settings

    if not settings.enable_offline_transport:
        numerics.validate_parameters_groundwater(state)
        vs.update(calc_topo_kernel(state))
        vs.update(calc_parameters_groundwater_kernel(state))


@roger_kernel
def calc_initial_conditions_groundwater_kernel(state):
    vs = state.variables

    # Calculates storativity
    dz = allocate(state.dimensions, ("x", "y"))
    z = allocate(state.dimensions, ("x", "y", 1001))
    z = update(
        z,
        at[2:-2, 2:-2, :], npx.linspace(vs.z_gw[2:-2, 2:-2, vs.tau], vs.z_gw_tot, num=1001, axis=-1) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    dz = update(
        dz,
        at[2:-2, 2:-2], (z[2:-2, 2:-2, 1] - z[2:-2, 2:-2, 0]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_gw = update(
        vs.S_gw,
        at[2:-2, 2:-2, vs.taum1], (npx.sum(_ss_z(z, vs.n0[2:-2, 2:-2, npx.newaxis], vs.bdec[2:-2, 2:-2, npx.newaxis]), axis=-1) * dz) * 1000 * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_gw = update(
        vs.S_gw,
        at[2:-2, 2:-2, vs.tau], vs.S_gw[2:-2, 2:-2, vs.taum1],
    )

    return KernelOutput(S_gw=vs.S_gw)


@roger_routine
def calculate_initial_conditions(state):
    """
    Calculates initial conditions of groundwater
    """
    vs = state.variables
    settings = state.settings

    if not settings.enable_offline_transport:
        numerics.validate_initial_conditions_groundwater(state)
        vs.update(calc_initial_conditions_groundwater_kernel(state))
