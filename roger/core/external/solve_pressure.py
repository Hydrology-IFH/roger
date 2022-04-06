"""
solve two dimensional Possion equation
    A * dz0 = forc,  where A = nabla_h^2
with Neumann boundary conditions
used for surface pressure or free surface
method same as pressure method in MITgcm
"""


from veros import veros_routine
from veros.routines import veros_kernel
from veros.state import KernelOutput
from veros.variables import allocate
from veros.core import utilities as mainutils
from veros.core.operators import update, update_add, at, for_loop
from veros.core.operators import numpy as npx
from veros.core.external.solvers import get_linear_solver


@veros_routine
def solve_pressure(state):
    vs = state.variables
    state_update, forc = prepare_forcing(state)
    vs.update(state_update)

    linear_solver = get_linear_solver(state)
    linear_sol = linear_solver.solve(state, forc, vs.z0[..., vs.taup1])
    linear_sol = mainutils.enforce_boundaries(linear_sol)

    if vs.itt == 0:
        vs.z0 = update(vs.z0, at[...], linear_sol[..., npx.newaxis])
    else:
        vs.z0 = update(vs.z0, at[..., vs.taup1], linear_sol)

    vs.update(barotropic_velocity_update(state))


@veros_kernel
def prepare_forcing(state):
    vs = state.variables
    settings = state.settings

    # add hydrostatic pressure gradient
    vs.dvx = update_add(
        vs.dvx,
        at[2:-2, 2:-2, :, vs.tau],
        -(vs.z0[3:-1, 2:-2, :] - vs.z0[2:-2, 2:-2, :])
        * settings.dx)
        * vs.maskCatch[2:-2, 2:-2, :],
    )
    vs.dvy = update_add(
        vs.dvy,
        at[2:-2, 2:-2, :, vs.tau],
        -(vs.z0[2:-2, 3:-1, :] - vs.z0[2:-2, 2:-2, :])
        / settings.dy
        * vs.maskCatch[2:-2, 2:-2, :],
    )

    # Integrate forward in time
    vs.vx = update(
        vs.vx,
        at[:, :, :, vs.taup1],
        vs.vx[:, :, :, vs.tau]
        + settings.dt_mom
        * (
              (1.5 + settings.AB_eps) * vs.dvx[:, :, :, vs.tau]
            - (0.5 + settings.AB_eps) * vs.dvx[:, :, :, vs.taum1]
        )
        * vs.maskCatch,
    )

    vs.vy = update(
        vs.vy,
        at[:, :, :, vs.taup1],
        vs.vy[:, :, :, vs.tau]
        + settings.dt_mom
        * (
              (1.5 + settings.AB_eps) * vs.dvy[:, :, :, vs.tau]
            - (0.5 + settings.AB_eps) * vs.dvy[:, :, :, vs.taum1]
        )
        * vs.maskCatch,
    )

    # forcing for surface pressure
    uloc = allocate(state.dimensions, ("x", "y"))
    vloc = allocate(state.dimensions, ("x", "y"))

    uloc = update(
        uloc,
        at[2:-2, 2:-2],
        npx.sum((vs.vx[2:-2, 2:-2, :, vs.taup1]) * vs.maskCatch[2:-2, 2:-2, :], axis=(2,)) / settings.dt_mom,
    )
    vloc = update(
        vloc,
        at[2:-2, 2:-2],
        npx.sum((vs.vy[2:-2, 2:-2, :, vs.taup1]) * vs.maskCatch[2:-2, 2:-2, :], axis=(2,)) / settings.dt_mom,
    )

    uloc = mainutils.enforce_boundaries(uloc)
    vloc = mainutils.enforce_boundaries(vloc)

    forc = allocate(state.dimensions, ("x", "y"))

    forc = update(
        forc,
        at[2:-2, 2:-2],
        (uloc[2:-2, 2:-2] - uloc[1:-3, 2:-2]) / settings.dx)
        + vloc[2:-2, 1:-3] / settings.dy
        # free surface
        - vs.z0[2:-2, 2:-2, vs.tau]
        * vs.maskCatch[2:-2, 2:-2, -1],
    )

    # first guess
    vs.z0 = update(vs.z0, at[:, :, vs.taup1], 2 * vs.z0[:, :, vs.tau] - vs.z0[:, :, vs.taum1])

    return KernelOutput(dvx=vs.dvx, dvy=vs.dvy, vx=vs.vx, vy=vs.vy, z0=vs.z0), forc


@veros_kernel
def barotropic_velocity_update(state):
    """
    solve for surface ponding
    """
    vs = state.variables
    settings = state.settings

    vs.vx = update_add(
        vs.vx,
        at[2:-2, 2:-2, :, vs.taup1],
        - settings.dt_mom
        * (vs.z0[3:-1, 2:-2, vs.taup1, npx.newaxis] - vs.z0[2:-2, 2:-2, vs.taup1, npx.newaxis])
        / settings.dx
        * vs.maskCatch[2:-2, 2:-2, :],
    )

    vs.vy = update_add(
        vs.vy,
        at[2:-2, 2:-2, :, vs.taup1],
        - settings.dt_mom
        * (vs.z0[2:-2, 3:-1, vs.taup1, npx.newaxis] - vs.z0[2:-2, 2:-2, vs.taup1, npx.newaxis])
        / settings.dy
        * vs.maskCatch[2:-2, 2:-2, :],
    )

    return KernelOutput(vx=vs.vx, vy=vs.vy)
