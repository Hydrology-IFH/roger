from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at


@roger_kernel
def calc_channel_routing(state):
    """
    Calculates channel routing
    """
    pass


@roger_kernel
def calc_surface_runoff_routing_1D(state):
    """
    Calculates surface runoff routing
    """
    vs = state.variables
    settings = state.settings

    vs.z0 = update_add(
        vs.z0,
        at[2:-2, 2:-2, vs.tau],
        vs.q_sof[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    area = allocate(state.dimensions, ("x", "y"))
    perimeter = allocate(state.dimensions, ("x", "y"))
    hydraulic_radius = allocate(state.dimensions, ("x", "y"))

    area = update(
        area,
        at[2:-2, 2:-2],
        (vs.z0[2:-2, 2:-2, vs.tau] / 1000) * 0.5 * (2 * settings.dx) * vs.maskCatch[2:-2, 2:-2],
    )
    perimeter = update(
        perimeter,
        at[2:-2, 2:-2],
        2 * (vs.z0[2:-2, 2:-2, vs.tau] / 1000) + settings.dx * vs.maskCatch[2:-2, 2:-2],
    )
    hydraulic_radius = update(
        hydraulic_radius,
        at[2:-2, 2:-2],
        area[2:-2, 2:-2] / perimeter[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # surface runoff (convert m3/s to mm/dt)
    vs.q_sur = update(
        vs.q_sur,
        at[2:-2, 2:-2],
        vs.k_st[2:-2, 2:-2]
        * (vs.slope[2:-2, 2:-2] ** 0.5)
        * (hydraulic_radius[2:-2, 2:-2] ** (2 / 3))
        * area[2:-2, 2:-2]
        * (vs.dt_secs / (settings.dx * settings.dy * 1000))
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sur = update(
        vs.q_sur,
        at[2:-2, 2:-2],
        npx.where(vs.q_sur[2:-2, 2:-2] > vs.z0[2:-2, 2:-2, vs.tau], vs.z0[2:-2, 2:-2, vs.tau], vs.q_sur[2:-2, 2:-2])
        * vs.maskCatch[2:-2, 2:-2],
    )

    mask_north = vs.flow_dir_topo == 64
    mask_northeast = vs.flow_dir_topo == 128
    mask_east = vs.flow_dir_topo == 1
    mask_southeast = vs.flow_dir_topo == 2
    mask_south = vs.flow_dir_topo == 4
    mask_southwest = vs.flow_dir_topo == 8
    mask_west = vs.flow_dir_topo == 16
    mask_northwest = vs.flow_dir_topo == 32

    # calculate lateral surface runoff
    vs.q_sur_out_d8 = update(
        vs.q_sur_out_d8,
        at[2:-2, 2:-2, :],
        0,
    )

    vs.q_sur_out_d8 = update(
        vs.q_sur_out_d8,
        at[2:-2, 2:-2, 0],
        npx.where(mask_north[2:-2, 2:-2], vs.q_sur[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sur_out_d8 = update(
        vs.q_sur_out_d8,
        at[2:-2, 2:-2, 1],
        npx.where(mask_northeast[2:-2, 2:-2], vs.q_sur[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sur_out_d8 = update(
        vs.q_sur_out_d8,
        at[2:-2, 2:-2, 2],
        npx.where(mask_east[2:-2, 2:-2], vs.q_sur[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sur_out_d8 = update(
        vs.q_sur_out_d8,
        at[2:-2, 2:-2, 3],
        npx.where(mask_southeast[2:-2, 2:-2], vs.q_sur[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sur_out_d8 = update(
        vs.q_sur_out_d8,
        at[2:-2, 2:-2, 4],
        npx.where(mask_south[2:-2, 2:-2], vs.q_sur[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sur_out_d8 = update(
        vs.q_sur_out_d8,
        at[2:-2, 2:-2, 5],
        npx.where(mask_southwest[2:-2, 2:-2], vs.q_sur[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sur_out_d8 = update(
        vs.q_sur_out_d8,
        at[2:-2, 2:-2, 6],
        npx.where(mask_west[2:-2, 2:-2], vs.q_sur[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sur_out_d8 = update(
        vs.q_sur_out_d8,
        at[2:-2, 2:-2, 7],
        npx.where(mask_northwest[2:-2, 2:-2], vs.q_sur[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sur_out = update(
        vs.q_sur_out,
        at[2:-2, 2:-2],
        npx.sum(vs.q_sur_out_d8[2:-2, 2:-2, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
    )

    # calculate lateral subsurface inflow
    vs.q_sur_in_d8 = update(
        vs.q_sur_in_d8,
        at[2:-2, 2:-2, :],
        0,
    )

    vs.q_sur_in_d8 = update(
        vs.q_sur_in_d8,
        at[2:-2, 1:-3, 0],
        npx.where(mask_north[2:-2, 2:-2], vs.q_sur_out_d8[2:-2, 2:-2, 0], vs.q_sur_in_d8[2:-2, 1:-3, 0])
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sur_in_d8 = update(
        vs.q_sur_in_d8,
        at[1:-3, 1:-3, 1],
        npx.where(mask_northeast[2:-2, 2:-2], vs.q_sur_out_d8[2:-2, 2:-2, 1], vs.q_sur_in_d8[1:-3, 1:-3, 1])
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sur_in_d8 = update(
        vs.q_sur_in_d8,
        at[3:-1, 2:-2, 2],
        npx.where(mask_east[2:-2, 2:-2], vs.q_sur_out_d8[2:-2, 2:-2, 2], vs.q_sur_in_d8[3:-1, 2:-2, 2])
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sur_in_d8 = update(
        vs.q_sur_in_d8,
        at[3:-1, 3:-1, 3],
        npx.where(mask_southeast[2:-2, 2:-2], vs.q_sur_out_d8[2:-2, 2:-2, 3], vs.q_sur_in_d8[3:-1, 3:-1, 3])
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sur_in_d8 = update(
        vs.q_sur_in_d8,
        at[2:-2, 3:-1, 4],
        npx.where(mask_south[2:-2, 2:-2], vs.q_sur_out_d8[2:-2, 2:-2, 4], vs.q_sur_in_d8[2:-2, 3:-1, 4])
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sur_in_d8 = update(
        vs.q_sur_in_d8,
        at[1:-3, 3:-1, 5],
        npx.where(mask_southwest[2:-2, 2:-2], vs.q_sur_out_d8[2:-2, 2:-2, 5], vs.q_sur_in_d8[1:-3, 3:-1, 5])
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sur_in_d8 = update(
        vs.q_sur_in_d8,
        at[1:-3, 2:-2, 6],
        npx.where(mask_west[2:-2, 2:-2], vs.q_sur_out_d8[2:-2, 2:-2, 6], vs.q_sur_in_d8[1:-3, 2:-2, 6])
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sur_in_d8 = update(
        vs.q_sur_in_d8,
        at[1:-3, 1:-3, 7],
        npx.where(mask_northwest[2:-2, 2:-2], vs.q_sur_out_d8[2:-2, 2:-2, 7], vs.q_sur_in_d8[1:-3, 1:-3, 7])
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sur_in = update(
        vs.q_sur_in,
        at[2:-2, 2:-2],
        npx.sum(vs.q_sur_in_d8[2:-2, 2:-2, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
    )

    # outflow boundaries
    vs.q_sur_in = update(
        vs.q_sur_in,
        at[2:-2, 2:-2],
        npx.where((vs.outer_boundary[2:-2, 2:-2] == 1), 0, vs.q_sur_in[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # update surface water level
    vs.z0 = update_add(
        vs.z0,
        at[2:-2, 2:-2, vs.tau],
        -vs.q_sur_out[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z0 = update_add(
        vs.z0,
        at[2:-2, 2:-2, vs.tau],
        vs.q_sur_in[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(
        q_sur_out_d8=vs.q_sur_out_d8, q_sur_in_d8=vs.q_sur_in_d8, q_sur_out=vs.q_sur_out, q_sur_in=vs.q_sur_in, z0=vs.z0
    )


@roger_kernel
def calc_surface_runoff_routing_2D(state):
    """
    Calculates surface runoff routing
    """
    pass


@roger_routine
def calculate_surface_runoff(state):
    """
    Calculates surface runoff
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_routing_1D:
        vs.update(calc_surface_runoff_routing_1D(state))
