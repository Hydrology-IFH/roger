from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at


@roger_kernel
def calc_int_top(state):
    """
    Calculates interception at upper interception storage
    """
    vs = state.variables
    settings = state.settings

    # interception is constrained by remaining storage
    mask_rain = (vs.ta[:, :, vs.tau] > settings.ta_fm)
    vs.rain_top = update(
        vs.rain_top,
        at[:, :], npx.where(mask_rain, vs.prec, 0) * vs.maskCatch,
    )

    # available interception storage
    int_top_free = allocate(state.dimensions, ("x", "y"))
    int_top_free = update(
        int_top_free,
        at[:, :], npx.where(vs.S_int_top[:, :, vs.tau] < vs.S_int_top_tot, 0, vs.S_int_top_tot - vs.S_int_top[:, :, vs.tau]) * vs.maskCatch,
    )

    mask1 = (int_top_free >= vs.prec * (1. - settings.throughfall_coeff)) & (vs.ta[:, :, vs.tau] > settings.ta_fm) & (int_top_free > 0)
    mask2 = (int_top_free < vs.prec * (1. - settings.throughfall_coeff)) & (vs.ta[:, :, vs.tau] > settings.ta_fm) & (int_top_free > 0)

    int_rain_top = allocate(state.dimensions, ("x", "y"))
    vs.int_rain_top = update(
        vs.int_rain_top,
        at[:, :], int_rain_top,
    )
    # rain is intercepted
    vs.int_rain_top = update_add(
        vs.int_rain_top,
        at[:, :], vs.prec * (1. - settings.throughfall_coeff) * mask1 * vs.maskCatch,
    )

    # interception is constrained by remaining storage
    vs.int_rain_top = update(
        vs.int_rain_top,
        at[:, :], npx.where(mask2, int_top_free, vs.int_rain_top) * vs.maskCatch,
    )

    # Update top interception storage after rainfall
    vs.S_int_top = update_add(
        vs.S_int_top,
        at[:, :, vs.tau], vs.int_rain_top * vs.maskCatch,
    )

    return KernelOutput(S_int_top=vs.S_int_top, rain_top=vs.rain_top, int_rain_top=vs.int_rain_top)


@roger_kernel
def calc_int_ground(state):
    """
    Calculates interception at lower interception storage
    """
    vs = state.variables
    settings = state.settings

    # rainfall after interception at upper surface layer
    rain = allocate(state.dimensions, ("x", "y"))
    mask_rain = (vs.ta[:, :, vs.tau] > settings.ta_fm)
    rain = update(
        rain,
        at[:, :], (vs.prec - vs.int_rain_top) * mask_rain * vs.maskCatch,
    )

    # available interception storage
    int_ground_free = allocate(state.dimensions, ("x", "y"))
    int_ground_free = update(
        int_ground_free,
        at[:, :], npx.where((vs.S_int_ground[:, :, vs.tau] < vs.S_int_ground_tot) & (vs.S_snow[:, :, vs.tau] > 0), 0, vs.S_int_ground_tot - vs.S_int_ground[:, :, vs.tau]) * vs.maskCatch * vs.maskCatch,
    )

    mask1 = (int_ground_free >= rain) & (vs.ta[:, :, vs.tau] > settings.ta_fm) & (int_ground_free > 0)
    mask2 = (int_ground_free < rain) & (vs.ta[:, :, vs.tau] > settings.ta_fm) & (int_ground_free > 0)

    int_rain_ground = allocate(state.dimensions, ("x", "y"))
    vs.int_rain_ground = update(
        vs.int_rain_ground,
        at[:, :], int_rain_ground,
    )
    # rain is intercepted
    vs.int_rain_ground = update_add(
        vs.int_rain_ground,
        at[:, :], rain * mask1 * vs.maskCatch,
    )

    # interception is constrained by remaining storage
    vs.int_rain_ground = update(
        vs.int_rain_ground,
        at[:, :], npx.where(mask2, int_ground_free, vs.int_rain_ground) * vs.maskCatch,
    )

    # Update top interception storage after rainfall
    vs.S_int_ground = update_add(
        vs.S_int_ground,
        at[:, :, vs.tau], vs.int_rain_ground * vs.maskCatch,
    )

    vs.rain_ground = update(
        vs.rain_ground,
        at[:, :], (vs.rain_top - vs.int_rain_top - vs.int_rain_ground) * vs.maskCatch,
    )

    vs.z0 = update_add(
        vs.z0,
        at[:, :, vs.tau], npx.where(vs.S_snow[:, :, vs.tau] > 0, 0, vs.rain_ground) * vs.maskCatch,
    )

    vs.prec_event_sum = update_add(
        vs.prec_event_sum,
        at[:, :, vs.tau], npx.where(vs.S_snow[:, :, vs.tau] > 0, 0, vs.rain_ground) * vs.maskCatch,
    )

    return KernelOutput(S_int_ground=vs.S_int_ground, rain_ground=vs.rain_ground, int_rain_ground=vs.int_rain_ground, z0=vs.z0, prec_event_sum=vs.prec_event_sum)


@roger_routine
def calculate_interception(state):
    """
    Calculates interception
    """
    vs = state.variables
    vs.update(calc_int_top(state))
    vs.update(calc_int_ground(state))
