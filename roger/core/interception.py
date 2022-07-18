from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at


@roger_kernel
def calc_rain_int_top(state):
    """
    Calculates interception at upper interception storage
    """
    vs = state.variables
    settings = state.settings

    # interception is constrained by remaining storage
    mask_rain = (vs.ta[:, :, vs.tau] > settings.ta_fm)
    vs.rain_top = update(
        vs.rain_top,
        at[2:-2, 2:-2], npx.where(mask_rain[2:-2, 2:-2], vs.prec[2:-2, 2:-2, vs.tau], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    # available interception storage
    int_top_free = allocate(state.dimensions, ("x", "y"))
    int_top_free = update(
        int_top_free,
        at[2:-2, 2:-2], npx.where(vs.S_int_top[2:-2, 2:-2, vs.tau] < vs.S_int_top_tot[2:-2, 2:-2], 0, vs.S_int_top_tot[2:-2, 2:-2] - vs.S_int_top[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    mask1 = (int_top_free >= vs.prec[:, :, vs.tau] * (1. - settings.throughfall_coeff)) & (vs.ta[:, :, vs.tau] > settings.ta_fm) & (int_top_free > 0)
    mask2 = (int_top_free < vs.prec[:, :, vs.tau] * (1. - settings.throughfall_coeff)) & (vs.ta[:, :, vs.tau] > settings.ta_fm) & (int_top_free > 0)

    vs.int_rain_top = update(
        vs.int_rain_top,
        at[2:-2, 2:-2], 0,
    )
    # rain is intercepted
    vs.int_rain_top = update_add(
        vs.int_rain_top,
        at[2:-2, 2:-2], vs.prec[2:-2, 2:-2, vs.tau] * (1. - settings.throughfall_coeff) * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # interception is constrained by remaining storage
    vs.int_rain_top = update(
        vs.int_rain_top,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], int_top_free[2:-2, 2:-2], vs.int_rain_top[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # Update top interception storage after rainfall
    vs.S_int_top = update_add(
        vs.S_int_top,
        at[2:-2, 2:-2, vs.tau], vs.int_rain_top[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(S_int_top=vs.S_int_top, rain_top=vs.rain_top, int_rain_top=vs.int_rain_top)


@roger_kernel
def calc_rain_int_ground(state):
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
        at[2:-2, 2:-2], (vs.prec[2:-2, 2:-2, vs.tau] - vs.int_rain_top[2:-2, 2:-2]) * mask_rain[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # available interception storage
    int_ground_free = allocate(state.dimensions, ("x", "y"))
    int_ground_free = update(
        int_ground_free,
        at[2:-2, 2:-2], npx.where((vs.S_int_ground[2:-2, 2:-2, vs.tau] < vs.S_int_ground_tot[2:-2, 2:-2]) & (vs.S_snow[2:-2, 2:-2, vs.tau] > 0), 0, vs.S_int_ground_tot[2:-2, 2:-2] - vs.S_int_ground[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    mask1 = (int_ground_free >= rain) & (vs.ta[:, :, vs.tau] > settings.ta_fm) & (int_ground_free > 0)
    mask2 = (int_ground_free < rain) & (vs.ta[:, :, vs.tau] > settings.ta_fm) & (int_ground_free > 0)

    vs.int_rain_ground = update(
        vs.int_rain_ground,
        at[2:-2, 2:-2], 0,
    )
    # rain is intercepted
    vs.int_rain_ground = update_add(
        vs.int_rain_ground,
        at[2:-2, 2:-2], rain[2:-2, 2:-2] * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # interception is constrained by remaining storage
    vs.int_rain_ground = update(
        vs.int_rain_ground,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], int_ground_free[2:-2, 2:-2], vs.int_rain_ground[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # Update top interception storage after rainfall
    vs.S_int_ground = update_add(
        vs.S_int_ground,
        at[2:-2, 2:-2, vs.tau], vs.int_rain_ground[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.rain_ground = update(
        vs.rain_ground,
        at[2:-2, 2:-2], (vs.rain_top[2:-2, 2:-2] - vs.int_rain_top[2:-2, 2:-2] - vs.int_rain_ground[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.z0 = update_add(
        vs.z0,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.S_snow[2:-2, 2:-2, vs.tau] > 0, 0, vs.rain_ground[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.prec_event_csum = update_add(
        vs.prec_event_csum,
        at[2:-2, 2:-2], npx.where(vs.S_snow[2:-2, 2:-2, vs.tau] > 0, 0, vs.rain_ground[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(S_int_ground=vs.S_int_ground, rain_ground=vs.rain_ground, int_rain_ground=vs.int_rain_ground, z0=vs.z0, prec_event_csum=vs.prec_event_csum)


@roger_kernel
def calc_snow_int_top(state):
    """
    Calculates snowfall interception at upper interception storage
    """
    vs = state.variables
    settings = state.settings

    # interception is constrained by remaining storage
    mask_snow = (vs.ta[:, :, vs.tau] <= settings.ta_fm)
    vs.snow_top = update(
        vs.snow_top,
        at[2:-2, 2:-2], npx.where(mask_snow[2:-2, 2:-2], vs.prec[2:-2, 2:-2, vs.tau], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    # available interception storage
    int_top_free = allocate(state.dimensions, ("x", "y"))
    int_top_free = update(
        int_top_free,
        at[2:-2, 2:-2], npx.where(vs.S_int_top[2:-2, 2:-2, vs.tau] >= vs.S_int_top_tot[2:-2, 2:-2], 0, vs.S_int_top_tot[2:-2, 2:-2] - vs.S_int_top[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    mask1 = (int_top_free >= vs.prec[:, :, vs.tau] * (1. - settings.throughfall_coeff)) & (vs.ta[:, :, vs.tau] <= settings.ta_fm) & (int_top_free > 0)
    mask2 = (int_top_free < vs.prec[:, :, vs.tau] * (1. - settings.throughfall_coeff)) & (vs.ta[:, :, vs.tau] <= settings.ta_fm) & (int_top_free > 0)

    # snow is intercepted
    vs.int_snow_top = update(
        vs.int_snow_top,
        at[2:-2, 2:-2], 0,
    )

    vs.int_snow_top = update_add(
        vs.int_snow_top,
        at[2:-2, 2:-2], vs.prec[2:-2, 2:-2, vs.tau] * (1. - settings.throughfall_coeff) * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # interception is constrained by remaining storage
    vs.int_snow_top = update(
        vs.int_snow_top,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], int_top_free[2:-2, 2:-2], vs.int_snow_top[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # Update top interception storage after snowfall
    vs.S_int_top = update_add(
        vs.S_int_top,
        at[2:-2, 2:-2, vs.tau], vs.int_snow_top[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.swe_top = update_add(
        vs.swe_top,
        at[2:-2, 2:-2, vs.tau], vs.int_snow_top[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(S_int_top=vs.S_int_top, swe_top=vs.swe_top, snow_top=vs.snow_top, int_snow_top=vs.int_snow_top)


@roger_kernel
def calc_snow_int_ground(state):
    """
    Calculates snowfall interception at lower interception storage
    """
    vs = state.variables
    settings = state.settings

    # snowfall after interception at upper surface layer
    snow = allocate(state.dimensions, ("x", "y"))
    mask_snow = (vs.ta[:, :, vs.tau] <= settings.ta_fm)
    snow = update(
        snow,
        at[2:-2, 2:-2], (vs.prec[2:-2, 2:-2, vs.tau] - vs.int_snow_top[2:-2, 2:-2]) * mask_snow[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # available interception storage
    int_ground_free = allocate(state.dimensions, ("x", "y"))
    int_ground_free = update(
        int_ground_free,
        at[2:-2, 2:-2], npx.where(vs.S_int_ground[2:-2, 2:-2, vs.tau] >= vs.S_int_ground_tot[2:-2, 2:-2], 0, vs.S_int_ground_tot[2:-2, 2:-2] - vs.S_int_ground[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    mask1 = (int_ground_free >= snow) & (vs.ta[:, :, vs.tau] <= settings.ta_fm) & (int_ground_free > 0)
    mask2 = (int_ground_free < snow) & (vs.ta[:, :, vs.tau] <= settings.ta_fm) & (int_ground_free > 0)

    # snow is intercepted
    vs.int_snow_ground = update(
        vs.int_snow_ground,
        at[2:-2, 2:-2], 0,
    )

    vs.int_snow_ground = update_add(
        vs.int_snow_ground,
        at[2:-2, 2:-2], snow[2:-2, 2:-2] * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # interception is constrained by remaining storage
    vs.int_snow_ground = update(
        vs.int_snow_ground,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], int_ground_free[2:-2, 2:-2], vs.int_snow_ground[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # Update top interception storage after snowfall
    vs.S_int_ground = update_add(
        vs.S_int_ground,
        at[2:-2, 2:-2, vs.tau], vs.int_snow_ground[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.swe_ground = update_add(
        vs.swe_ground,
        at[2:-2, 2:-2, vs.tau], vs.int_snow_ground[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.snow_ground = update(
        vs.snow_ground,
        at[2:-2, 2:-2], (vs.snow_top[2:-2, 2:-2] - vs.int_snow_top[2:-2, 2:-2] - vs.int_snow_ground[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.prec_event_csum = update_add(
        vs.prec_event_csum,
        at[2:-2, 2:-2], vs.snow_ground[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(S_int_ground=vs.S_int_ground, swe_ground=vs.swe_ground, snow_ground=vs.snow_ground, int_snow_ground=vs.int_snow_ground, prec_event_csum=vs.prec_event_csum)


@roger_kernel
def calc_int(state):
    """
    Calculates interception
    """
    vs = state.variables

    vs.int_prec = update(
        vs.int_prec,
        at[2:-2, 2:-2], (vs.int_rain_top[2:-2, 2:-2] + vs.int_rain_ground[2:-2, 2:-2] + vs.int_snow_top[2:-2, 2:-2] + vs.int_snow_ground[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(int_prec=vs.int_prec)


@roger_routine
def calculate_interception(state):
    """
    Calculates interception
    """
    vs = state.variables
    vs.update(calc_rain_int_top(state))
    vs.update(calc_rain_int_ground(state))
    vs.update(calc_snow_int_top(state))
    vs.update(calc_snow_int_ground(state))
    vs.update(calc_int(state))
