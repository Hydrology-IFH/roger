from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at


@roger_kernel
def calc_snow_int_top(state):
    """
    Calculates snowfall interception at upper interception storage
    """
    vs = state.variables
    settings = state.settings

    # interception is constrained by remaining storage
    mask_snow = (vs.ta[2:-2, 2:-2, vs.tau] <= settings.ta_fm)
    vs.snow_top = update(
        vs.snow_top,
        at[2:-2, 2:-2], npx.where(mask_snow, vs.prec, 0) * vs.maskCatch,
    )

    # available interception storage
    int_top_free = allocate(state.dimensions, ("x", "y"))
    int_top_free = update(
        int_top_free,
        at[2:-2, 2:-2], npx.where(vs.S_int_top[2:-2, 2:-2, vs.tau] >= vs.S_int_top_tot, 0, vs.S_int_top_tot - vs.S_int_top[2:-2, 2:-2, vs.tau]) * vs.maskCatch,
    )

    mask1 = (int_top_free >= vs.prec * (1. - settings.throughfall_coeff)) & (vs.ta[2:-2, 2:-2, vs.tau] <= settings.ta_fm) & (int_top_free > 0)
    mask2 = (int_top_free < vs.prec * (1. - settings.throughfall_coeff)) & (vs.ta[2:-2, 2:-2, vs.tau] <= settings.ta_fm) & (int_top_free > 0)

    # snow is intercepted
    int_snow_top = allocate(state.dimensions, ("x", "y"))
    vs.int_snow_top = update(
        vs.int_snow_top,
        at[2:-2, 2:-2], int_snow_top,
    )

    vs.int_snow_top = update_add(
        vs.int_snow_top,
        at[2:-2, 2:-2], vs.prec * (1. - settings.throughfall_coeff) * mask1 * vs.maskCatch,
    )

    # interception is constrained by remaining storage
    vs.int_snow_top = update(
        vs.int_snow_top,
        at[2:-2, 2:-2], npx.where(mask2, int_top_free, vs.int_snow_top) * vs.maskCatch,
    )

    # Update top interception storage after snowfall
    vs.S_int_top = update_add(
        vs.S_int_top,
        at[2:-2, 2:-2, vs.tau], vs.int_snow_top * vs.maskCatch,
    )

    vs.swe_top = update_add(
        vs.swe_top,
        at[2:-2, 2:-2, vs.tau], vs.int_snow_top * vs.maskCatch,
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
    mask_snow = (vs.ta[2:-2, 2:-2, vs.tau] <= settings.ta_fm)
    snow = update(
        snow,
        at[2:-2, 2:-2], (vs.prec - vs.int_snow_top) * mask_snow * vs.maskCatch,
    )

    # available interception storage
    int_ground_free = allocate(state.dimensions, ("x", "y"))
    int_ground_free = update(
        int_ground_free,
        at[2:-2, 2:-2], npx.where(vs.S_int_ground[2:-2, 2:-2, vs.tau] >= vs.S_int_ground_tot, 0, vs.S_int_ground_tot - vs.S_int_ground[2:-2, 2:-2, vs.tau]) * vs.maskCatch * vs.maskCatch,
    )

    mask1 = (int_ground_free >= snow) & (vs.ta[2:-2, 2:-2, vs.tau] <= settings.ta_fm) & (int_ground_free > 0)
    mask2 = (int_ground_free < snow) & (vs.ta[2:-2, 2:-2, vs.tau] <= settings.ta_fm) & (int_ground_free > 0)

    # snow is intercepted
    int_snow_ground = allocate(state.dimensions, ("x", "y"))
    vs.int_snow_ground = update(
        vs.int_snow_ground,
        at[2:-2, 2:-2], int_snow_ground,
    )

    vs.int_snow_ground = update_add(
        vs.int_snow_ground,
        at[2:-2, 2:-2], snow * mask1 * vs.maskCatch,
    )

    # interception is constrained by remaining storage
    vs.int_snow_ground = update(
        vs.int_snow_ground,
        at[2:-2, 2:-2], npx.where(mask2, int_ground_free, vs.int_snow_ground) * vs.maskCatch,
    )

    # Update top interception storage after snowfall
    vs.S_int_ground = update_add(
        vs.S_int_ground,
        at[2:-2, 2:-2, vs.tau], vs.int_snow_ground * vs.maskCatch,
    )

    vs.swe_ground = update_add(
        vs.swe_ground,
        at[2:-2, 2:-2, vs.tau], vs.int_snow_ground * vs.maskCatch,
    )

    vs.snow_ground = update(
        vs.snow_ground,
        at[2:-2, 2:-2], (vs.snow_top - vs.int_snow_top - vs.int_snow_ground) * vs.maskCatch,
    )

    vs.prec_event_sum = update_add(
        vs.prec_event_sum,
        at[2:-2, 2:-2, vs.tau], vs.snow_ground * vs.maskCatch,
    )

    return KernelOutput(S_int_ground=vs.S_int_ground, swe_ground=vs.swe_ground, snow_ground=vs.snow_ground, int_snow_ground=vs.int_snow_ground, prec_event_sum=vs.prec_event_sum)


@roger_kernel
def calc_snow_accumulation(state):
    """
    Calculates snow cover
    """
    vs = state.variables
    settings = state.settings

    mask1 = (vs.ta[2:-2, 2:-2, vs.tau] <= settings.ta_fm)

    vs.S_snow = update_add(
        vs.S_snow ,
        at[2:-2, 2:-2, vs.tau], vs.snow_ground * mask1 * vs.maskCatch,
    )

    vs.swe = update_add(
        vs.swe ,
        at[2:-2, 2:-2, vs.tau], vs.snow_ground * mask1 * vs.maskCatch,
    )

    return KernelOutput(S_snow=vs.S_snow, swe=vs.swe)


@roger_kernel
def calc_rain_on_snow(state):
    """
    Calculates snow on snow
    """
    vs = state.variables
    settings = state.settings

    mask1 = ((vs.swe[2:-2, 2:-2, vs.tau] > 0) | (vs.S_snow[2:-2, 2:-2, vs.tau] > 0)) & (vs.ta[2:-2, 2:-2, vs.tau] > settings.ta_fm)

    vs.S_snow = update_add(
        vs.S_snow ,
        at[2:-2, 2:-2, vs.tau], vs.rain_ground * mask1 * vs.maskCatch,
    )

    return KernelOutput(S_snow=vs.S_snow)


@roger_kernel
def calc_snow_melt_top_int(state):
    """
    Calculates snow melt in top interception storage
    """
    vs = state.variables
    settings = state.settings

    # hourly potential snow melt
    snow_melt_pot = allocate(state.dimensions, ("x", "y"))
    snow_melt_pot = update(
        snow_melt_pot,
        at[2:-2, 2:-2], (settings.sf * (vs.ta[2:-2, 2:-2, vs.tau] - settings.ta_fm) * vs.dt) * vs.maskCatch,
    )

    mask1 = (snow_melt_pot > 0) & (snow_melt_pot <= vs.swe_top[2:-2, 2:-2, vs.tau]) & (vs.swe_top[2:-2, 2:-2, vs.tau] == vs.S_int_top[2:-2, 2:-2, vs.tau])
    mask2 = (snow_melt_pot > 0) & (snow_melt_pot > vs.swe_top[2:-2, 2:-2, vs.tau]) & (vs.swe_top[2:-2, 2:-2, vs.tau] == vs.S_int_top[2:-2, 2:-2, vs.tau])

    snow_melt_top = allocate(state.dimensions, ("x", "y"))
    vs.snow_melt_top = update(
        vs.snow_melt_top,
        at[2:-2, 2:-2], snow_melt_top,
    )

    vs.snow_melt_top = update(
        vs.snow_melt_top,
        at[2:-2, 2:-2], npx.where(mask1, snow_melt_pot, vs.snow_melt_top) * vs.maskCatch,
    )
    vs.snow_melt_top = update(
        vs.snow_melt_top,
        at[2:-2, 2:-2], npx.where(mask2, vs.swe_top[2:-2, 2:-2, vs.tau], vs.snow_melt_top) * vs.maskCatch,
    )

    # snow melt when snow cover contains liquid water
    wtmx = allocate(state.dimensions, ("x", "y"))
    wtmx = (10000. / (100 - settings.rmax) / 100.) * vs.swe_top[2:-2, 2:-2, vs.tau]
    mask3 = (vs.swe_top[2:-2, 2:-2, vs.tau] < vs.S_int_top[2:-2, 2:-2, vs.tau]) & (vs.swe_top[2:-2, 2:-2, vs.tau] > 0) & (vs.S_int_top[2:-2, 2:-2, vs.tau] > wtmx)
    vs.snow_melt_top = update_add(
        vs.snow_melt_top,
        at[2:-2, 2:-2], (vs.S_int_top[2:-2, 2:-2, vs.tau] - wtmx) * mask3 * vs.maskCatch,
    )

    # energy consumption of snow melt, will be subtracted from
    # evapotranspiration snow cover is partly melted, some snow cover
    # remains
    mask4 = (vs.snow_melt_top > 0) & (vs.snow_melt_top <= vs.swe_top[2:-2, 2:-2, vs.tau])
    # snow cover is fully melted, no snow cover left
    mask5 = (vs.snow_melt_top > 0) & (vs.snow_melt_top > vs.swe_top[2:-2, 2:-2, vs.tau])

    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.snow_melt_top * mask4 * vs.maskCatch,
    )
    vs.swe_top = update_add(
        vs.swe_top,
        at[2:-2, 2:-2, vs.tau], -vs.snow_melt_top * mask4 * vs.maskCatch,
    )

    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.swe_top[2:-2, 2:-2, vs.tau] * mask5 * vs.maskCatch,
    )
    vs.swe_top = update_add(
        vs.swe_top,
        at[2:-2, 2:-2, vs.tau], npx.where(mask5, 0, -vs.swe_top[2:-2, 2:-2, vs.tau]) * vs.maskCatch,
    )

    return KernelOutput(snow_melt_top=vs.snow_melt_top, pet_res=vs.pet_res, swe_top=vs.swe_top)


@roger_kernel
def calc_snow_melt_ground_int(state):
    """
    Calculates snow melt in ground interception storage
    """
    vs = state.variables
    settings = state.settings

    # hourly potential snow melt
    snow_melt_pot = allocate(state.dimensions, ("x", "y"))
    snow_melt_pot = update(
        snow_melt_pot,
        at[2:-2, 2:-2], (settings.sf * (vs.ta[2:-2, 2:-2, vs.tau] - settings.ta_fm) * vs.dt) * vs.maskCatch,
    )

    mask1 = (snow_melt_pot > 0) & (snow_melt_pot <= vs.swe_ground[2:-2, 2:-2, vs.tau]) & (vs.swe_ground[2:-2, 2:-2, vs.tau] == vs.S_int_ground[2:-2, 2:-2, vs.tau])
    mask2 = (snow_melt_pot > 0) & (snow_melt_pot > vs.swe_ground[2:-2, 2:-2, vs.tau]) & (vs.swe_ground[2:-2, 2:-2, vs.tau] == vs.S_int_ground[2:-2, 2:-2, vs.tau])

    snow_melt_ground = allocate(state.dimensions, ("x", "y"))
    vs.snow_melt_ground = update(
        vs.snow_melt_ground,
        at[2:-2, 2:-2], snow_melt_ground,
    )

    vs.snow_melt_ground = update(
        vs.snow_melt_ground,
        at[2:-2, 2:-2], npx.where(mask1, snow_melt_pot, vs.snow_melt_ground) * vs.maskCatch,
    )
    vs.snow_melt_ground = update(
        vs.snow_melt_ground,
        at[2:-2, 2:-2], npx.where(mask2, vs.swe_ground[2:-2, 2:-2, vs.tau], vs.snow_melt_ground) * vs.maskCatch,
    )

    # snow melt when snow cover contains liquid water
    wtmx = allocate(state.dimensions, ("x", "y"))
    wtmx = (10000. / (100 - settings.rmax) / 100.) * vs.swe_ground[2:-2, 2:-2, vs.tau]
    mask3 = (vs.swe_ground[2:-2, 2:-2, vs.tau] < vs.S_int_ground[2:-2, 2:-2, vs.tau]) & (vs.swe_ground[2:-2, 2:-2, vs.tau] > 0) & (vs.S_int_ground[2:-2, 2:-2, vs.tau] > wtmx)
    vs.snow_melt_ground = update_add(
        vs.snow_melt_ground,
        at[2:-2, 2:-2], (vs.S_int_ground[2:-2, 2:-2, vs.tau] - wtmx) * mask3 * vs.maskCatch,
    )

    # energy consumption of snow melt, will be subtracted from
    # evapotranspiration snow cover is partly melted, some snow cover
    # remains
    mask4 = (vs.snow_melt_ground > 0) & (vs.snow_melt_ground <= vs.swe_ground[2:-2, 2:-2, vs.tau])
    # snow cover is fully melted, no snow cover left
    mask5 = (vs.snow_melt_ground > 0) & (vs.snow_melt_ground > vs.swe_ground[2:-2, 2:-2, vs.tau])

    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.snow_melt_ground * mask4 * vs.maskCatch,
    )
    vs.swe_ground = update_add(
        vs.swe_ground,
        at[2:-2, 2:-2, vs.tau], -vs.snow_melt_ground * mask4 * vs.maskCatch,
    )

    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.swe_ground[2:-2, 2:-2, vs.tau] * mask5 * vs.maskCatch,
    )
    vs.swe_ground = update_add(
        vs.swe_ground,
        at[2:-2, 2:-2, vs.tau], npx.where(mask5, 0, -vs.swe_ground[2:-2, 2:-2, vs.tau]) * vs.maskCatch,
    )

    return KernelOutput(snow_melt_ground=vs.snow_melt_ground, pet_res=vs.pet_res, swe_ground=vs.swe_ground)


@roger_kernel
def calc_snow_melt(state):
    """
    Calculates snow melt of snow cover
    """
    vs = state.variables
    settings = state.settings

    # hourly potential snow melt
    snow_melt_pot = allocate(state.dimensions, ("x", "y"))
    snow_melt_pot = update(
        snow_melt_pot,
        at[2:-2, 2:-2], (settings.sf * (vs.ta[2:-2, 2:-2, vs.tau] - settings.ta_fm) * vs.dt) * vs.maskCatch,
    )
    mask1 = (snow_melt_pot > 0) & (snow_melt_pot <= vs.swe[2:-2, 2:-2, vs.tau]) & (vs.swe[2:-2, 2:-2, vs.tau] > 0)
    mask2 = (snow_melt_pot > 0) & (snow_melt_pot > vs.swe[2:-2, 2:-2, vs.tau]) & (vs.swe[2:-2, 2:-2, vs.tau] > 0)

    snow_melt = allocate(state.dimensions, ("x", "y"))
    vs.snow_melt = update(
        vs.snow_melt,
        at[2:-2, 2:-2], snow_melt,
    )
    vs.snow_melt = update(
        vs.snow_melt,
        at[2:-2, 2:-2], npx.where(mask1, snow_melt_pot, vs.snow_melt) * vs.maskCatch,
    )
    vs.snow_melt = update(
        vs.snow_melt,
        at[2:-2, 2:-2], npx.where(mask2, vs.swe[2:-2, 2:-2, vs.tau], vs.snow_melt) * vs.maskCatch,
    )

    # snow melt when snow cover contains liquid water
    # retention storage
    wtmx = allocate(state.dimensions, ("x", "y"))
    wtmx = (10000. / (100 - settings.rmax) / 100.) * vs.swe[2:-2, 2:-2, vs.tau]
    # retained water
    q_ret = allocate(state.dimensions, ("x", "y"))
    q_ret = update(
        q_ret,
        at[2:-2, 2:-2], (vs.S_snow[2:-2, 2:-2, vs.tau] - vs.swe[2:-2, 2:-2, vs.tau]) * vs.maskCatch,
    )
    mask3 = (vs.swe[2:-2, 2:-2, vs.tau] < vs.S_snow[2:-2, 2:-2, vs.tau]) & (vs.swe[2:-2, 2:-2, vs.tau] > 0) & (q_ret > wtmx)
    vs.q_rain_on_snow = update(
        vs.q_rain_on_snow,
        at[2:-2, 2:-2], 0,
    )
    vs.q_rain_on_snow = update(
        vs.q_rain_on_snow,
        at[2:-2, 2:-2], npx.where(mask3, q_ret - wtmx, vs.q_rain_on_snow) * vs.maskCatch,
    )
    vs.q_rain_on_snow = update(
        vs.q_rain_on_snow,
        at[2:-2, 2:-2], npx.where(vs.q_rain_on_snow < 0, 0, vs.q_rain_on_snow) * vs.maskCatch,
    )

    mask4 = (vs.S_snow[2:-2, 2:-2, vs.tau] > 0) & (vs.swe[2:-2, 2:-2, vs.tau] <= 0) & (vs.snow_melt <= 0)
    vs.q_rain_on_snow = update(
        vs.q_rain_on_snow,
        at[2:-2, 2:-2], npx.where(mask4, vs.S_snow[2:-2, 2:-2, vs.tau], vs.q_rain_on_snow) * vs.maskCatch,
    )
    mask44 = (vs.ta[2:-2, 2:-2, vs.tau] < 0)
    vs.q_rain_on_snow = update(
        vs.q_rain_on_snow,
        at[2:-2, 2:-2], npx.where(mask44, 0, vs.q_rain_on_snow) * vs.maskCatch,
    )

    # energy consumption of snow melt, will be subtracted from
    # evapotranspiration snow cover is partly melted, some snow cover
    # remains
    mask5 = (vs.snow_melt > 0) & (vs.q_rain_on_snow <= 0) & (vs.snow_melt <= vs.swe[2:-2, 2:-2, vs.tau]) & (vs.snow_melt <= vs.S_snow[2:-2, 2:-2, vs.tau])
    # snow melt, snow cover is fully melted
    mask6 = (vs.snow_melt > 0) & (vs.q_rain_on_snow <= 0) & (vs.snow_melt > vs.swe[2:-2, 2:-2, vs.tau]) & (vs.snow_melt <= vs.S_snow[2:-2, 2:-2, vs.tau])
    # rain-on-snow
    mask7 = (vs.snow_melt > 0) & (vs.q_rain_on_snow > 0) & (vs.snow_melt <= vs.swe[2:-2, 2:-2, vs.tau]) & (vs.q_rain_on_snow + vs.snow_melt <= vs.S_snow[2:-2, 2:-2, vs.tau])
    # snow cover, snow cover is fully melted
    mask8 = (vs.snow_melt > 0) & (vs.q_rain_on_snow > 0) & (vs.snow_melt > vs.swe[2:-2, 2:-2, vs.tau]) & (vs.q_rain_on_snow + vs.snow_melt <= vs.S_snow[2:-2, 2:-2, vs.tau])
    # residual snow melt
    mask9 = (vs.S_snow[2:-2, 2:-2, vs.tau] > 0) & (vs.swe[2:-2, 2:-2, vs.tau] <= 0)

    # snow melt
    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.snow_melt * mask5 * vs.maskCatch,
    )
    vs.swe = update_add(
        vs.swe,
        at[2:-2, 2:-2, vs.tau], -vs.snow_melt * mask5 * vs.maskCatch,
    )
    vs.S_snow = update_add(
        vs.S_snow,
        at[2:-2, 2:-2, vs.tau], -vs.snow_melt * mask5 * vs.maskCatch,
    )

    # snow cover is fully melted
    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.swe[2:-2, 2:-2, vs.tau] * mask6 * vs.maskCatch,
    )
    vs.swe = update(
        vs.swe,
        at[2:-2, 2:-2, vs.tau], npx.where(mask6, 0, vs.swe[2:-2, 2:-2, vs.tau]) * vs.maskCatch,
    )
    vs.S_snow = update_add(
        vs.S_snow,
        at[2:-2, 2:-2, vs.tau], -vs.snow_melt * mask6 * vs.maskCatch,
    )

    # rain-on-snow
    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.snow_melt * mask7 * vs.maskCatch,
    )
    vs.swe = update_add(
        vs.swe,
        at[2:-2, 2:-2, vs.tau], -vs.snow_melt * mask7 * vs.maskCatch,
    )
    vs.S_snow = update_add(
        vs.S_snow,
        at[2:-2, 2:-2, vs.tau], -(vs.snow_melt + vs.q_rain_on_snow) * mask7 * vs.maskCatch,
    )

    # rain-on-snow, snow cover is fully melted
    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.snow_melt * mask8 * vs.maskCatch,
    )
    vs.swe = update(
        vs.swe,
        at[2:-2, 2:-2, vs.tau], npx.where(mask8, 0, vs.swe[2:-2, 2:-2, vs.tau]) * vs.maskCatch,
    )
    vs.S_snow = update_add(
        vs.S_snow,
        at[2:-2, 2:-2, vs.tau], -(vs.snow_melt + vs.q_rain_on_snow) * mask8 * vs.maskCatch,
    )

    # residual snow melt
    vs.swe = update(
        vs.swe,
        at[2:-2, 2:-2, vs.tau], npx.where(mask9, 0, vs.swe[2:-2, 2:-2, vs.tau]) * vs.maskCatch,
    )
    vs.S_snow = update(
        vs.S_snow,
        at[2:-2, 2:-2, vs.tau], npx.where(mask9, 0, vs.S_snow[2:-2, 2:-2, vs.tau]) * vs.maskCatch,
    )

    vs.q_snow = update(
        vs.q_snow,
        at[2:-2, 2:-2], (vs.snow_melt + vs.q_rain_on_snow) * vs.maskCatch,
    )

    vs.z0 = update_add(
        vs.z0,
        at[2:-2, 2:-2, vs.tau], vs.q_snow * vs.maskCatch,
    )

    vs.prec_event_sum = update_add(
        vs.prec_event_sum,
        at[2:-2, 2:-2, vs.tau], vs.q_snow * vs.maskCatch,
    )

    return KernelOutput(snow_melt=vs.snow_melt, q_rain_on_snow=vs.q_rain_on_snow, q_snow=vs.q_snow, z0=vs.z0, S_snow=vs.S_snow, swe=vs.swe, pet_res=vs.pet_res, prec_event_sum=vs.prec_event_sum)


@roger_routine
def calculate_snow(state):
    """
    Calculates snow accumlutaion, snow melt and rain-on-snow
    """
    vs = state.variables

    vs.update(calc_snow_int_top(state))
    vs.update(calc_snow_int_ground(state))
    vs.update(calc_snow_accumulation(state))
    vs.update(calc_rain_on_snow(state))
    vs.update(calc_snow_melt_top_int(state))
    vs.update(calc_snow_melt_ground_int(state))
    vs.update(calc_snow_melt(state))
