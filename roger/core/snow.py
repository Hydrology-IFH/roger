from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at


@roger_kernel
def calc_snow_accumulation(state):
    """
    Calculates snow cover
    """
    vs = state.variables
    settings = state.settings

    mask1 = (vs.ta[:, :, vs.tau] <= settings.ta_fm)

    vs.S_snow = update_add(
        vs.S_snow ,
        at[2:-2, 2:-2, vs.tau], vs.snow_ground[2:-2, 2:-2] * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.swe = update_add(
        vs.swe ,
        at[2:-2, 2:-2, vs.tau], vs.snow_ground[2:-2, 2:-2] * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(S_snow=vs.S_snow, swe=vs.swe)


@roger_kernel
def calc_rain_on_snow(state):
    """
    Calculates snow on snow
    """
    vs = state.variables
    settings = state.settings

    mask1 = ((vs.swe[:, :, vs.tau] > 0) | (vs.S_snow[:, :, vs.tau] > 0)) & (vs.ta[:, :, vs.tau] > settings.ta_fm)

    vs.S_snow = update_add(
        vs.S_snow ,
        at[2:-2, 2:-2, vs.tau], vs.rain_ground[2:-2, 2:-2] * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
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
        at[2:-2, 2:-2], (settings.sf * (vs.ta[2:-2, 2:-2, vs.tau] - settings.ta_fm) * vs.dt) * vs.maskCatch[2:-2, 2:-2],
    )

    mask1 = (snow_melt_pot > 0) & (snow_melt_pot <= vs.swe_top[:, :, vs.tau]) & (vs.swe_top[:, :, vs.tau] == vs.S_int_top[:, :, vs.tau])
    mask2 = (snow_melt_pot > 0) & (snow_melt_pot > vs.swe_top[:, :, vs.tau]) & (vs.swe_top[:, :, vs.tau] == vs.S_int_top[:, :, vs.tau])

    vs.snow_melt_top = update(
        vs.snow_melt_top,
        at[2:-2, 2:-2], 0,
    )

    vs.snow_melt_top = update(
        vs.snow_melt_top,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], snow_melt_pot[2:-2, 2:-2], vs.snow_melt_top[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.snow_melt_top = update(
        vs.snow_melt_top,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], vs.swe_top[2:-2, 2:-2, vs.tau], vs.snow_melt_top[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # snow melt when snow cover contains liquid water
    wtmx = allocate(state.dimensions, ("x", "y"))
    wtmx = (10000. / (100 - settings.rmax) / 100.) * vs.swe_top[:, :, vs.tau]
    mask3 = (vs.swe_top[:, :, vs.tau] < vs.S_int_top[:, :, vs.tau]) & (vs.swe_top[:, :, vs.tau] > 0) & (vs.S_int_top[:, :, vs.tau] > wtmx)
    vs.snow_melt_top = update_add(
        vs.snow_melt_top,
        at[2:-2, 2:-2], (vs.S_int_top[2:-2, 2:-2, vs.tau] - wtmx[2:-2, 2:-2]) * mask3[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # energy consumption of snow melt, will be subtracted from
    # evapotranspiration snow cover is partly melted, some snow cover
    # remains
    mask4 = (vs.snow_melt_top > 0) & (vs.snow_melt_top <= vs.swe_top[:, :, vs.tau])
    # snow cover is fully melted, no snow cover left
    mask5 = (vs.snow_melt_top > 0) & (vs.snow_melt_top > vs.swe_top[:, :, vs.tau])

    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.snow_melt_top[2:-2, 2:-2] * mask4[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_top = update_add(
        vs.swe_top,
        at[2:-2, 2:-2, vs.tau], -vs.snow_melt_top[2:-2, 2:-2] * mask4[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.swe_top[2:-2, 2:-2, vs.tau] * mask5[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_top = update_add(
        vs.swe_top,
        at[2:-2, 2:-2, vs.tau], npx.where(mask5[2:-2, 2:-2], 0, -vs.swe_top[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
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
        at[2:-2, 2:-2], (settings.sf * (vs.ta[2:-2, 2:-2, vs.tau] - settings.ta_fm) * vs.dt) * vs.maskCatch[2:-2, 2:-2],
    )

    mask1 = (snow_melt_pot > 0) & (snow_melt_pot <= vs.swe_ground[:, :, vs.tau]) & (vs.swe_ground[:, :, vs.tau] == vs.S_int_ground[:, :, vs.tau])
    mask2 = (snow_melt_pot > 0) & (snow_melt_pot > vs.swe_ground[:, :, vs.tau]) & (vs.swe_ground[:, :, vs.tau] == vs.S_int_ground[:, :, vs.tau])

    vs.snow_melt_ground = update(
        vs.snow_melt_ground,
        at[2:-2, 2:-2], 0,
    )

    vs.snow_melt_ground = update(
        vs.snow_melt_ground,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], snow_melt_pot[2:-2, 2:-2], vs.snow_melt_ground[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.snow_melt_ground = update(
        vs.snow_melt_ground,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], vs.swe_ground[2:-2, 2:-2, vs.tau], vs.snow_melt_ground[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # snow melt when snow cover contains liquid water
    wtmx = allocate(state.dimensions, ("x", "y"))
    wtmx = (10000. / (100 - settings.rmax) / 100.) * vs.swe_ground[:, :, vs.tau]
    mask3 = (vs.swe_ground[:, :, vs.tau] < vs.S_int_ground[:, :, vs.tau]) & (vs.swe_ground[:, :, vs.tau] > 0) & (vs.S_int_ground[:, :, vs.tau] > wtmx)
    vs.snow_melt_ground = update_add(
        vs.snow_melt_ground,
        at[2:-2, 2:-2], (vs.S_int_ground[2:-2, 2:-2, vs.tau] - wtmx[2:-2, 2:-2]) * mask3[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # energy consumption of snow melt, will be subtracted from
    # evapotranspiration snow cover is partly melted, some snow cover
    # remains
    mask4 = (vs.snow_melt_ground > 0) & (vs.snow_melt_ground <= vs.swe_ground[:, :, vs.tau])
    # snow cover is fully melted, no snow cover left
    mask5 = (vs.snow_melt_ground > 0) & (vs.snow_melt_ground > vs.swe_ground[:, :, vs.tau])

    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.snow_melt_ground[2:-2, 2:-2] * mask4[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_ground = update_add(
        vs.swe_ground,
        at[2:-2, 2:-2, vs.tau], -vs.snow_melt_ground[2:-2, 2:-2] * mask4[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.swe_ground[2:-2, 2:-2, vs.tau] * mask5[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_ground = update_add(
        vs.swe_ground,
        at[2:-2, 2:-2, vs.tau], npx.where(mask5[2:-2, 2:-2], 0, -vs.swe_ground[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
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
        at[2:-2, 2:-2], (settings.sf * (vs.ta[2:-2, 2:-2, vs.tau] - settings.ta_fm) * vs.dt) * vs.maskCatch[2:-2, 2:-2],
    )
    mask1 = (snow_melt_pot > 0) & (snow_melt_pot <= vs.swe[:, :, vs.tau]) & (vs.swe[:, :, vs.tau] > 0)
    mask2 = (snow_melt_pot > 0) & (snow_melt_pot > vs.swe[:, :, vs.tau]) & (vs.swe[:, :, vs.tau] > 0)

    vs.snow_melt = update(
        vs.snow_melt,
        at[2:-2, 2:-2], 0,
    )
    vs.snow_melt = update(
        vs.snow_melt,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], snow_melt_pot[2:-2, 2:-2], vs.snow_melt[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.snow_melt = update(
        vs.snow_melt,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], vs.swe[2:-2, 2:-2, vs.tau], vs.snow_melt[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # snow melt when snow cover contains liquid water
    # retention storage
    wtmx = allocate(state.dimensions, ("x", "y"))
    wtmx = (10000. / (100 - settings.rmax) / 100.) * vs.swe[:, :, vs.tau]
    # retained water
    q_ret = allocate(state.dimensions, ("x", "y"))
    q_ret = update(
        q_ret,
        at[2:-2, 2:-2], (vs.S_snow[2:-2, 2:-2, vs.tau] - vs.swe[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    mask3 = (vs.swe[:, :, vs.tau] < vs.S_snow[:, :, vs.tau]) & (vs.swe[:, :, vs.tau] > 0) & (q_ret > wtmx)
    vs.q_rain_on_snow = update(
        vs.q_rain_on_snow,
        at[2:-2, 2:-2], 0,
    )
    vs.q_rain_on_snow = update(
        vs.q_rain_on_snow,
        at[2:-2, 2:-2], npx.where(mask3[2:-2, 2:-2], q_ret[2:-2, 2:-2] - wtmx[2:-2, 2:-2], vs.q_rain_on_snow[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.q_rain_on_snow = update(
        vs.q_rain_on_snow,
        at[2:-2, 2:-2], npx.where(vs.q_rain_on_snow[2:-2, 2:-2] < 0, 0, vs.q_rain_on_snow[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    mask4 = (vs.S_snow[:, :, vs.tau] > 0) & (vs.swe[:, :, vs.tau] <= 0) & (vs.snow_melt <= 0)
    vs.q_rain_on_snow = update(
        vs.q_rain_on_snow,
        at[2:-2, 2:-2], npx.where(mask4[2:-2, 2:-2], vs.S_snow[2:-2, 2:-2, vs.tau], vs.q_rain_on_snow[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    mask44 = (vs.ta[:, :, vs.tau] < 0)
    vs.q_rain_on_snow = update(
        vs.q_rain_on_snow,
        at[2:-2, 2:-2], npx.where(mask44[2:-2, 2:-2], 0, vs.q_rain_on_snow[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # energy consumption of snow melt, will be subtracted from
    # evapotranspiration snow cover is partly melted, some snow cover
    # remains
    mask5 = (vs.snow_melt > 0) & (vs.q_rain_on_snow <= 0) & (vs.snow_melt <= vs.swe[:, :, vs.tau]) & (vs.snow_melt <= vs.S_snow[:, :, vs.tau])
    # snow melt, snow cover is fully melted
    mask6 = (vs.snow_melt > 0) & (vs.q_rain_on_snow <= 0) & (vs.snow_melt > vs.swe[:, :, vs.tau]) & (vs.snow_melt <= vs.S_snow[:, :, vs.tau])
    # rain-on-snow
    mask7 = (vs.snow_melt > 0) & (vs.q_rain_on_snow > 0) & (vs.snow_melt <= vs.swe[:, :, vs.tau]) & (vs.q_rain_on_snow + vs.snow_melt <= vs.S_snow[:, :, vs.tau])
    # snow cover, snow cover is fully melted
    mask8 = (vs.snow_melt > 0) & (vs.q_rain_on_snow > 0) & (vs.snow_melt > vs.swe[:, :, vs.tau]) & (vs.q_rain_on_snow + vs.snow_melt <= vs.S_snow[:, :, vs.tau])
    # residual snow melt
    mask9 = (vs.S_snow[:, :, vs.tau] > 0) & (vs.swe[:, :, vs.tau] <= 0)

    # snow melt
    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.snow_melt[2:-2, 2:-2] * mask5[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe = update_add(
        vs.swe,
        at[2:-2, 2:-2, vs.tau], -vs.snow_melt[2:-2, 2:-2] * mask5[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_snow = update_add(
        vs.S_snow,
        at[2:-2, 2:-2, vs.tau], -vs.snow_melt[2:-2, 2:-2] * mask5[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # snow cover is fully melted
    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.swe[2:-2, 2:-2, vs.tau] * mask6[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe = update(
        vs.swe,
        at[2:-2, 2:-2, vs.tau], npx.where(mask6[2:-2, 2:-2], 0, vs.swe[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_snow = update_add(
        vs.S_snow,
        at[2:-2, 2:-2, vs.tau], -vs.snow_melt[2:-2, 2:-2] * mask6[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # rain-on-snow
    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.snow_melt[2:-2, 2:-2] * mask7[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe = update_add(
        vs.swe,
        at[2:-2, 2:-2, vs.tau], -vs.snow_melt[2:-2, 2:-2] * mask7[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_snow = update_add(
        vs.S_snow,
        at[2:-2, 2:-2, vs.tau], -(vs.snow_melt[2:-2, 2:-2] + vs.q_rain_on_snow[2:-2, 2:-2]) * mask7[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # rain-on-snow, snow cover is fully melted
    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.snow_melt[2:-2, 2:-2] * mask8[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe = update(
        vs.swe,
        at[2:-2, 2:-2, vs.tau], npx.where(mask8[2:-2, 2:-2], 0, vs.swe[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_snow = update_add(
        vs.S_snow,
        at[2:-2, 2:-2, vs.tau], -(vs.snow_melt[2:-2, 2:-2] + vs.q_rain_on_snow[2:-2, 2:-2]) * mask8[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # residual snow melt
    vs.swe = update(
        vs.swe,
        at[2:-2, 2:-2, vs.tau], npx.where(mask9[2:-2, 2:-2], 0, vs.swe[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_snow = update(
        vs.S_snow,
        at[2:-2, 2:-2, vs.tau], npx.where(mask9[2:-2, 2:-2], 0, vs.S_snow[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_snow = update(
        vs.q_snow,
        at[2:-2, 2:-2], (vs.snow_melt[2:-2, 2:-2] + vs.q_rain_on_snow[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.z0 = update_add(
        vs.z0,
        at[2:-2, 2:-2, vs.tau], vs.q_snow[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.prec_event_sum = update_add(
        vs.prec_event_sum,
        at[2:-2, 2:-2, vs.tau], vs.q_snow[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(snow_melt=vs.snow_melt, q_rain_on_snow=vs.q_rain_on_snow, q_snow=vs.q_snow, z0=vs.z0, S_snow=vs.S_snow, swe=vs.swe, pet_res=vs.pet_res, prec_event_sum=vs.prec_event_sum)


@roger_routine
def calculate_snow(state):
    """
    Calculates snow accumlutaion, snow melt and rain-on-snow
    """
    vs = state.variables

    vs.update(calc_snow_accumulation(state))
    vs.update(calc_rain_on_snow(state))
    vs.update(calc_snow_melt_top_int(state))
    vs.update(calc_snow_melt_ground_int(state))
    vs.update(calc_snow_melt(state))
