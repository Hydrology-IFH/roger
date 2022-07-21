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

    mask1 = (vs.swe[:, :, vs.tau] > 0) & (vs.ta[:, :, vs.tau] > settings.ta_fm)

    vs.S_snow = update_add(
        vs.S_snow ,
        at[2:-2, 2:-2, vs.tau], vs.rain_ground[2:-2, 2:-2] * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(S_snow=vs.S_snow)


@roger_kernel
def calc_snow_melt_int_top(state):
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

    mask1 = (snow_melt_pot > 0) & (snow_melt_pot <= vs.swe_top[:, :, vs.tau]) & (vs.swe_top[:, :, vs.tau] > 0)
    mask2 = (snow_melt_pot > 0) & (snow_melt_pot > vs.swe_top[:, :, vs.tau]) & (vs.swe_top[:, :, vs.tau] > 0)

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

    # snow layer retention storage
    wtmx = allocate(state.dimensions, ("x", "y"))
    wtmx = (10000. / (100 - settings.rmax) / 100.) * vs.swe_top[:, :, vs.tau]
    # retained water
    q_ret = allocate(state.dimensions, ("x", "y"))
    q_ret = update(
        q_ret,
        at[2:-2, 2:-2], npx.where(vs.S_int_top[2:-2, 2:-2, vs.tau] > vs.S_int_top_tot[2:-2, 2:-2], vs.S_int_top[2:-2, 2:-2, vs.tau] - vs.swe_top[2:-2, 2:-2, vs.tau], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    # add snow melt dripping to snow storage
    vs.snow_melt_drip = update(
        vs.snow_melt_drip,
        at[2:-2, 2:-2], npx.where(q_ret[2:-2, 2:-2] > wtmx[2:-2, 2:-2], q_ret[2:-2, 2:-2] - wtmx[2:-2, 2:-2], npx.where((wtmx[2:-2, 2:-2] <= 0) & (vs.S_int_top_tot[2:-2, 2:-2] < vs.S_int_top[2:-2, 2:-2, vs.tau]), vs.S_int_top[2:-2, 2:-2, vs.tau] - vs.S_int_top_tot[2:-2, 2:-2], 0)) * vs.maskCatch[2:-2, 2:-2],
    )
    mask6 = (vs.S_int_top_tot[:, :] < vs.S_int_top[:, :, vs.tau])
    vs.S_snow = update_add(
        vs.S_snow,
        at[2:-2, 2:-2, vs.tau], npx.where(mask6[2:-2, 2:-2], vs.snow_melt_drip[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_int_top = update_add(
        vs.S_int_top,
        at[2:-2, 2:-2, vs.tau], npx.where(mask6[2:-2, 2:-2], -vs.snow_melt_drip[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(snow_melt_top=vs.snow_melt_top, pet_res=vs.pet_res, swe_top=vs.swe_top, S_int_top=vs.S_int_top, S_snow=vs.S_snow, snow_melt_drip=vs.snow_melt_drip)


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

    mask1 = (snow_melt_pot > 0) & (snow_melt_pot <= vs.swe_ground[:, :, vs.tau]) & (vs.swe_ground[:, :, vs.tau] > 0)
    mask2 = (snow_melt_pot > 0) & (snow_melt_pot > vs.swe_ground[:, :, vs.tau]) & (vs.swe_ground[:, :, vs.tau] > 0)

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

    # energy consumption of snow melt, will be subtracted from
    # evapotranspiration snow cover is partly melted, some snow cover
    # remains
    mask5 = (vs.snow_melt > 0) & (vs.snow_melt <= vs.swe[:, :, vs.tau])
    # snow melt, snow cover is fully melted
    mask6 = (vs.snow_melt > 0) & (vs.snow_melt > vs.swe[:, :, vs.tau])

    # snow melt
    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.snow_melt[2:-2, 2:-2] * mask5[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe = update_add(
        vs.swe,
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

    # snow layer retention storage
    wtmx = allocate(state.dimensions, ("x", "y"))
    wtmx = (10000. / (100 - settings.rmax) / 100.) * vs.swe[:, :, vs.tau]
    # retained water
    q_ret = allocate(state.dimensions, ("x", "y"))
    q_ret = update(
        q_ret,
        at[2:-2, 2:-2], npx.where(vs.S_snow[2:-2, 2:-2, vs.tau] > 0, vs.S_snow[2:-2, 2:-2, vs.tau] - vs.swe[2:-2, 2:-2, vs.tau], 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.q_snow = update(
        vs.q_snow,
        at[2:-2, 2:-2], 0
    )
    vs.q_snow = update(
        vs.q_snow,
        at[2:-2, 2:-2], npx.where(q_ret[2:-2, 2:-2] > wtmx[2:-2, 2:-2], q_ret[2:-2, 2:-2] - wtmx[2:-2, 2:-2], npx.where(wtmx[2:-2, 2:-2] <= 0, vs.S_snow[2:-2, 2:-2, vs.tau], 0)) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_snow = update_add(
        vs.S_snow,
        at[2:-2, 2:-2, vs.tau], -vs.q_snow[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.z0 = update_add(
        vs.z0,
        at[2:-2, 2:-2, vs.tau], vs.q_snow[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.prec_event_csum = update_add(
        vs.prec_event_csum,
        at[2:-2, 2:-2], vs.q_snow[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(snow_melt=vs.snow_melt, q_snow=vs.q_snow, z0=vs.z0, S_snow=vs.S_snow, swe=vs.swe, pet_res=vs.pet_res, prec_event_csum=vs.prec_event_csum)


@roger_routine
def calculate_snow(state):
    """
    Calculates snow accumlutaion, snow melt and rain-on-snow
    """
    vs = state.variables

    vs.update(calc_snow_accumulation(state))
    vs.update(calc_rain_on_snow(state))
    vs.update(calc_snow_melt_int_top(state))
    vs.update(calc_snow_melt_ground_int(state))
    vs.update(calc_snow_melt(state))
