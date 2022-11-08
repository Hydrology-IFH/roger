from roger import roger_kernel, roger_routine, KernelOutput, logger
from roger.core.operators import numpy as npx, update, at, update_add
from roger.variables import allocate


@roger_routine
def adaptive_time_stepping(state):
    vs = state.variables
    settings = state.settings

    cond0 = (vs.prec_day[2:-2, 2:-2, :] <= 0).all() & (vs.swe[2:-2, 2:-2, vs.tau] <= 0).all() & (vs.swe_top[2:-2, 2:-2, vs.tau] <= 0).all() & (vs.ta_day[2:-2, 2:-2, :] > settings.ta_fm).all()
    cond00 = ((vs.prec_day[2:-2, 2:-2, :] > 0) & (vs.ta_day[2:-2, 2:-2, :] <= settings.ta_fm)).any() | ((vs.prec_day[2:-2, 2:-2, :] <= 0) & (vs.ta_day[2:-2, 2:-2, :] <= settings.ta_fm)).all()
    cond1 = (vs.prec_day[2:-2, 2:-2, :] > settings.hpi).any() & (vs.prec_day[2:-2, 2:-2, :] > 0).any() & (vs.ta_day[2:-2, 2:-2, :] > settings.ta_fm).any()
    cond2 = (vs.prec_day[2:-2, 2:-2, :] <= settings.hpi).all() & (vs.prec_day[2:-2, 2:-2, :] > 0).any() & (vs.ta_day[2:-2, 2:-2, :] > settings.ta_fm).any()
    cond3 = (vs.prec_day[2:-2, 2:-2, :] > settings.hpi).any() & (vs.prec_day[2:-2, 2:-2, :] > 0).any() & (((vs.swe[2:-2, 2:-2, vs.tau] > 0).any() | (vs.swe_top[2:-2, 2:-2, vs.tau] > 0).any()) & (vs.ta_day[2:-2, 2:-2, :] > settings.ta_fm).any())
    cond4 = (vs.prec_day[2:-2, 2:-2, :] <= settings.hpi).all() & (vs.prec_day[2:-2, 2:-2, :] > 0).any() & (((vs.swe[2:-2, 2:-2, vs.tau] > 0).any() | (vs.swe_top[2:-2, 2:-2, vs.tau] > 0).any()) & (vs.ta_day[2:-2, 2:-2, :] > settings.ta_fm).any())
    cond5 = (vs.prec_day[2:-2, 2:-2, :] <= 0).all() & (((vs.swe[2:-2, 2:-2, vs.tau] > 0).any() | (vs.swe_top[2:-2, 2:-2, vs.tau] > 0).any()) & (vs.ta_day[2:-2, 2:-2, :] > settings.ta_fm).any())
    # no event or snowfall - daily time steps
    if cond0 or cond00:
        vs.update(calc_prec_ta_daily(state))
        if (vs.time % (24 * 60 * 60) == 0):
            vs.dt_secs = 24 * 60 * 60
        else:
            vs.dt_secs = 60 * 60
    # rainfall/snow melt event - hourly time steps
    elif (cond2 or cond4 or cond5) and not cond1 and not cond3:
        vs.update(calc_prec_ta_hourly(state))
        vs.dt_secs = 60 * 60
    # heavy rainfall event - 10 minutes time steps
    elif (cond1 or cond3) and not cond2 and not cond4 and not cond5:
        vs.update(calc_prec_ta_10_minutes(state))
        vs.dt_secs = 10 * 60

    # determine end of event
    if ((vs.prec[2:-2, 2:-2] > 0).any() | ((vs.swe[2:-2, 2:-2, vs.tau] > 0) | (vs.swe_top[2:-2, 2:-2, vs.tau] > 0)).any() and (vs.ta[2:-2, 2:-2] > settings.ta_fm)).any():
        vs.time_event0 = 0
    elif ((vs.prec[2:-2, 2:-2] <= 0) & (vs.ta[2:-2, 2:-2] > settings.ta_fm)).all() or ((vs.prec[2:-2, 2:-2] > 0) & (vs.ta[2:-2, 2:-2] <= settings.ta_fm)).all() or ((vs.swe[2:-2, 2:-2, vs.taum1] > 0).any() & (vs.swe[2:-2, 2:-2, vs.tau] <= 0).all()):
        vs.time_event0 = vs.time_event0 + vs.dt_secs

    # increase time stepping at end of event if either full hour
    # or full day, respectively
    if vs.time_event0 <= settings.end_event and (vs.dt_secs == 10 * 60):
        vs.update(calc_ta_pet_10_minutes(state))
        vs.event_id = update(
            vs.event_id,
            at[vs.tau], vs.event_id_counter,
        )
        vs.dt = 1 / 6
        vs.itt_day = vs.itt_day + 1
    elif vs.time_event0 <= settings.end_event and (vs.dt_secs == 60 * 60):
        vs.update(calc_ta_pet_hourly(state))
        vs.event_id = update(
            vs.event_id,
            at[vs.tau], vs.event_id_counter,
        )
        vs.dt = 1
        vs.itt_day = vs.itt_day + 6
    elif vs.time_event0 <= settings.end_event and (vs.dt_secs == 24 * 60 * 60):
        vs.update(calc_ta_pet_daily(state))
        vs.dt = 24
        vs.itt_day = 0
    elif vs.time_event0 > settings.end_event and (vs.time % (60 * 60) != 0) and (vs.dt_secs == 10 * 60):
        vs.update(calc_ta_pet_10_minutes(state))
        vs.dt_secs = 10 * 60
        vs.dt = 1 / 6
        vs.itt_day = vs.itt_day + 1
        vs.event_id = update(
            vs.event_id,
            at[vs.tau], 0,
        )
    elif vs.time_event0 > settings.end_event and (vs.time % (60 * 60) == 0) and ((vs.dt_secs == 10 * 60) or (vs.dt_secs == 60 * 60)):
        vs.update(calc_ta_pet_hourly(state))
        vs.dt_secs = 60 * 60
        vs.dt = 1
        vs.itt_day = vs.itt_day + 6
        vs.event_id = update(
            vs.event_id,
            at[vs.tau], 0,
        )
    elif vs.time_event0 > settings.end_event and (vs.time % (24 * 60 * 60) == 0) and (vs.dt_secs == 24 * 60 * 60):
        vs.update(calc_ta_pet_daily(state))
        vs.dt_secs = 24 * 60 * 60
        vs.dt = 24
        vs.itt_day = 0
        vs.event_id = update(
            vs.event_id,
            at[vs.tau], 0,
        )

    # set event id for next event
    if (vs.event_id[vs.taum1] > 0) & (vs.event_id[vs.tau] == 0):
        vs.event_id_counter = vs.event_id_counter + 1

    # set residual PET
    vs.pet_res = update(vs.pet_res, at[2:-2, 2:-2], vs.pet[2:-2, 2:-2])

    vs.prec_check = update_add(vs.prec_check, at[2:-2, 2:-2], vs.prec[2:-2, 2:-2, vs.tau])

    if (vs.prec_check > npx.sum(vs.prec_day, axis=-1) + 0.001).any():
        logger.warning(f"Precipitation diverged at iteration {vs.itt}.")


@roger_kernel
def calc_prec_ta_10_minutes(state):
    vs = state.variables

    vs.prec = update(
        vs.prec,
        at[2:-2, 2:-2, vs.tau], vs.prec_day[2:-2, 2:-2, vs.itt_day],
    )
    vs.ta = update(
        vs.ta,
        at[2:-2, 2:-2, vs.tau], vs.ta_day[2:-2, 2:-2, vs.itt_day],
    )

    return KernelOutput(ta=vs.ta,
                        prec=vs.prec,
                        )


@roger_kernel
def calc_prec_ta_hourly(state):
    vs = state.variables
    idx = allocate(state.dimensions, ("x", "y", 6*24))
    idx = update(
        idx,
        at[2:-2, 2:-2, :], npx.arange(0, 6*24)[npx.newaxis, npx.newaxis, :],
    )

    vs.prec = update(
        vs.prec,
        at[2:-2, 2:-2, vs.tau], npx.sum(npx.where((idx[2:-2, 2:-2, :] >= vs.itt_day) & (idx[2:-2, 2:-2, :] < vs.itt_day+6), vs.prec_day[2:-2, 2:-2, :], 0), axis=-1),
    )
    vs.ta = update(
        vs.ta,
        at[2:-2, 2:-2, vs.tau], npx.nanmean(npx.where((idx[2:-2, 2:-2, :] >= vs.itt_day) & (idx[2:-2, 2:-2, :] < vs.itt_day+6), vs.ta_day[2:-2, 2:-2, :], npx.nan), axis=-1),
    )

    return KernelOutput(ta=vs.ta,
                        prec=vs.prec,
                        )


@roger_kernel
def calc_prec_ta_daily(state):
    vs = state.variables

    vs.prec = update(
        vs.prec,
        at[2:-2, 2:-2, vs.tau], npx.sum(vs.prec_day, axis=-1)[2:-2, 2:-2],
    )
    vs.ta = update(
        vs.ta,
        at[2:-2, 2:-2, vs.tau], npx.mean(vs.ta_day[2:-2, 2:-2, 0:24*6], axis=-1),
    )

    return KernelOutput(ta=vs.ta,
                        prec=vs.prec,
                        )


@roger_kernel
def calc_ta_pet_10_minutes(state):
    vs = state.variables

    vs.ta = update(
        vs.ta,
        at[2:-2, 2:-2, vs.tau], vs.ta_day[2:-2, 2:-2, vs.itt_day],
    )
    vs.pet = update(
        vs.pet,
        at[2:-2, 2:-2], vs.pet_day[2:-2, 2:-2, vs.itt_day],
    )

    return KernelOutput(ta=vs.ta,
                        pet=vs.pet,
                        )


@roger_kernel
def calc_ta_pet_hourly(state):
    vs = state.variables

    idx = allocate(state.dimensions, ("x", "y", 6 * 24))
    idx = update(
        idx,
        at[2:-2, 2:-2, :], npx.arange(0, 6*24)[npx.newaxis, npx.newaxis, :],
    )

    vs.ta = update(
        vs.ta,
        at[2:-2, 2:-2, vs.tau], npx.nanmean(npx.where((idx[2:-2, 2:-2, :] >= vs.itt_day) & (idx[2:-2, 2:-2, :] < vs.itt_day+6), vs.ta_day[2:-2, 2:-2, :], npx.nan), axis=-1),
    )
    vs.pet = update(
        vs.pet,
        at[2:-2, 2:-2], npx.sum(npx.where((idx[2:-2, 2:-2, :] >= vs.itt_day) & (idx[2:-2, 2:-2, :] < vs.itt_day+6), vs.pet_day[2:-2, 2:-2, :], 0), axis=-1),
    )

    return KernelOutput(ta=vs.ta,
                        pet=vs.pet,
                        )


@roger_kernel
def calc_ta_pet_daily(state):
    vs = state.variables

    vs.ta = update(
        vs.ta,
        at[2:-2, 2:-2, vs.tau], npx.mean(vs.ta_day[2:-2, 2:-2, 0:24*6], axis=-1),
    )
    vs.pet = update(
        vs.pet,
        at[2:-2, 2:-2], npx.sum(vs.pet_day[2:-2, 2:-2, 0:24*6], axis=-1),
    )

    return KernelOutput(ta=vs.ta,
                        pet=vs.pet,
                        )
