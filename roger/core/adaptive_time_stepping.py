from roger import roger_kernel, roger_routine, KernelOutput
from roger.core.utilities import _get_first_row_no, _get_last_row_no
from roger.core.operators import numpy as npx, update, at
from roger.variables import allocate


@roger_routine
def adaptive_time_stepping(state):
    vs = state.variables
    settings = state.settings

    vs.update(adaptive_time_stepping_kernel(state))

    if settings.enable_film_flow:
        cond_event = (vs.EVENTS == vs.event_id_counter) & (vs.EVENTS > 0)
        t1 = _get_first_row_no(cond_event, vs.event_id[vs.tau])[0]
        t2 = _get_last_row_no(cond_event, vs.event_id[vs.tau])[0]
        prec_event = vs.PREC[t1:t2]
        
        vs.rain_event = update(vs.rain_event, at[2:-2, 2:-2, t1:t2], prec_event[npx.newaxis, npx.newaxis, :])
        vs.rain_event_csum = update(vs.rain_event_csum, at[2:-2, 2:-2, t1:t2], npx.cumsum(prec_event, axis=-1))
        vs.rain_event_sum = update(vs.rain_event_sum, at[2:-2, 2:-2], npx.sum(prec_event, axis=-1))


@roger_kernel
def adaptive_time_stepping_kernel(state):
    vs = state.variables
    settings = state.settings

    cond0 = npx.array(
        (vs.prec_day[2:-2, 2:-2, :] <= 0).all()
        & (vs.swe[2:-2, 2:-2, vs.tau] <= 0).all()
        & (vs.swe_top[2:-2, 2:-2, vs.tau] <= 0).all()
        & (vs.ta_day[2:-2, 2:-2, :] > settings.ta_fm).all()
    )
    cond00 = npx.array(
        ((vs.prec_day[2:-2, 2:-2, :] > 0) & (vs.ta_day[2:-2, 2:-2, :] <= settings.ta_fm)).any()
        | ((vs.prec_day[2:-2, 2:-2, :] <= 0) & (vs.ta_day[2:-2, 2:-2, :] <= settings.ta_fm)).all()
    )
    cond1 = npx.array(
        (vs.prec_day[2:-2, 2:-2, :] > settings.hpi).any()
        & (vs.prec_day[2:-2, 2:-2, :] > 0).any()
        & (vs.ta_day[2:-2, 2:-2, :] > settings.ta_fm).any()
    )
    cond2 = npx.array(
        (vs.prec_day[2:-2, 2:-2, :] <= settings.hpi).all()
        & (vs.prec_day[2:-2, 2:-2, :] > 0).any()
        & (vs.ta_day[2:-2, 2:-2, :] > settings.ta_fm).any()
    )
    cond3 = npx.array(
        (vs.prec_day[2:-2, 2:-2, :] > settings.hpi).any()
        & (vs.prec_day[2:-2, 2:-2, :] > 0).any()
        & (
            ((vs.swe[2:-2, 2:-2, vs.tau] > 0).any() | (vs.swe_top[2:-2, 2:-2, vs.tau] > 0).any())
            & (vs.ta_day[2:-2, 2:-2, :] > settings.ta_fm).any()
        )
    )
    cond4 = npx.array(
        (vs.prec_day[2:-2, 2:-2, :] <= settings.hpi).all()
        & (vs.prec_day[2:-2, 2:-2, :] > 0).any()
        & (
            ((vs.swe[2:-2, 2:-2, vs.tau] > 0).any() | (vs.swe_top[2:-2, 2:-2, vs.tau] > 0).any())
            & (vs.ta_day[2:-2, 2:-2, :] > settings.ta_fm).any()
        )
    )
    cond5 = npx.array(
        (vs.prec_day[2:-2, 2:-2, :] <= 0).all()
        & (
            ((vs.swe[2:-2, 2:-2, vs.tau] > 0).any() | (vs.swe_top[2:-2, 2:-2, vs.tau] > 0).any())
            & (vs.ta_day[2:-2, 2:-2, :] > settings.ta_fm).any()
        )
    )
    cond_time = npx.array((vs.time % (24 * 60 * 60) == 0))

    cond0_arr = allocate(state.dimensions, ("x", "y"))
    cond00_arr = allocate(state.dimensions, ("x", "y"))
    cond1_arr = allocate(state.dimensions, ("x", "y"))
    cond2_arr = allocate(state.dimensions, ("x", "y"))
    cond3_arr = allocate(state.dimensions, ("x", "y"))
    cond4_arr = allocate(state.dimensions, ("x", "y"))
    cond5_arr = allocate(state.dimensions, ("x", "y"))
    cond0_arr = update(
        cond0_arr,
        at[2:-2, 2:-2],
        npx.where(cond0[npx.newaxis, npx.newaxis], 1, 0),
    )
    cond00_arr = update(
        cond00_arr,
        at[2:-2, 2:-2],
        npx.where(cond00[npx.newaxis, npx.newaxis], 1, 0),
    )
    cond1_arr = update(
        cond1_arr,
        at[2:-2, 2:-2],
        npx.where(cond1[npx.newaxis, npx.newaxis], 1, 0),
    )
    cond2_arr = update(
        cond2_arr,
        at[2:-2, 2:-2],
        npx.where(cond2[npx.newaxis, npx.newaxis], 1, 0),
    )
    cond3_arr = update(
        cond3_arr,
        at[2:-2, 2:-2],
        npx.where(cond3[npx.newaxis, npx.newaxis], 1, 0),
    )
    cond4_arr = update(
        cond4_arr,
        at[2:-2, 2:-2],
        npx.where(cond4[npx.newaxis, npx.newaxis], 1, 0),
    )
    cond5_arr = update(
        cond5_arr,
        at[2:-2, 2:-2],
        npx.where(cond5[npx.newaxis, npx.newaxis], 1, 0),
    )

    prec_daily, ta_daily, pet_daily = calc_prec_ta_pet_daily(state)
    prec_hourly, ta_hourly, pet_hourly = calc_prec_ta_pet_hourly(state)
    prec_10mins, ta_10mins, pet_10mins = calc_prec_ta_pet_10_minutes(state)

    # no event or snowfall - daily time steps
    vs.prec = update(
        vs.prec,
        at[2:-2, 2:-2, vs.tau],
        npx.where(
            (cond0_arr[2:-2, 2:-2] == 1) | (cond00_arr[2:-2, 2:-2] == 1), prec_daily, vs.prec[2:-2, 2:-2, vs.tau]
        ),
    )
    vs.ta = update(
        vs.ta,
        at[2:-2, 2:-2, vs.tau],
        npx.where(((cond0_arr[2:-2, 2:-2] == 1) | (cond00_arr[2:-2, 2:-2] == 1)), ta_daily, vs.ta[2:-2, 2:-2, vs.tau]),
    )
    vs.dt_secs = npx.where((cond0 | cond00), 24 * 60 * 60, vs.dt_secs)
    vs.dt_secs = npx.where(cond_time, 24 * 60 * 60, 60 * 60)

    # rainfall/snow melt event - hourly time steps
    vs.prec = update(
        vs.prec,
        at[2:-2, 2:-2, vs.tau],
        npx.where(
            ((cond2_arr[2:-2, 2:-2] == 1) | (cond4_arr[2:-2, 2:-2] == 1) | (cond5_arr[2:-2, 2:-2] == 1))
            & ((cond1_arr[2:-2, 2:-2] == 0) & (cond3_arr[2:-2, 2:-2] == 0)),
            prec_hourly,
            vs.prec[2:-2, 2:-2, vs.tau],
        ),
    )
    vs.ta = update(
        vs.ta,
        at[2:-2, 2:-2, vs.tau],
        npx.where(
            ((cond2_arr[2:-2, 2:-2] == 1) | (cond4_arr[2:-2, 2:-2] == 1) | (cond5_arr[2:-2, 2:-2] == 1))
            & ((cond1_arr[2:-2, 2:-2] == 0) & (cond3_arr[2:-2, 2:-2] == 0)),
            ta_hourly,
            vs.ta[2:-2, 2:-2, vs.tau],
        ),
    )
    vs.dt_secs = npx.where((cond2 | cond4 | cond5) & ~cond1 & ~cond3, 60 * 60, vs.dt_secs)
    # heavy rainfall event - 10 minutes time steps
    vs.prec = update(
        vs.prec,
        at[2:-2, 2:-2, vs.tau],
        npx.where(
            ((cond1_arr[2:-2, 2:-2] == 1) | (cond3_arr[2:-2, 2:-2] == 1))
            & ((cond2_arr[2:-2, 2:-2] == 0) & (cond4_arr[2:-2, 2:-2] == 0) & (cond5_arr[2:-2, 2:-2] == 0)),
            prec_10mins,
            vs.prec[2:-2, 2:-2, vs.tau],
        ),
    )
    vs.ta = update(
        vs.ta,
        at[2:-2, 2:-2, vs.tau],
        npx.where(
            ((cond1_arr[2:-2, 2:-2] == 1) | (cond3_arr[2:-2, 2:-2] == 1))
            & ((cond2_arr[2:-2, 2:-2] == 0) & (cond4_arr[2:-2, 2:-2] == 0) & (cond5_arr[2:-2, 2:-2] == 0)),
            ta_10mins,
            vs.ta[2:-2, 2:-2, vs.tau],
        ),
    )
    vs.dt_secs = npx.where((cond1 | cond3) & ~cond2 & ~cond4 & ~cond5, 10 * 60, vs.dt_secs)

    # determine end of event
    cond_event1 = (
        (vs.prec[2:-2, 2:-2] > 0).any()
        | ((vs.swe[2:-2, 2:-2, vs.tau] > 0) | (vs.swe_top[2:-2, 2:-2, vs.tau] > 0)).any()
        & (vs.ta[2:-2, 2:-2] > settings.ta_fm)
    ).any()
    cond_event2 = (
        ((vs.prec[2:-2, 2:-2] <= 0) & (vs.ta[2:-2, 2:-2] > settings.ta_fm)).all()
        | ((vs.prec[2:-2, 2:-2] > 0) & (vs.ta[2:-2, 2:-2] <= settings.ta_fm)).all()
        | ((vs.swe[2:-2, 2:-2, vs.taum1] > 0).any() & (vs.swe[2:-2, 2:-2, vs.tau] <= 0).all())
    )
    vs.time_event0 = npx.where(cond_event1, 0, vs.time_event0)
    vs.time_event0 = npx.where(cond_event2, vs.time_event0 + vs.dt_secs, vs.time_event0)

    # increase time stepping at end of event if either full hour
    # or full day, respectively
    cond6 = npx.array((vs.time_event0 <= settings.end_event) & (vs.dt_secs == 10 * 60))
    cond7 = npx.array((vs.time_event0 <= settings.end_event) & (vs.dt_secs == 60 * 60))
    cond8 = npx.array((vs.time_event0 <= settings.end_event) & (vs.dt_secs == 24 * 60 * 60))
    cond9 = npx.array((vs.time_event0 > settings.end_event) & (vs.time % (60 * 60) != 0) & (vs.dt_secs == 10 * 60))
    cond10 = npx.array(
        (vs.time_event0 > settings.end_event)
        & (vs.time % (60 * 60) == 0)
        & ((vs.dt_secs == 10 * 60) | (vs.dt_secs == 60 * 60))
    )
    cond11 = npx.array(
        (vs.time_event0 > settings.end_event) & (vs.time % (24 * 60 * 60) == 0) & (vs.dt_secs == 24 * 60 * 60)
    )
    cond6_arr = allocate(state.dimensions, ("x", "y"))
    cond7_arr = allocate(state.dimensions, ("x", "y"))
    cond8_arr = allocate(state.dimensions, ("x", "y"))
    cond9_arr = allocate(state.dimensions, ("x", "y"))
    cond10_arr = allocate(state.dimensions, ("x", "y"))
    cond11_arr = allocate(state.dimensions, ("x", "y"))
    cond6_arr = update(
        cond6_arr,
        at[2:-2, 2:-2],
        npx.where(cond6[npx.newaxis, npx.newaxis], 1, 0),
    )
    cond7_arr = update(
        cond7_arr,
        at[2:-2, 2:-2],
        npx.where(cond7[npx.newaxis, npx.newaxis], 1, 0),
    )
    cond8_arr = update(
        cond8_arr,
        at[2:-2, 2:-2],
        npx.where(cond8[npx.newaxis, npx.newaxis], 1, 0),
    )
    cond9_arr = update(
        cond9_arr,
        at[2:-2, 2:-2],
        npx.where(cond9[npx.newaxis, npx.newaxis], 1, 0),
    )
    cond10_arr = update(
        cond10_arr,
        at[2:-2, 2:-2],
        npx.where(cond10[npx.newaxis, npx.newaxis], 1, 0),
    )
    cond11_arr = update(
        cond11_arr,
        at[2:-2, 2:-2],
        npx.where(cond11[npx.newaxis, npx.newaxis], 1, 0),
    )

    vs.pet = update(
        vs.pet,
        at[2:-2, 2:-2],
        npx.where((cond6_arr[2:-2, 2:-2] == 1), pet_10mins, vs.pet[2:-2, 2:-2]),
    )
    vs.ta = update(
        vs.ta,
        at[2:-2, 2:-2, vs.tau],
        npx.where((cond6_arr[2:-2, 2:-2] == 1), ta_10mins, vs.ta[2:-2, 2:-2, vs.tau]),
    )
    vs.event_id = update(
        vs.event_id,
        at[vs.tau],
        npx.where(cond6, vs.event_id_counter, vs.event_id[vs.tau]),
    )
    vs.dt = npx.where(cond6, 1 / 6, vs.dt)
    vs.itt_day = npx.where(cond6, vs.itt_day + 1, vs.itt_day)

    vs.pet = update(
        vs.pet,
        at[2:-2, 2:-2],
        npx.where((cond7_arr[2:-2, 2:-2] == 1), pet_hourly, vs.pet[2:-2, 2:-2]),
    )
    vs.ta = update(
        vs.ta,
        at[2:-2, 2:-2, vs.tau],
        npx.where((cond7_arr[2:-2, 2:-2] == 1), ta_hourly, vs.ta[2:-2, 2:-2, vs.tau]),
    )
    vs.event_id = update(
        vs.event_id,
        at[vs.tau],
        npx.where(cond7, vs.event_id_counter, vs.event_id[vs.tau]),
    )
    vs.dt = npx.where(cond7, 1, vs.dt)
    vs.itt_day = npx.where(cond7, vs.itt_day + 6, vs.itt_day)

    vs.pet = update(
        vs.pet,
        at[2:-2, 2:-2],
        npx.where((cond8_arr[2:-2, 2:-2] == 1), pet_daily, vs.pet[2:-2, 2:-2]),
    )
    vs.ta = update(
        vs.ta,
        at[2:-2, 2:-2, vs.tau],
        npx.where((cond8_arr[2:-2, 2:-2] == 1), ta_daily, vs.ta[2:-2, 2:-2, vs.tau]),
    )
    vs.dt = npx.where(cond8, 24, vs.dt)
    vs.itt_day = npx.where(cond8, 0, vs.itt_day)

    vs.pet = update(
        vs.pet,
        at[2:-2, 2:-2],
        npx.where((cond9_arr[2:-2, 2:-2] == 1), pet_10mins, vs.pet[2:-2, 2:-2]),
    )
    vs.ta = update(
        vs.ta,
        at[2:-2, 2:-2, vs.tau],
        npx.where((cond9_arr[2:-2, 2:-2] == 1), ta_10mins, vs.ta[2:-2, 2:-2, vs.tau]),
    )
    vs.event_id = update(
        vs.event_id,
        at[vs.tau],
        npx.where(cond9, 0, vs.event_id[vs.tau]),
    )
    vs.dt = npx.where(cond9, 1 / 6, vs.dt)
    vs.dt_secs = npx.where(cond9, 10 * 60, vs.dt_secs)
    vs.itt_day = npx.where(cond9, vs.itt_day + 1, vs.itt_day)

    vs.pet = update(
        vs.pet,
        at[2:-2, 2:-2],
        npx.where((cond10_arr[2:-2, 2:-2] == 1), pet_hourly, vs.pet[2:-2, 2:-2]),
    )
    vs.ta = update(
        vs.ta,
        at[2:-2, 2:-2, vs.tau],
        npx.where((cond10_arr[2:-2, 2:-2] == 1), ta_hourly, vs.ta[2:-2, 2:-2, vs.tau]),
    )
    vs.event_id = update(
        vs.event_id,
        at[vs.tau],
        npx.where(cond10, 0, vs.event_id[vs.tau]),
    )
    vs.dt = npx.where(cond10, 1, vs.dt)
    vs.dt_secs = npx.where(cond10, 60 * 60, vs.dt_secs)
    vs.itt_day = npx.where(cond10, vs.itt_day + 6, vs.itt_day)

    vs.pet = update(
        vs.pet,
        at[2:-2, 2:-2],
        npx.where((cond11_arr[2:-2, 2:-2] == 1), pet_daily, vs.pet[2:-2, 2:-2]),
    )
    vs.ta = update(
        vs.ta,
        at[2:-2, 2:-2, vs.tau],
        npx.where((cond11_arr[2:-2, 2:-2] == 1), ta_daily, vs.ta[2:-2, 2:-2, vs.tau]),
    )
    vs.event_id = update(
        vs.event_id,
        at[vs.tau],
        npx.where(cond11, 0, vs.event_id[vs.tau]),
    )
    vs.dt = npx.where(cond11, 24, vs.dt)
    vs.dt_secs = npx.where(cond11, 24 * 60 * 60, vs.dt_secs)
    vs.itt_day = npx.where(cond11, 0, vs.itt_day)

    # set event id for next event
    vs.event_id_counter = npx.where(
        (vs.event_id[vs.taum1] > 0) & (vs.event_id[vs.tau] == 0), vs.event_id_counter + 1, vs.event_id_counter
    )

    # set residual PET
    vs.pet_res = update(vs.pet_res, at[2:-2, 2:-2], vs.pet[2:-2, 2:-2])

    return KernelOutput(
        prec=vs.prec,
        ta=vs.ta,
        pet=vs.pet,
        pet_res=vs.pet_res,
        dt=vs.dt,
        dt_secs=vs.dt_secs,
        itt_day=vs.itt_day,
        event_id=vs.event_id,
        time_event0=vs.time_event0,
        event_id_counter=vs.event_id_counter,
    )


@roger_kernel
def calc_prec_ta_pet_10_minutes(state):
    vs = state.variables

    prec = vs.prec_day[2:-2, 2:-2, vs.itt_day]
    ta = vs.ta_day[2:-2, 2:-2, vs.itt_day]
    pet = vs.pet_day[2:-2, 2:-2, vs.itt_day]

    return prec, ta, pet


@roger_kernel
def calc_prec_ta_pet_hourly(state):
    vs = state.variables
    idx = allocate(state.dimensions, ("x", "y", 6 * 24))
    idx = update(
        idx,
        at[2:-2, 2:-2, :],
        npx.arange(0, 6 * 24)[npx.newaxis, npx.newaxis, :],
    )

    prec = npx.sum(
        npx.where(
            (idx[2:-2, 2:-2, :] >= vs.itt_day) & (idx[2:-2, 2:-2, :] < vs.itt_day + 6), vs.prec_day[2:-2, 2:-2, :], 0
        ),
        axis=-1,
    )
    ta = npx.nanmean(
        npx.where(
            (idx[2:-2, 2:-2, :] >= vs.itt_day) & (idx[2:-2, 2:-2, :] < vs.itt_day + 6),
            vs.ta_day[2:-2, 2:-2, :],
            npx.nan,
        ),
        axis=-1,
    )
    pet = npx.sum(
        npx.where(
            (idx[2:-2, 2:-2, :] >= vs.itt_day) & (idx[2:-2, 2:-2, :] < vs.itt_day + 6), vs.pet_day[2:-2, 2:-2, :], 0
        ),
        axis=-1,
    )

    return prec, ta, pet


@roger_kernel
def calc_prec_ta_pet_daily(state):
    vs = state.variables

    prec = npx.sum(vs.prec_day, axis=-1)[2:-2, 2:-2]
    ta = npx.nanmean(vs.ta_day[2:-2, 2:-2, 0 : 24 * 6], axis=-1)
    pet = npx.sum(vs.pet_day[2:-2, 2:-2, 0 : 24 * 6], axis=-1)

    return prec, ta, pet
