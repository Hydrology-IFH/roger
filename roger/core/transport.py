from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at
from roger.core import sas


@roger_kernel
def calc_SA(state, SA, sa):
    """Calculate cumulative StorAge (residence time distribution).
    """
    vs = state.variables

    # cumulative StorAge (residence time distribution)
    SA = update(
        SA,
        at[:, :, vs.tau, 1:], npx.cumsum(sa[:, :, vs.tau, :], axis=-1),
    )
    SA = update(
        SA,
        at[:, :, vs.tau, 0], 0,
    )

    return SA


@roger_kernel
def calc_MSA(state, MSA, msa):
    """Calculate cumulative solute mass StorAge (residence time distribution).
    """
    vs = state.variables

    # cumulative StorAge (residence time distribution)
    MSA = update(
        MSA,
        at[:, :, vs.tau, 1:], npx.cumsum(msa[:, :, vs.tau, :], axis=-1),
    )
    MSA = update(
        MSA,
        at[:, :, vs.tau, 0], 0,
    )

    return MSA


@roger_kernel
def calc_tt(state, SA, sa, flux, sas_params):
    """Calculates backward travel time distribution.
    """
    vs = state.variables

    TT = allocate(state.dimensions, ("x", "y", "nages"))
    tt = allocate(state.dimensions, ("x", "y", "ages"))
    # cumulative backward travel time distribution
    TT = update_add(
        TT,
        at[:, :, :], sas.uniform(state, SA, sas_params),
    )
    TT = update_add(
        TT,
        at[:, :, :], sas.dirac(state, sas_params),
    )
    TT = update_add(
        TT,
        at[:, :, :], sas.kumaraswami(state, SA, sas_params),
    )
    TT = update_add(
        TT,
        at[:, :, :], sas.gamma(state, SA, sas_params),
    )
    TT = update_add(
        TT,
        at[:, :, :], sas.exponential(state, SA, sas_params),
    )
    TT = update_add(
        TT,
        at[:, :, :], sas.power(state, SA, sas_params),
    )

    # travel time distribution
    mask_old = npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([1, 22, 32, 34, 52, 62])) | (npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([3])) & (sas_params[:, :, 1, npx.newaxis] >= sas_params[:, :, 2, npx.newaxis]))
    mask_young = npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([21, 31, 33, 4, 41, 51, 61])) | (npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([3])) & (sas_params[:, :, 1, npx.newaxis] < sas_params[:, :, 2, npx.newaxis]))

    tt = update(
        tt,
        at[:, :, :], npx.diff(TT, axis=2),
    )
    # set travel time distribution for dirac function
    tt = update(
        tt,
        at[:, :, :], npx.where(sas_params[:, :, 0, npx.newaxis] == 21, 0, tt),
    )
    tt = update(
        tt,
        at[:, :, 0], npx.where(sas_params[:, :, 0] == 21, 1, tt[:, :, 0]),
    )
    tt = update(
        tt,
        at[:, :, :], npx.where(sas_params[:, :, 0, npx.newaxis] == 22, 0, tt),
    )
    tt = update(
        tt,
        at[:, :, -1], npx.where(sas_params[:, :, 0] == 22, 1, tt[:, :, -1]),
    )
    # outflux
    flux_tt = allocate(state.dimensions, ("x", "y", "ages"))
    flux_tt = update(
        flux_tt,
        at[:, :, :], flux[:, :, npx.newaxis] * tt,
    )

    # set travel time distribution to zero if no outflux
    tt = update(
        tt,
        at[:, :, :], npx.where(flux_tt > 0, tt, 0),
    )

    TT = update(
        TT,
        at[:, :, 1:], npx.cumsum(tt, axis=2),
    )
    TT = update(
        TT,
        at[:, :, 0], 0,
    )

    # outflux is greater than StorAge
    # redistribution of probabilities to guarantee mass conservation
    flux_tt_init = allocate(state.dimensions, ("x", "y", "ages"))
    diff_q = allocate(state.dimensions, ("x", "y", "ages"))
    q_re = allocate(state.dimensions, ("x", "y", "ages"))
    q_re_sum = allocate(state.dimensions, ("x", "y"))
    sa_free = allocate(state.dimensions, ("x", "y", "ages"))
    sa_free_norm = allocate(state.dimensions, ("x", "y", "ages"))
    SA_re = allocate(state.dimensions, ("x", "y", "nages"))
    omega_re = allocate(state.dimensions, ("x", "y", "ages"))
    Omega_re = allocate(state.dimensions, ("x", "y", "nages"))
    flux_tt_re = allocate(state.dimensions, ("x", "y", "ages"))
    # outflux is greater than StorAge
    # difference between outflux and StorAge
    flux_tt_init = update(
        flux_tt_init,
        at[:, :, :], flux_tt,
    )
    diff_q = update(
        diff_q,
        at[:, :, :], flux_tt - sa[:, :, vs.tau, :],
    )
    q_re = update(
        q_re,
        at[:, :, :], npx.where((diff_q < 0), 0, diff_q),
    )
    q_re_sum = update(
        q_re_sum,
        at[:, :], npx.sum(q_re, axis=-1),
    )
    # available StorAge for sampling
    sa_free = update(
        sa_free,
        at[:, :, :], npx.where((diff_q > 0), 0, sa[:, :, vs.tau, :] - flux_tt),
    )
    # normalize StorAge with outflux of age T
    sa_free_norm = update(
        sa_free_norm,
        at[:, :, :], npx.where((q_re_sum[:, :, npx.newaxis] > 0), sa_free / q_re_sum[:, :, npx.newaxis], 0),
    )
    SA_re = update(
        SA_re,
        at[:, :, 1:], npx.where(mask_old, npx.cumsum(sa_free_norm[..., ::-1], axis=-1)[..., ::-1], SA_re[:, :, 1:]),
    )
    SA_re = update(
        SA_re,
        at[:, :, 1:], npx.where(mask_young, npx.cumsum(sa_free_norm, axis=-1), SA_re[:, :, 1:]),
    )
    # cumulative probability to redistribute outflux
    Omega_re = update(
        Omega_re,
        at[:, :, :], npx.where(SA_re < 1, SA_re, 1),
    )
    # probability to redistribute outflux
    omega_re = update(
        omega_re,
        at[:, :, :], npx.diff(Omega_re, axis=-1),
    )
    # redistribute outflux
    flux_tt_re = update(
        flux_tt_re,
        at[:, :, :], q_re_sum[:, :, npx.newaxis] * omega_re,
    )
    flux_tt = update(
        flux_tt,
        at[:, :, :], npx.where((flux_tt_re > 0), flux_tt_re + flux_tt, flux_tt),
    )
    flux_tt = update(
        flux_tt,
        at[:, :, :], npx.where((diff_q > 0), flux_tt - diff_q, flux_tt),
    )
    # recalculate travel time distribution
    tt = update(
        tt,
        at[:, :, :], npx.where(npx.any(flux_tt_init > sa[:, :, vs.tau, :], axis=-1)[:, :, npx.newaxis], flux_tt/flux[:, :, npx.newaxis], tt),
    )

    return tt


@roger_kernel
def calc_conc_iso_flux(state, mtt, tt):
    """Calculates isotope signal of hydrologic flux.
    """
    mask = npx.isfinite(mtt)
    vals = allocate(state.dimensions, ("x", "y", "ages"))
    weights = allocate(state.dimensions, ("x", "y", "ages"))
    vals = update(
        vals,
        at[:, :, :], npx.where(mask, mtt, 0),
    )
    weights = update(
        weights,
        at[:, :, :], npx.where(tt * mask > 0, tt / npx.sum(tt * mask, axis=-1)[:, :, npx.newaxis], 0),
    )
    conc = allocate(state.dimensions, ("x", "y"))
    # calculate weighted average
    conc = update(
        conc,
        at[:, :], npx.sum(vals * weights, axis=-1),
    )
    conc = update(
        conc,
        at[:, :], npx.where(conc != 0, conc, npx.NaN),
    )

    return conc


@roger_kernel
def calc_msa_iso(state, sa, msa, flux, tt, mtt):
    """Calculates isotope signal of StorAge.
    """
    vs = state.variables

    flux_in = allocate(state.dimensions, ("x", "y", "ages"))
    sa1 = allocate(state.dimensions, ("x", "y", "ages"))
    msa1 = allocate(state.dimensions, ("x", "y", "ages"))
    msa2 = allocate(state.dimensions, ("x", "y", "ages"))
    msa3 = allocate(state.dimensions, ("x", "y", "ages"))
    flux_in = update(
        flux_in,
        at[:, :, :], flux[:, :, npx.newaxis] * tt,
    )
    sa1 = update(
        sa1,
        at[:, :, :], sa[:, :, vs.tau, :] + flux_in,
    )
    # weighted average isotope ratios of influx and storage
    msa1 = update(
        msa1,
        at[:, :, :], (msa[:, :, vs.tau, :] * (sa[:, :, vs.tau, :]/sa1)) + (mtt * (flux_in/sa1)),
    )
    msa2 = update(
        msa2,
        at[:, :, :], npx.where(npx.isnan(msa1), msa[:, :, vs.tau, :], msa1),
    )
    msa3 = update(
        msa3,
        at[:, :, :], npx.where(npx.isnan(msa2), mtt, msa2),
    )
    msa = update(
        msa,
        at[:, :, vs.tau, :], msa3,
    )
    msa = update(
        msa,
        at[:, :, vs.tau, :], npx.where((msa[:, :, vs.tau, :] != 0), msa[:, :, vs.tau, :], npx.NaN),
    )

    return msa


@roger_kernel
def calc_conc_iso_storage(state, sa, msa):
    """Calculates isotope signal of storage.
    """
    vs = state.variables

    mask = npx.isfinite(msa[:, :, vs.tau, :])
    vals = allocate(state.dimensions, ("x", "y", "ages"))
    weights = allocate(state.dimensions, ("x", "y", "ages"))
    vals = update(
        vals,
        at[:, :, :], npx.where(mask, msa[:, :, vs.tau, :], 0),
    )
    weights = update(
        weights,
        at[:, :, :], npx.where(sa[:, :, vs.tau, :] * mask > 0, sa[:, :, vs.tau, :] / npx.sum(sa[:, :, vs.tau, :] * mask, axis=-1)[:, :, npx.newaxis], 0),
    )
    conc = allocate(state.dimensions, ("x", "y"))
    # calculate weighted average
    conc = update(
        conc,
        at[:, :], npx.sum(vals * weights, axis=-1),
    )
    conc = update(
        conc,
        at[:, :], npx.where(conc != 0, conc, npx.NaN),
    )

    return conc


@roger_kernel
def calc_mtt(state, sa, tt, flux, msa, alpha):
    """Calculate solute travel time distribution at time step t.
    """
    vs = state.variables
    settings = state.settings

    mtt = allocate(state.dimensions, ("x", "y", "ages"))

    if settings.enable_oxygen18 or settings.enable_deuterium:
        # isotope travel time distribution at current time step
        mtt = update(
            mtt,
            at[:, :, :], npx.where(tt > 0, msa[:, :, vs.tau, :], npx.NaN),
        )

    else:
        # solute travel time distribution at current time step
        mask = (tt > 0) & (sa[:, :, vs.tau, :] > 0) & (msa[:, :, vs.tau, :] > 0)
        mtt = update(
            mtt,
            at[:, :, :], npx.where(mask, (msa[:, :, vs.tau, :] / sa[:, :, vs.tau, :]) * alpha[:, :, npx.newaxis] * tt * flux[:, :, npx.newaxis], 0.),
        )

        mask1 = (mtt > msa[:, :, vs.tau, :])
        mtt = update(
            mtt,
            at[:, :, :], npx.where(mask1, msa[:, :, vs.tau, :], mtt),
        )

    return mtt


@roger_kernel
def update_sa(state, sa, tt, flux):
    """Update StorAge (residence time distribution) with outflux at time step t.
    """
    vs = state.variables

    # update StorAge (residence time distribution)
    # remove outflux
    sa = update_add(
        sa,
        at[:, :, vs.tau, :], -flux[:, :, npx.newaxis] * tt,
    )

    # avoid numerical errors
    mask0 = (sa[:, :, vs.tau, :] > -0.001) & (sa[:, :, vs.tau, :] < 0)
    sa = update(
        sa,
        at[:, :, vs.tau, :], npx.where(mask0, 0, sa[:, :, vs.tau, :]),
    )

    return sa


@roger_kernel
def calc_ageing_sa(state, sa):
    """Calculates ageing of StorAge
    """
    vs = state.variables

    sam1 = allocate(state.dimensions, ("x", "y", "ages"))
    sam1 = update(
        sam1,
        at[:, :, :], sa[:, :, vs.tau, :],
    )
    sa = update(
        sa,
        at[:, :, vs.tau, 1:], sam1[:, :, :-1],
    )
    sa = update(
        sa,
        at[:, :, vs.tau, 0], 0,
    )
    # merge oldest water
    sa = update_add(
        sa,
        at[:, :, vs.tau, -1], sam1[:, :, -1],
    )

    return sa


@roger_kernel
def calc_ageing_msa(state, msa):
    """Calculates ageing of solute StorAge
    """
    vs = state.variables

    msam1 = allocate(state.dimensions, ("x", "y", "ages"))
    msam1 = update(
        msam1,
        at[:, :, :], msa[:, :, vs.tau, :],
    )
    msa = update(
        msa,
        at[:, :, vs.tau, 1:], msam1[:, :, :-1],
    )
    msa = update(
        msa,
        at[:, :, vs.tau, 0], 0,
    )
    # merge oldest water
    msa = update_add(
        msa,
        at[:, :, vs.tau, -1], msam1[:, :, -1],
    )

    return msa


@roger_kernel
def calc_ageing_iso(state, sa, msa):
    """Calculates ageing of StorAge
    """
    vs = state.variables

    sam1 = allocate(state.dimensions, ("x", "y", "ages"))
    sam1 = update(
        sam1,
        at[:, :, :], sa[:, :, vs.tau, :],
    )
    sa = update(
        sa,
        at[:, :, vs.tau, 1:], sam1[:, :, :-1],
    )
    sa = update(
        sa,
        at[:, :, vs.tau, 0], 0,
    )
    # isotope ageing
    msam1 = allocate(state.dimensions, ("x", "y", "ages"))
    msam1 = update(
        msam1,
        at[:, :, :], msa[:, :, vs.tau, :],
    )
    msa = update(
        msa,
        at[:, :, vs.tau, 1:], msam1[:, :, :-1],
    )
    # add youngest isotope input to isotope StorAge
    msa = update(
        msa,
        at[:, :, vs.tau, 0], npx.NaN,
    )
    # merge oldest isotopes
    sum_old = allocate(state.dimensions, ("x", "y"))
    iso_old1 = allocate(state.dimensions, ("x", "y"))
    iso_old2 = allocate(state.dimensions, ("x", "y"))
    sum_old = update(
        sum_old,
        at[:, :], sam1[:, :, -1] + sa[:, :, vs.tau, -1],
    )
    iso_old1 = update(
        iso_old1,
        at[:, :], (msam1[:, :, -1] * (sam1[:, :, -1]/sum_old)) + (msa[:, :, vs.tau, -1] * (sa[:, :, vs.tau, -1]/sum_old)),
    )
    iso_old2 = update(
        iso_old2,
        at[:, :], npx.where(npx.isnan(iso_old1), npx.where(npx.isnan(msam1[:, :, -1]), msa[:, :, vs.tau, -1], msam1[:, :, -1]), iso_old1),
    )
    msa = update(
        msa,
        at[:, :, vs.tau, -1], iso_old2,
    )
    # merge oldest water
    sa = update_add(
        sa,
        at[:, :, vs.tau, -1], sam1[:, :, -1],
    )

    return sa, msa


@roger_kernel
def calculate_ageing_sa_kernel(state):
    """Calculates ageing of solute StorAge
    """
    vs = state.variables

    vs.sa_rz = update(
        vs.sa_rz,
        at[:, :, :, :], calc_ageing_sa(state, vs.sa_rz),
    )

    vs.sa_ss = update(
        vs.sa_ss,
        at[:, :, :, :], calc_ageing_sa(state, vs.sa_ss),
    )

    return KernelOutput(sa_rz=vs.sa_rz, sa_ss=vs.sa_ss)


@roger_kernel
def calculate_ageing_msa_kernel(state):
    """Calculates ageing of solute StorAge
    """
    vs = state.variables

    vs.msa_rz = update(
        vs.msa_rz,
        at[:, :, :, :], calc_ageing_msa(state, vs.msa_rz),
    )

    vs.msa_ss = update(
        vs.msa_ss,
        at[:, :, :, :], calc_ageing_msa(state, vs.msa_ss),
    )

    return KernelOutput(msa_rz=vs.msa_rz, msa_ss=vs.msa_ss)


@roger_kernel
def calculate_ageing_iso_kernel(state):
    """Calculates ageing of solute StorAge
    """
    vs = state.variables

    sa_rz, msa_rz = calc_ageing_iso(state, vs.sa_rz, vs.msa_rz)

    vs.sa_rz = update(
        vs.sa_rz,
        at[:, :, vs.tau, :], sa_rz[:, :, vs.tau, :],
    )

    vs.msa_rz = update(
        vs.msa_rz,
        at[:, :, vs.tau, :], msa_rz[:, :, vs.tau, :],
    )

    sa_ss, msa_ss = calc_ageing_iso(state, vs.sa_ss, vs.msa_ss)

    vs.sa_ss = update(
        vs.sa_ss,
        at[:, :, vs.tau, :], sa_ss[:, :, vs.tau, :],
    )

    vs.msa_ss = update(
        vs.msa_ss,
        at[:, :, vs.tau, :], msa_ss[:, :, vs.tau, :],
    )

    return KernelOutput(sa_rz=vs.sa_rz, sa_ss=vs.sa_ss, msa_rz=vs.msa_rz, msa_ss=vs.msa_ss)


@roger_kernel
def calculate_ageing_Nmin_kernel(state):
    """Calculates ageing of mineral nitrogen StorAge
    """
    vs = state.variables

    vs.Nmin_rz = update(
        vs.Nmin_rz,
        at[:, :, :, :], calc_ageing_msa(state, vs.Nmin_rz),
    )

    vs.Nmin_ss = update(
        vs.Nmin_ss,
        at[:, :, :, :], calc_ageing_msa(state, vs.Nmin_ss),
    )

    vs.Nmin_s = update(
        vs.Nmin_s,
        at[:, :, :, :], calc_ageing_msa(state, vs.Nmin_s),
    )

    return KernelOutput(Nmin_rz=vs.Nmin_rz, Nmin_ss=vs.Nmin_ss, Nmin_s=vs.Nmin_s)


@roger_routine
def calculate_ageing(state):
    """
    Calculates ageing of StorAge
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate):
        vs.update(calculate_ageing_sa_kernel(state))
        if settings.enable_groundwater:
            pass

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(calculate_ageing_iso_kernel(state))
        if settings.enable_groundwater:
            pass

    if settings.enable_offline_transport and (settings.enable_chloride | settings.enable_bromide):
        vs.update(calculate_ageing_sa_kernel(state))
        vs.update(calculate_ageing_msa_kernel(state))
        if settings.enable_groundwater:
            pass

    if settings.enable_offline_transport and settings.enable_nitrate:
        vs.update(calculate_ageing_sa_kernel(state))
        vs.update(calculate_ageing_msa_kernel(state))
        vs.update(calculate_ageing_Nmin_kernel(state))
        if settings.enable_groundwater:
            pass
