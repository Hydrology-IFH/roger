from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at
from roger.core import sas, infiltration, evapotranspiration, subsurface_runoff, capillary_rise, crop, root_zone, subsoil, soil, numerics, nitrate
from roger import runtime_settings as rs, logger
from roger import diagnostics


@roger_kernel
def delta_to_conc(state, delta_iso):
    """Calculate isotope concentration from isotope ratio
    """
    settings = state.settings
    if settings.enable_oxygen18:
        conc = settings.VSMOW_conc18O*(delta_iso/1000.+1.)/(1.+(delta_iso/1000.+1.)*settings.VSMOW_conc18O)
    elif settings.enable_deuterium:
        conc = settings.VSMOW_conc2H*(delta_iso/1000.+1.)/(1.+(delta_iso/1000.+1.)*settings.VSMOW_conc2H)

    return conc


@roger_kernel
def conc_to_delta(state, conc):
    """Calculate isotope ratio from isotope concentration
    """
    settings = state.settings
    if settings.enable_oxygen18:
        delta_iso = 1000.*(conc/(settings.VSMOW_conc18O*(1.-conc))-1.)
    elif settings.enable_deuterium:
        delta_iso = 1000.*(conc/(settings.VSMOW_conc2H*(1.-conc))-1.)
    delta_iso = npx.where(delta_iso < -999, npx.nan, delta_iso)

    return delta_iso


@roger_kernel
def calc_SA(state, SA, sa):
    """Calculate cumulative StorAge (residence time distribution).
    """
    vs = state.variables

    # cumulative StorAge (residence time distribution)
    SA = update(
        SA,
        at[2:-2, 2:-2, vs.tau, 1:], npx.cumsum(sa[2:-2, 2:-2, vs.tau, :], axis=-1),
    )
    SA = update(
        SA,
        at[2:-2, 2:-2, vs.tau, 0], 0,
    )

    return SA


@roger_kernel
def calc_tt(state, SA, sa, flux, sas_params):
    """Calculates backward travel time distribution. Used within deterministic
    scheme.
    """
    vs = state.variables
    settings = state.settings

    TT = allocate(state.dimensions, ("x", "y", "nages"))
    tt = allocate(state.dimensions, ("x", "y", "ages"))
    # cumulative backward travel time distribution
    TT = update_add(
        TT,
        at[2:-2, 2:-2, :], sas.uniform(state, SA, sas_params)[2:-2, 2:-2, :],
    )
    TT = update_add(
        TT,
        at[2:-2, 2:-2, :], sas.dirac(state, SA, sas_params)[2:-2, 2:-2, :],
    )
    TT = update(
        TT,
        at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 21) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where((flux[2:-2, 2:-2, npx.newaxis] >= SA[2:-2, 2:-2, 1, :]) & (SA[2:-2, 2:-2, 1, :]/flux[2:-2, 2:-2, npx.newaxis] <= 1), SA[2:-2, 2:-2, 1, :]/flux[2:-2, 2:-2, npx.newaxis], 1), TT[2:-2, 2:-2, :]),
    )
    TT = update(
        TT,
        at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 21) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(TT[2:-2, 2:-2, :] > 1, 1, TT[2:-2, 2:-2, :]), TT[2:-2, 2:-2, :]),
    )
    dirac_old = npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 22) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.cumsum(npx.diff(npx.where((npx.cumsum(npx.diff(SA[2:-2, 2:-2, 1, :], axis=-1)[:, :, ::-1]/flux[2:-2, 2:-2, npx.newaxis], axis=-1) <= 1), npx.cumsum(npx.diff(SA[2:-2, 2:-2, 1, :], axis=-1)[:, :, ::-1]/flux[2:-2, 2:-2, npx.newaxis], axis=-1), 1)[...,::-1], axis=-1), axis=-1), 0)
    TT = update(
        TT,
        at[2:-2, 2:-2, 1:-1], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 22) & (flux[2:-2, 2:-2, npx.newaxis] > 0), dirac_old, TT[2:-2, 2:-2, 1:-1]),
    )
    TT = update(
        TT,
        at[2:-2, 2:-2, 0], npx.where((sas_params[2:-2, 2:-2, 0] == 22), 0, TT[2:-2, 2:-2, 0]),
    )
    TT = update(
        TT,
        at[2:-2, 2:-2, -1], npx.where((sas_params[2:-2, 2:-2, 0] == 22), 1, TT[2:-2, 2:-2, -1]),
    )
    TT_kum, sas_params = sas.kumaraswami(state, SA, sas_params)
    TT = update_add(
        TT,
        at[2:-2, 2:-2, :], TT_kum[2:-2, 2:-2, :],
    )
    TT = update_add(
        TT,
        at[2:-2, 2:-2, :], sas.gamma(state, SA, sas_params)[2:-2, 2:-2, :],
    )
    TT = update_add(
        TT,
        at[2:-2, 2:-2, :], sas.exponential(state, SA, sas_params)[2:-2, 2:-2, :],
    )
    TT_pow, sas_params = sas.power(state, SA, sas_params)
    TT = update_add(
        TT,
        at[2:-2, 2:-2, :], TT_pow[2:-2, 2:-2, :],
    )

    # calculate travel time distribution
    tt = update(
        tt,
        at[2:-2, 2:-2, :], npx.where(npx.diff(TT[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(TT[2:-2, 2:-2, :], axis=-1), 0),
    )

    # age-distributed outflux
    ttq = allocate(state.dimensions, ("x", "y", "ages"))
    ttq = update(
        ttq,
        at[2:-2, 2:-2, :], flux[2:-2, 2:-2, npx.newaxis] * tt[2:-2, 2:-2, :],
    )

    if settings.enable_sas_redistribution:
        # preference for redistribution
        mask_old = npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([1, 22, 32, 34, 52, 6, 63, 64])) | (npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([3, 35])) & (sas_params[:, :, 1, npx.newaxis] >= sas_params[:, :, 2, npx.newaxis])) | (npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([6, 63, 64])) & (sas_params[:, :, 1, npx.newaxis] <= 1))
        mask_old = npx.where(npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([3, 35])) & (sas_params[:, :, 1, npx.newaxis] >= sas_params[:, :, 2, npx.newaxis]), True, mask_old)
        mask_old = npx.where(npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([6, 63, 64])) & (sas_params[:, :, 1, npx.newaxis] >= 1), True, mask_old)
        mask_young = npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([21, 31, 33, 4, 41, 51, 6, 63, 64]))
        mask_young = npx.where(npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([3, 35])) & (sas_params[:, :, 1, npx.newaxis] < sas_params[:, :, 2, npx.newaxis]), True, mask_young)
        mask_young = npx.where(npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([3, 35])) & (sas_params[:, :, 1, npx.newaxis] >= sas_params[:, :, 2, npx.newaxis]), False, mask_young)
        mask_young = npx.where(npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([6, 63, 64])) & (sas_params[:, :, 1, npx.newaxis] < 1), True, mask_young)
        mask_young = npx.where(npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([6, 63, 64])) & (sas_params[:, :, 1, npx.newaxis] >= 1), False, mask_young)

        # set travel time distribution to zero if no outflux
        tt = update(
            tt,
            at[2:-2, 2:-2, :], npx.where(ttq[2:-2, 2:-2, :] > 0, tt[2:-2, 2:-2, :], 0),
        )

        TT = update(
            TT,
            at[2:-2, 2:-2, 1:], npx.cumsum(tt[2:-2, 2:-2, :], axis=-1),
        )
        TT = update(
            TT,
            at[2:-2, 2:-2, 0], 0,
        )

        # outflux is greater than StorAge
        # redistribution of probabilities to guarantee mass conservation
        ttq_init = allocate(state.dimensions, ("x", "y", "ages"))
        diff_q = allocate(state.dimensions, ("x", "y", "ages"))
        q_re = allocate(state.dimensions, ("x", "y", "ages"))
        q_re_sum = allocate(state.dimensions, ("x", "y"))
        sa_free = allocate(state.dimensions, ("x", "y", "ages"))
        sa_free_norm = allocate(state.dimensions, ("x", "y", "ages"))
        SA_re = allocate(state.dimensions, ("x", "y", "nages"))
        omega_re = allocate(state.dimensions, ("x", "y", "ages"))
        Omega_re = allocate(state.dimensions, ("x", "y", "nages"))
        ttq_re = allocate(state.dimensions, ("x", "y", "ages"))
        # outflux is greater than StorAge
        # difference between outflux and StorAge
        ttq_init = update(
            ttq_init,
            at[2:-2, 2:-2, :], ttq[2:-2, 2:-2, :],
        )
        diff_q = update(
            diff_q,
            at[2:-2, 2:-2, :], ttq[2:-2, 2:-2, :] - sa[2:-2, 2:-2, vs.tau, :],
        )
        q_re = update(
            q_re,
            at[2:-2, 2:-2, :], npx.where((diff_q[2:-2, 2:-2, :] > 0), diff_q[2:-2, 2:-2, :], 0),
        )
        q_re_sum = update(
            q_re_sum,
            at[2:-2, 2:-2], npx.sum(q_re[2:-2, 2:-2, :], axis=-1),
        )
        # available StorAge for sampling
        sa_free = update(
            sa_free,
            at[2:-2, 2:-2, :], npx.where((sa[2:-2, 2:-2, vs.tau, :] - ttq[2:-2, 2:-2, :] > 0), sa[2:-2, 2:-2, vs.tau, :] - ttq[2:-2, 2:-2, :], 0),
        )
        # normalize StorAge with outflux of age T
        sa_free_norm = update(
            sa_free_norm,
            at[2:-2, 2:-2, :], npx.where((q_re_sum[2:-2, 2:-2, npx.newaxis] > 0), sa_free[2:-2, 2:-2, :] / q_re_sum[2:-2, 2:-2, npx.newaxis], 0),
        )
        SA_re = update(
            SA_re,
            at[2:-2, 2:-2, :-1], npx.where(mask_old[2:-2, 2:-2, :], npx.cumsum(sa_free_norm[2:-2, 2:-2, ::-1], axis=-1)[:, :, ::-1], SA_re[2:-2, 2:-2, :-1]),
        )
        SA_re = update(
            SA_re,
            at[2:-2, 2:-2, 1:], npx.where(mask_young[2:-2, 2:-2, :], npx.cumsum(sa_free_norm[2:-2, 2:-2, :], axis=-1), SA_re[2:-2, 2:-2, 1:]),
        )
        # cumulative probability to redistribute outflux
        Omega_re = update(
            Omega_re,
            at[2:-2, 2:-2, :], npx.where(SA_re[2:-2, 2:-2, :] < 1, SA_re[2:-2, 2:-2, :], 1),
        )
        # probability to redistribute outflux
        omega_re = update(
            omega_re,
            at[2:-2, 2:-2, :], npx.where(mask_old[2:-2, 2:-2, :], npx.diff(Omega_re[2:-2, 2:-2, ::-1], axis=-1)[:, :, ::-1], omega_re[2:-2, 2:-2, :]),
        )
        omega_re = update(
            omega_re,
            at[2:-2, 2:-2, :], npx.where(mask_young[2:-2, 2:-2, :], npx.diff(Omega_re[2:-2, 2:-2, :], axis=-1), omega_re[2:-2, 2:-2, :]),
        )
        # redistribute outflux
        ttq_re = update(
            ttq_re,
            at[2:-2, 2:-2, :], q_re_sum[2:-2, 2:-2, npx.newaxis] * omega_re[2:-2, 2:-2, :],
        )
        ttq = update(
            ttq,
            at[2:-2, 2:-2, :], npx.where((ttq_re[2:-2, 2:-2, :] > 0), ttq_re[2:-2, 2:-2, :] + ttq[2:-2, 2:-2, :], ttq[2:-2, 2:-2, :]),
        )
        ttq = update(
            ttq,
            at[2:-2, 2:-2, :], npx.where((diff_q[2:-2, 2:-2, :] > 0), ttq[2:-2, 2:-2, :] - diff_q[2:-2, 2:-2, :], ttq[2:-2, 2:-2, :]),
        )
        # recalculate travel time distribution
        tt = update(
            tt,
            at[2:-2, 2:-2, :], npx.where(npx.any(ttq_init[2:-2, 2:-2, :] > sa[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis], ttq[2:-2, 2:-2, :]/flux[2:-2, 2:-2, npx.newaxis], tt[2:-2, 2:-2, :]),
        )

    # impose nonnegative constraint of solution
    ttq = update(
        ttq,
        at[2:-2, 2:-2, :], npx.where(flux[2:-2, 2:-2, npx.newaxis] * tt[2:-2, 2:-2, :] > sa[2:-2, 2:-2, vs.tau, :], sa[2:-2, 2:-2, vs.tau, :], flux[2:-2, 2:-2, npx.newaxis] * tt[2:-2, 2:-2, :]),
    )
    tt = update(
        tt,
        at[2:-2, 2:-2, :], npx.where(flux[2:-2, 2:-2, npx.newaxis] > 0, ttq[2:-2, 2:-2, :]/flux[2:-2, 2:-2, npx.newaxis], 0),
    )

    if rs.backend == 'numpy':
        # sanity check of SAS function (works only for numpy backend)
        mask = npx.isclose(npx.sum(tt, axis=-1) * flux, flux, atol=settings.atol)
        if not npx.all(mask[2:-2, 2:-2]):
            if rs.loglevel == 'error':
                logger.warning(f"Solution of SAS function diverged at iteration {vs.itt}")
            else:
                raise RuntimeError(f"Solution of SAS function diverged at iteration {vs.itt}")
        mask2 = (tt >= 0)
        if not npx.all(mask2[2:-2, 2:-2, :]):
            logger.warning(f"Negative probabilities at iteration {vs.itt}")

    return tt


@roger_kernel
def calc_conc_iso_flux(state, mtt, tt, flux):
    """Calculates isotope signal of hydrologic flux.
    """
    conc = allocate(state.dimensions, ("x", "y"))
    conc = update(
        conc,
        at[2:-2, 2:-2], npx.sum(mtt[2:-2, 2:-2, :] * tt[2:-2, 2:-2, :], axis=-1),
    )
    conc = update(
        conc,
        at[2:-2, 2:-2], npx.where(conc[2:-2, 2:-2] != 0, conc[2:-2, 2:-2], npx.nan),
    )
    if rs.backend == 'numpy':
        mask = (npx.isnan(conc[2:-2, 2:-2]) & (flux[2:-2, 2:-2] > 1))
        if mask.any():
            rows = npx.where(mask)[0].tolist()
            cols = npx.where(mask)[1].tolist()
            rowscols = tuple(zip(rows, cols))
            if rowscols:
                logger.warning(f"Solution of isotope ratio diverged at {state.variables.itt}")
                logger.warning(f"Solution of isotope ratio diverged at {rowscols}")

    return conc


@roger_kernel
def calc_conc_iso_storage(state, sa, msa):
    """Calculates isotope signal of storage.
    """
    vs = state.variables
    settings = state.settings

    conc = allocate(state.dimensions, ("x", "y"))
    if settings.enable_oxygen18 or settings.enable_deuterium:
        conc = update(
            conc,
            at[2:-2, 2:-2], npx.where(npx.sum(sa[2:-2, 2:-2, vs.tau, :], axis=-1) > 0, npx.sum(msa[2:-2, 2:-2, vs.tau, :] * sa[2:-2, 2:-2, vs.tau, :], axis=-1) / npx.sum(sa[2:-2, 2:-2, vs.tau, :], axis=-1), 0),
        )

    elif settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate:
        conc = update(
            conc,
            at[2:-2, 2:-2], npx.sum(npx.where(sa[2:-2, 2:-2, vs.tau, :] > 0, (msa[2:-2, 2:-2, vs.tau, :] / sa[2:-2, 2:-2, vs.tau, :]) * (sa[2:-2, 2:-2, vs.tau, :] / npx.sum(sa[2:-2, 2:-2, vs.tau, :], axis=-1)), 0), axis=-1),
        )
        conc = update(
            conc,
            at[2:-2, 2:-2], npx.where(npx.isnan(conc[2:-2, 2:-2]), 0, conc[2:-2, 2:-2]),
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
        # solute travel time distribution at current time step
        mtt = update(
            mtt,
            at[2:-2, 2:-2, :], npx.where(tt[2:-2, 2:-2, :] > 0, msa[2:-2, 2:-2, vs.tau, :], 0),
        )

    # solute travel time distribution at current time step
    elif settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate:
        mtt = update(
            mtt,
            at[2:-2, 2:-2, :], npx.where(sa[2:-2, 2:-2, vs.tau, :] > 0, (msa[2:-2, 2:-2, vs.tau, :] / sa[2:-2, 2:-2, vs.tau, :]), 0) * alpha[2:-2, 2:-2, npx.newaxis] * tt[2:-2, 2:-2, :] * flux[2:-2, 2:-2, npx.newaxis],
        )
        mtt = update(
            mtt,
            at[2:-2, 2:-2, :], npx.where(mtt[2:-2, 2:-2, :] == -0, 0, mtt[2:-2, 2:-2, :]),
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
        at[2:-2, 2:-2, vs.tau, :], -flux[2:-2, 2:-2, npx.newaxis] * tt[2:-2, 2:-2, :],
    )

    # avoid numerical errors
    mask0 = (sa[:, :, vs.tau, :] > -1e-5) & (sa[:, :, vs.tau, :] < 0)
    sa = update(
        sa,
        at[2:-2, 2:-2, vs.tau, :], npx.where(mask0[2:-2, 2:-2, :], 0, sa[2:-2, 2:-2, vs.tau, :]),
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
        at[2:-2, 2:-2, :], sa[2:-2, 2:-2, vs.tau, :],
    )
    sa = update(
        sa,
        at[2:-2, 2:-2, vs.tau, 1:], sam1[2:-2, 2:-2, :-1],
    )
    sa = update(
        sa,
        at[2:-2, 2:-2, vs.tau, 0], 0,
    )
    # merge oldest water
    sa = update_add(
        sa,
        at[2:-2, 2:-2, vs.tau, -1], sam1[2:-2, 2:-2, -1],
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
        at[2:-2, 2:-2, :], msa[2:-2, 2:-2, vs.tau, :],
    )
    msa = update(
        msa,
        at[2:-2, 2:-2, vs.tau, 1:], msam1[2:-2, 2:-2, :-1],
    )
    msa = update(
        msa,
        at[2:-2, 2:-2, vs.tau, 0], 0,
    )
    # merge oldest water
    msa = update_add(
        msa,
        at[2:-2, 2:-2, vs.tau, -1], msam1[2:-2, 2:-2, -1],
    )

    return msa


@roger_kernel
def calc_ageing_msa_iso(state, msa, sa):
    """Calculates ageing of isotope StorAge
    """
    vs = state.variables

    sam1 = allocate(state.dimensions, ("x", "y", "ages"))
    sam1 = update(
        sam1,
        at[2:-2, 2:-2, :], sa[2:-2, 2:-2, vs.tau, :],
    )
    sa = update(
        sa,
        at[2:-2, 2:-2, vs.tau, 1:], sam1[2:-2, 2:-2, :-1],
    )

    msam1 = allocate(state.dimensions, ("x", "y", "ages"))
    msam1 = update(
        msam1,
        at[2:-2, 2:-2, :], msa[2:-2, 2:-2, vs.tau, :],
    )
    msa = update(
        msa,
        at[2:-2, 2:-2, vs.tau, 1:], msam1[2:-2, 2:-2, :-1],
    )
    msa = update(
        msa,
        at[2:-2, 2:-2, vs.tau, 0], 0,
    )
    # merge oldest water
    msa = update(
        msa,
        at[2:-2, 2:-2, vs.tau, -1], npx.where(sa[2:-2, 2:-2, vs.tau, -1] + sam1[2:-2, 2:-2, -1] > 0, msam1[2:-2, 2:-2, -1] * (sam1[2:-2, 2:-2, -1] / (sa[2:-2, 2:-2, vs.tau, -1] + sam1[2:-2, 2:-2, -1])) + msa[2:-2, 2:-2, vs.tau, -1] * (sa[2:-2, 2:-2, vs.tau, -1] / (sa[2:-2, 2:-2, vs.tau, -1] + sam1[2:-2, 2:-2, -1])), 0),
    )
    msa = update(
        msa,
        at[2:-2, 2:-2, vs.tau, -1], npx.where(npx.isnan(msa[2:-2, 2:-2, vs.tau, -1]), 0, msa[2:-2, 2:-2, vs.tau, -1]),
    )

    sa = update(
        sa,
        at[2:-2, 2:-2, vs.tau, 0], 0,
    )
    # merge oldest water
    sa = update_add(
        sa,
        at[2:-2, 2:-2, vs.tau, -1], sam1[2:-2, 2:-2, -1],
    )

    return msa, sa


@roger_kernel
def calculate_ageing_sa_kernel(state):
    """Calculates ageing of solute StorAge
    """
    vs = state.variables

    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, :, :], calc_ageing_sa(state, vs.sa_rz)[2:-2, 2:-2, :, :],
    )

    vs.sa_ss = update(
        vs.sa_ss,
        at[2:-2, 2:-2, :, :], calc_ageing_sa(state, vs.sa_ss)[2:-2, 2:-2, :, :],
    )

    return KernelOutput(sa_rz=vs.sa_rz, sa_ss=vs.sa_ss)


@roger_kernel
def calculate_ageing_msa_kernel(state):
    """Calculates ageing of solute StorAge
    """
    vs = state.variables

    vs.msa_rz = update(
        vs.msa_rz,
        at[2:-2, 2:-2, :, :], calc_ageing_msa(state, vs.msa_rz)[2:-2, 2:-2, :, :],
    )

    vs.msa_ss = update(
        vs.msa_ss,
        at[2:-2, 2:-2, :, :], calc_ageing_msa(state, vs.msa_ss)[2:-2, 2:-2, :, :],
    )

    return KernelOutput(msa_rz=vs.msa_rz, msa_ss=vs.msa_ss)


@roger_kernel
def calculate_ageing_sa_msa_iso_kernel(state):
    """Calculates ageing of solute StorAge
    """
    vs = state.variables

    msa_rz, sa_rz = calc_ageing_msa_iso(state, vs.msa_rz, vs.sa_rz)
    msa_ss, sa_ss = calc_ageing_msa_iso(state, vs.msa_ss, vs.sa_ss)
    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, :, :], sa_rz[2:-2, 2:-2, :, :],
    )
    vs.msa_rz = update(
        vs.msa_rz,
        at[2:-2, 2:-2, :, :], msa_rz[2:-2, 2:-2, :, :],
    )
    vs.sa_ss = update(
        vs.sa_ss,
        at[2:-2, 2:-2, :, :], sa_ss[2:-2, 2:-2, :, :],
    )
    vs.msa_ss = update(
        vs.msa_ss,
        at[2:-2, 2:-2, :, :], msa_ss[2:-2, 2:-2, :, :],
    )

    return KernelOutput(msa_rz=vs.msa_rz, msa_ss=vs.msa_ss)


@roger_kernel
def calculate_ageing_Nmin_kernel(state):
    """Calculates ageing of mineral nitrogen StorAge
    """
    vs = state.variables

    vs.Nmin_rz = update(
        vs.Nmin_rz,
        at[2:-2, 2:-2, :, :], calc_ageing_msa(state, vs.Nmin_rz)[2:-2, 2:-2, :, :],
    )

    vs.Nmin_ss = update(
        vs.Nmin_ss,
        at[2:-2, 2:-2, :, :], calc_ageing_msa(state, vs.Nmin_ss)[2:-2, 2:-2, :, :],
    )

    vs.Nmin_s = update(
        vs.Nmin_s,
        at[2:-2, 2:-2, :, :], calc_ageing_msa(state, vs.Nmin_s)[2:-2, 2:-2, :, :],
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

    elif settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(calculate_ageing_sa_msa_iso_kernel(state))
        if settings.enable_groundwater:
            pass

    elif settings.enable_offline_transport and (settings.enable_chloride | settings.enable_bromide):
        vs.update(calculate_ageing_sa_kernel(state))
        vs.update(calculate_ageing_msa_kernel(state))
        if settings.enable_groundwater:
            pass

    elif settings.enable_offline_transport and settings.enable_nitrate:
        vs.update(calculate_ageing_sa_kernel(state))
        vs.update(calculate_ageing_msa_kernel(state))
        vs.update(calculate_ageing_Nmin_kernel(state))
        if settings.enable_groundwater:
            pass


@roger_kernel
def calc_TT_num(state, SA, sas_params, flux):
    """
    Calculates cumulative travel time distribution. Used within numerical
    schemes.
    """
    TTq = allocate(state.dimensions, ("x", "y", "nages"))
    TTq = update_add(
        TTq,
        at[2:-2, 2:-2, :], sas.uniform(state, SA, sas_params)[2:-2, 2:-2, :],
    )
    TTq = update_add(
        TTq,
        at[2:-2, 2:-2, :], sas.dirac(state, SA, sas_params)[2:-2, 2:-2, :],
    )
    TTq = update(
        TTq,
        at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 21) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where((flux[2:-2, 2:-2, npx.newaxis] >= SA[2:-2, 2:-2, 1, :]) & (SA[2:-2, 2:-2, 1, :]/flux[2:-2, 2:-2, npx.newaxis] <= 1), SA[2:-2, 2:-2, 1, :]/flux[2:-2, 2:-2, npx.newaxis], 1), TTq[2:-2, 2:-2, :]),
    )
    TTq = update(
        TTq,
        at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 21) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(TTq[2:-2, 2:-2, :] > 1, 1, TTq[2:-2, 2:-2, :]), TTq[2:-2, 2:-2, :]),
    )
    dirac_old = npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 22) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.cumsum(npx.diff(npx.where((npx.cumsum(npx.diff(SA[2:-2, 2:-2, 1, :], axis=-1)[:, :, ::-1]/flux[2:-2, 2:-2, npx.newaxis], axis=-1) <= 1), npx.cumsum(npx.diff(SA[2:-2, 2:-2, 1, :], axis=-1)[:, :, ::-1]/flux[2:-2, 2:-2, npx.newaxis], axis=-1), 1)[..., ::-1], axis=-1), axis=-1), 0)
    TTq = update(
        TTq,
        at[2:-2, 2:-2, 1:-1], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 22) & (flux[2:-2, 2:-2, npx.newaxis] > 0), dirac_old, TTq[2:-2, 2:-2, 1:-1]),
    )
    TTq = update(
        TTq,
        at[2:-2, 2:-2, 0], npx.where((sas_params[2:-2, 2:-2, 0] == 22), 0, TTq[2:-2, 2:-2, 0]),
    )
    TTq = update(
        TTq,
        at[2:-2, 2:-2, -1], npx.where((sas_params[2:-2, 2:-2, 0] == 22), 1, TTq[2:-2, 2:-2, -1]),
    )
    TTq_kum, sas_params = sas.kumaraswami(state, SA, sas_params)
    TTq = update_add(
        TTq,
        at[2:-2, 2:-2, :], TTq_kum[2:-2, 2:-2, :],
    )
    TTq = update_add(
        TTq,
        at[2:-2, 2:-2, :], sas.gamma(state, SA, sas_params)[2:-2, 2:-2, :],
    )
    TTq = update_add(
        TTq,
        at[2:-2, 2:-2, :], sas.exponential(state, SA, sas_params)[2:-2, 2:-2, :],
    )
    TTq_pow, sas_params = sas.power(state, SA, sas_params)
    TTq = update_add(
        TTq,
        at[2:-2, 2:-2, :], TTq_pow[2:-2, 2:-2, :],
    )
    TTq = update(
        TTq,
        at[2:-2, 2:-2, :], npx.where(flux[2:-2, 2:-2, npx.newaxis] <= 0, 0, TTq[2:-2, 2:-2, :]),
    )

    if rs.backend == 'numpy':
        mask = (npx.sum(npx.where(npx.diff(TTq[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(TTq[2:-2, 2:-2, :], axis=-1), 0), axis=-1) <= 0) & (flux[2:-2, 2:-2] > 0)
        if mask.any():
            rows = npx.where(mask)[0].tolist()
            cols = npx.where(mask)[1].tolist()
            rowscols = tuple(zip(rows, cols))
            if rowscols:
                logger.warning(f"Solution of SAS diverged at {state.variables.itt}")
                logger.warning(f"Solution of SAS diverged at {rowscols}")

    return TTq


@roger_kernel
def calc_TT_num_nonneg(state, SA, TTq, flux):
    """Impose nonnegative solution for cumulative travel time distribution.
    Used within numerical schemes. Cumulated travel time distribution might not
    cumulate to 1.
    """
    sa = allocate(state.dimensions, ("x", "y", "ages"))
    ttq = allocate(state.dimensions, ("x", "y", "ages"))
    ttq_nonneg = allocate(state.dimensions, ("x", "y", "ages"))
    sa = update(
        sa,
        at[2:-2, 2:-2, :], npx.diff(SA[2:-2, 2:-2, 1, :], axis=-1),
    )
    ttq = update(
        ttq,
        at[2:-2, 2:-2, :], npx.diff(TTq[2:-2, 2:-2, :], axis=-1) * flux[2:-2, 2:-2, npx.newaxis],
    )
    ttq_nonneg = update(
        ttq_nonneg,
        at[2:-2, 2:-2, :], npx.where(sa + ttq < 0, -sa, ttq)[2:-2, 2:-2, :],
    )
    ttq_nonneg = update(
        ttq_nonneg,
        at[2:-2, 2:-2, :], npx.where(ttq_nonneg == -0, 0, ttq_nonneg)[2:-2, 2:-2, :],
    )
    ttq_nonneg = update(
        ttq_nonneg,
        at[2:-2, 2:-2, :], npx.where(ttq_nonneg > 0, ttq_nonneg/npx.sum(ttq_nonneg, axis=-1)[:, :, npx.newaxis], 0)[2:-2, 2:-2, :],
    )
    TTq_nonneg = allocate(state.dimensions, ("x", "y", "nages"))
    TTq_nonneg = update(
        TTq_nonneg,
        at[2:-2, 2:-2, 1:], npx.cumsum(ttq_nonneg, axis=-1)[2:-2, 2:-2, :],
    )

    return TTq_nonneg


@roger_routine
def svat_transport_model_deterministic(state):
    """Calculates water transport model with deterministic method (i.e. not numerically)
    """
    vs = state.variables
    settings = state.settings

    with state.timers["infiltration into root zone"]:
        infiltration.calculate_infiltration_rz_transport(state)
    with state.timers["evapotranspiration"]:
        evapotranspiration.calculate_evapotranspiration_transport(state)
    with state.timers["infiltration into subsoil"]:
        infiltration.calculate_infiltration_ss_transport(state)
    with state.timers["subsurface runoff of root zone"]:
        subsurface_runoff.calculate_percolation_rz_transport(state)
    with state.timers["subsurface runoff of subsoil"]:
        subsurface_runoff.calculate_percolation_ss_transport(state)
    with state.timers["capillary rise into root zone"]:
        capillary_rise.calculate_capillary_rise_rz_transport(state)
    if settings.enable_nitrate:
        with state.timers["nitrogen cycle"]:
            nitrate.calculate_nitrogen_cycle(state)
    with state.timers["StorAge"]:
        root_zone.calculate_root_zone_transport(state)
        subsoil.calculate_subsoil_transport(state)
        soil.calculate_soil_transport(state)
    with state.timers["diagnostics"]:
        write_output(state)
    with state.timers["ageing"]:
        calculate_ageing(state)
    if settings.enable_oxygen18 or settings.enable_deuterium:
        vs.update(after_substep_iso(state))
    else:
        vs.update(after_substep_anion(state))


@roger_routine
def svat_lbc_transport_model_deterministic(state):
    """Calculates water transport model with deterministic method (i.e. not numerically)
    """
    vs = state.variables
    settings = state.settings

    with state.timers["infiltration into root zone"]:
        infiltration.calculate_infiltration_rz_transport(state)
    with state.timers["evapotranspiration"]:
        evapotranspiration.calculate_evapotranspiration_transport(state)
    with state.timers["infiltration into subsoil"]:
        infiltration.calculate_infiltration_ss_transport(state)
    with state.timers["subsurface runoff of root zone"]:
        subsurface_runoff.calculate_percolation_rz_transport(state)
    with state.timers["subsurface runoff of subsoil"]:
        subsurface_runoff.calculate_percolation_ss_transport(state)
    with state.timers["capillary rise into root zone"]:
        capillary_rise.calculate_capillary_rise_rz_transport(state)
    with state.timers["capillary rise into subsoil"]:
        capillary_rise.calculate_capillary_rise_ss_transport(state)
    if settings.enable_nitrate:
        with state.timers["nitrogen cycle"]:
            nitrate.calculate_nitrogen_cycle(state)
    with state.timers["StorAge"]:
        root_zone.calculate_root_zone_transport(state)
        subsoil.calculate_subsoil_transport(state)
        soil.calculate_soil_transport(state)
    with state.timers["diagnostics"]:
        write_output(state)
    with state.timers["ageing"]:
        calculate_ageing(state)
    if settings.enable_oxygen18 or settings.enable_deuterium:
        vs.update(after_substep_iso(state))
    else:
        vs.update(after_substep_anion(state))


@roger_routine
def svat_crop_transport_model_deterministic(state):
    """Calculates water transport model with deterministic method (i.e. not numerically)
    """
    vs = state.variables
    settings = state.settings

    with state.timers["redistribution after root growth/harvesting"]:
        crop.calculate_redistribution_transport(state)
    with state.timers["infiltration into root zone"]:
        infiltration.calculate_infiltration_rz_transport(state)
    with state.timers["evapotranspiration"]:
        evapotranspiration.calculate_evapotranspiration_transport(state)
    with state.timers["infiltration into subsoil"]:
        infiltration.calculate_infiltration_ss_transport(state)
    with state.timers["subsurface runoff of root zone"]:
        subsurface_runoff.calculate_percolation_rz_transport(state)
    with state.timers["subsurface runoff of subsoil"]:
        subsurface_runoff.calculate_percolation_ss_transport(state)
    with state.timers["capillary rise into root zone"]:
        capillary_rise.calculate_capillary_rise_rz_transport(state)
    if settings.enable_nitrate:
        with state.timers["nitrogen cycle"]:
            nitrate.calculate_nitrogen_cycle(state)
    with state.timers["StorAge"]:
        root_zone.calculate_root_zone_transport(state)
        subsoil.calculate_subsoil_transport(state)
        soil.calculate_soil_transport(state)
    with state.timers["diagnostics"]:
        write_output(state)
    with state.timers["ageing"]:
        calculate_ageing(state)
    if settings.enable_oxygen18 or settings.enable_deuterium:
        vs.update(after_substep_iso(state))
    else:
        vs.update(after_substep_anion(state))


@roger_kernel
def svat_transport_model_rk4(state):
    """Calculates svat transport model with Runge-Kutta method
    """
    vs = state.variables
    settings = state.settings

    # upper boundary condition
    vs.tt_inf_mat_rz = update(
        vs.tt_inf_mat_rz,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.tt_inf_pf_rz = update(
        vs.tt_inf_pf_rz,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.tt_inf_pf_ss = update(
        vs.tt_inf_pf_ss,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    if settings.enable_deuterium or settings.enable_oxygen18:
        vs.mtt_inf_mat_rz = update(
            vs.mtt_inf_mat_rz,
            at[2:-2, 2:-2, 0], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_pf_rz = update(
            vs.mtt_inf_pf_rz,
            at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_pf_ss = update(
            vs.mtt_inf_pf_ss,
            at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
    else:
        vs.mtt_inf_pf_ss = update(
            vs.mtt_inf_pf_ss,
            at[2:-2, 2:-2, 0], vs.C_in[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_mat_rz = update(
            vs.mtt_inf_mat_rz,
            at[2:-2, 2:-2, 0], vs.C_in[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_pf_rz = update(
            vs.mtt_inf_pf_rz,
            at[2:-2, 2:-2, 0], vs.C_in[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )

    # update StorAge with upper boundary condition
    if settings.enable_oxygen18 or settings.enable_deuterium:
        dsa_rz = (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :]) * settings.h
        dsa_ss = (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :]) * settings.h
        dsa_rz1 = (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :]) * settings.h
        dmsa_rz1 = npx.where(npx.isnan(vs.mtt_inf_mat_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_mat_rz[2:-2, 2:-2, :]) * npx.where(dsa_rz1 > 0, ((vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] * settings.h) / dsa_rz1), 0) + npx.where(npx.isnan(vs.mtt_inf_pf_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_rz[2:-2, 2:-2, :]) * npx.where(dsa_rz1 > 0, ((vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] * settings.h) / dsa_rz1), 0)
        dsa_ss1 = (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :]) * settings.h
        dmsa_ss1 = npx.where(npx.isnan(vs.mtt_inf_pf_ss[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_ss[2:-2, 2:-2, :]) * npx.where(dsa_ss1 > 0, ((vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] * settings.h) / dsa_ss1), 0)
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :] > 0) & (vs.msa_rz[2:-2, 2:-2, 1, :] > 0), vs.msa_rz[2:-2, 2:-2, 1, :] * (vs.sa_rz[2:-2, 2:-2, 1, :] / (dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :])), 0) + npx.where(dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :] > 0, dmsa_rz1 * (dsa_rz1 / (dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :])), 0),
        )
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_rz1 > 0) & (vs.msa_rz[2:-2, 2:-2, 1, :] <= 0), dmsa_rz1, vs.msa_rz[2:-2, 2:-2, 1, :]),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where(dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :] > 0, vs.msa_ss[2:-2, 2:-2, 1, :] * (vs.sa_ss[2:-2, 2:-2, 1, :] / (dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :])), 0) + npx.where(dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :] > 0, dmsa_ss1 * (dsa_ss1 / (dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :])), 0),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_ss1 > 0) & (vs.msa_ss[2:-2, 2:-2, 1, :] <= 0), dmsa_ss1, vs.msa_ss[2:-2, 2:-2, 1, :]),
        )
        vs.sa_rz = update_add(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :], dsa_rz,
        )
        vs.sa_ss = update_add(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :], dsa_ss,
        )
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where(vs.sa_rz[2:-2, 2:-2, 1, :] <= 0, 0, vs.msa_rz[2:-2, 2:-2, 1, :]),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where(vs.sa_ss[2:-2, 2:-2, 1, :] <= 0, 0, vs.msa_ss[2:-2, 2:-2, 1, :]),
        )
    elif settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate:
        dsa_rz = (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :]) * settings.h
        dsa_ss = (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :]) * settings.h
        vs.sa_rz = update_add(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :], dsa_rz,
        )
        vs.sa_ss = update_add(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :], dsa_ss,
        )
        dmsa_rz = npx.where(npx.isnan(vs.mtt_inf_mat_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_mat_rz[2:-2, 2:-2, :]) * settings.h + npx.where(npx.isnan(vs.mtt_inf_pf_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_rz[2:-2, 2:-2, :]) * settings.h
        dmsa_ss = npx.where(npx.isnan(vs.mtt_inf_pf_ss[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_ss[2:-2, 2:-2, :]) * settings.h
        vs.msa_rz = update_add(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], dmsa_rz,
        )
        vs.msa_ss = update_add(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], dmsa_ss,
        )

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :], calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    # arrays to store results of 4 approximation points within substep
    TTrkn_evap_soil = allocate(state.dimensions, ("x", "y", "nages", 4))
    ttrkn_evap_soil = allocate(state.dimensions, ("x", "y", "ages", 4))
    TTrkn_transp = allocate(state.dimensions, ("x", "y", "nages", 4))
    ttrkn_transp = allocate(state.dimensions, ("x", "y", "ages", 4))
    TTrkn_cpr_rz = allocate(state.dimensions, ("x", "y", "nages", 4))
    ttrkn_cpr_rz = allocate(state.dimensions, ("x", "y", "ages", 4))
    TTrkn_q_rz = allocate(state.dimensions, ("x", "y", "nages", 4))
    ttrkn_q_rz = allocate(state.dimensions, ("x", "y", "ages", 4))
    TTrkn_q_ss = allocate(state.dimensions, ("x", "y", "nages", 4))
    ttrkn_q_ss = allocate(state.dimensions, ("x", "y", "ages", 4))
    mttrkn_evap_soil = allocate(state.dimensions, ("x", "y", "ages", 4))
    mttrkn_transp = allocate(state.dimensions, ("x", "y", "ages", 4))
    mttrkn_cpr_rz = allocate(state.dimensions, ("x", "y", "ages", 4))
    mttrkn_q_rz = allocate(state.dimensions, ("x", "y", "ages", 4))
    mttrkn_q_ss = allocate(state.dimensions, ("x", "y", "ages", 4))

    SArkn_rz = allocate(state.dimensions, ("x", "y", "timesteps", "nages"))
    sarkn_rz = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
    msarkn_rz = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
    SArkn_ss = allocate(state.dimensions, ("x", "y", "timesteps", "nages"))
    sarkn_ss = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
    msarkn_ss = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
    SArkn_rz = update(
        SArkn_rz,
        at[2:-2, 2:-2, :, :], vs.SA_rz[2:-2, 2:-2, :, :],
    )
    sarkn_rz = update(
        sarkn_rz,
        at[2:-2, 2:-2, :, :], vs.sa_rz[2:-2, 2:-2, :, :],
    )
    msarkn_rz = update(
        msarkn_rz,
        at[2:-2, 2:-2, :, :], vs.msa_rz[2:-2, 2:-2, :, :],
    )
    SArkn_ss = update(
        SArkn_ss,
        at[2:-2, 2:-2, :, :], vs.SA_ss[2:-2, 2:-2, :, :],
    )
    sarkn_ss = update(
        sarkn_ss,
        at[2:-2, 2:-2, :, :], vs.sa_ss[2:-2, 2:-2, :, :],
    )
    msarkn_ss = update(
        msarkn_ss,
        at[2:-2, 2:-2, :, :], vs.msa_ss[2:-2, 2:-2, :, :],
    )

    # step 1 (first approximation point)
    TTrkn_evap_soil = update(
        TTrkn_evap_soil,
        at[2:-2, 2:-2, :, 0], calc_TT_num(state, SArkn_rz, vs.sas_params_evap_soil, vs.evap_soil * settings.h)[2:-2, 2:-2, :],
    )
    TTrkn_evap_soil = update(
        TTrkn_evap_soil,
        at[2:-2, 2:-2, :, 0], calc_TT_num_nonneg(state, SArkn_rz, TTrkn_evap_soil[..., 0], vs.evap_soil * settings.h)[2:-2, 2:-2, :],
    )
    ttrkn_evap_soil = update(
        ttrkn_evap_soil,
        at[2:-2, 2:-2, :, 0], npx.where(npx.diff(TTrkn_evap_soil[2:-2, 2:-2, :, 0], axis=-1) >= 0, npx.diff(TTrkn_evap_soil[2:-2, 2:-2, :, 0], axis=-1), 0),
    )
    if settings.enable_oxygen18 or settings.enable_deuterium:
        mttrkn_evap_soil = update(
            mttrkn_evap_soil,
            at[2:-2, 2:-2, :, 0], calc_mtt(state, sarkn_rz, ttrkn_evap_soil[:, :, :, 0], vs.evap_soil * settings.h, msarkn_rz, vs.alpha_q)[2:-2, 2:-2, :],
        )
    TTrkn_transp = update(
        TTrkn_transp,
        at[2:-2, 2:-2, :, 0], calc_TT_num(state, SArkn_rz, vs.sas_params_transp, vs.transp * settings.h)[2:-2, 2:-2, :],
    )
    TTrkn_transp = update(
        TTrkn_transp,
        at[2:-2, 2:-2, :, 0], calc_TT_num_nonneg(state, SArkn_rz, TTrkn_transp[..., 0], vs.transp * settings.h)[2:-2, 2:-2, :],
    )
    ttrkn_transp = update(
        ttrkn_transp,
        at[2:-2, 2:-2, :, 0], npx.where(npx.diff(TTrkn_transp[2:-2, 2:-2, :, 0], axis=-1) >= 0, npx.diff(TTrkn_transp[2:-2, 2:-2, :, 0], axis=-1), 0),
    )
    mttrkn_transp = update(
        mttrkn_transp,
        at[2:-2, 2:-2, :, 0], calc_mtt(state, sarkn_rz, ttrkn_transp[:, :, :, 0], vs.transp * settings.h, msarkn_rz, vs.alpha_transp)[2:-2, 2:-2, :],
    )
    TTrkn_q_rz = update(
        TTrkn_q_rz,
        at[2:-2, 2:-2, :, 0], calc_TT_num(state, SArkn_rz, vs.sas_params_q_rz, vs.q_rz * settings.h)[2:-2, 2:-2, :],
    )
    TTrkn_q_rz = update(
        TTrkn_q_rz,
        at[2:-2, 2:-2, :, 0], calc_TT_num_nonneg(state, SArkn_rz, TTrkn_q_rz[..., 0], vs.q_rz * settings.h)[2:-2, 2:-2, :],
    )
    ttrkn_q_rz = update(
        ttrkn_q_rz,
        at[2:-2, 2:-2, :, 0], npx.where(npx.diff(TTrkn_q_rz[2:-2, 2:-2, :, 0], axis=-1) >= 0, npx.diff(TTrkn_q_rz[2:-2, 2:-2, :, 0], axis=-1), 0),
    )
    mttrkn_q_rz = update(
        mttrkn_q_rz,
        at[2:-2, 2:-2, :, 0], calc_mtt(state, sarkn_rz, ttrkn_q_rz[:, :, :, 0], vs.q_rz * settings.h, msarkn_rz, vs.alpha_q)[2:-2, 2:-2, :],
    )
    TTrkn_cpr_rz = update(
        TTrkn_cpr_rz,
        at[2:-2, 2:-2, :, 0], calc_TT_num(state, SArkn_ss, vs.sas_params_cpr_rz, vs.cpr_rz * settings.h)[2:-2, 2:-2, :],
    )
    TTrkn_cpr_rz = update(
        TTrkn_cpr_rz,
        at[2:-2, 2:-2, :, 0], calc_TT_num_nonneg(state, SArkn_ss, TTrkn_cpr_rz[..., 0], vs.cpr_rz * settings.h)[2:-2, 2:-2, :],
    )
    ttrkn_cpr_rz = update(
        ttrkn_cpr_rz,
        at[2:-2, 2:-2, :, 0], npx.where(npx.diff(TTrkn_cpr_rz[2:-2, 2:-2, :, 0], axis=-1) >= 0, npx.diff(TTrkn_cpr_rz[2:-2, 2:-2, :, 0], axis=-1), 0),
    )
    mttrkn_cpr_rz = update(
        mttrkn_cpr_rz,
        at[2:-2, 2:-2, :, 0], calc_mtt(state, sarkn_ss, ttrkn_cpr_rz[:, :, :, 0], vs.cpr_rz * settings.h, msarkn_ss, vs.alpha_q)[2:-2, 2:-2, :],
    )
    TTrkn_q_ss = update(
        TTrkn_q_ss,
        at[2:-2, 2:-2, :, 0], calc_TT_num(state, SArkn_ss, vs.sas_params_q_ss, vs.q_ss * settings.h)[2:-2, 2:-2, :],
    )
    TTrkn_q_ss = update(
        TTrkn_q_ss,
        at[2:-2, 2:-2, :, 0], calc_TT_num_nonneg(state, SArkn_ss, TTrkn_q_ss[..., 0], vs.q_ss * settings.h)[2:-2, 2:-2, :],
    )
    ttrkn_q_ss = update(
        ttrkn_q_ss,
        at[2:-2, 2:-2, :, 0], npx.where(npx.diff(TTrkn_q_ss[2:-2, 2:-2, :, 0], axis=-1) >= 0, npx.diff(TTrkn_q_ss[2:-2, 2:-2, :, 0], axis=-1), 0),
    )
    mttrkn_q_ss = update(
        mttrkn_q_ss,
        at[2:-2, 2:-2, :, 0], calc_mtt(state, sarkn_ss, ttrkn_q_ss[:, :, :, 0], vs.q_ss * settings.h, msarkn_ss, vs.alpha_q)[2:-2, 2:-2, :],
    )

    # update StorAge
    if settings.enable_oxygen18 or settings.enable_deuterium:
        # impose nonegative constrain to numerical solution
        dsarkn_rz = (vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 0] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * ttrkn_evap_soil[2:-2, 2:-2, :, 0] - vs.transp[2:-2, 2:-2, npx.newaxis] * ttrkn_transp[2:-2, 2:-2, :, 0] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 0]) * settings.h
        dsarkn_rz = npx.where(sarkn_rz[2:-2, 2:-2, 1, :] + dsarkn_rz < 0, -sarkn_rz[2:-2, 2:-2, 1, :], dsarkn_rz)
        dsarkn_ss = (vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 0] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 0] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * ttrkn_q_ss[2:-2, 2:-2, :, 0]) * settings.h
        dsarkn_ss = npx.where(sarkn_ss[2:-2, 2:-2, 1, :] + dsarkn_ss < 0, -sarkn_ss[2:-2, 2:-2, 1, :], dsarkn_ss)
        dsarkn_rz1 = (vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 0]) * settings.h
        dmsarkn_rz1 = npx.where(npx.isnan(mttrkn_cpr_rz[2:-2, 2:-2, :, 0]), 0, mttrkn_cpr_rz[2:-2, 2:-2, :, 0]) * npx.where(dsarkn_rz1 > 0, ((vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 0] * settings.h) / dsarkn_rz1), 0)
        dsarkn_ss1 = (ttrkn_q_rz[2:-2, 2:-2, :, 0]) * settings.h
        dmsarkn_ss1 = npx.where(npx.isnan(mttrkn_q_rz[2:-2, 2:-2, :, 0]), 0, mttrkn_q_rz[2:-2, 2:-2, :, 0]) * npx.where(dsarkn_ss1 > 0, ((vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 0] * settings.h) / dsarkn_ss1), 0)
        msarkn_rz = update(
            msarkn_rz,
            at[2:-2, 2:-2, 1, :], npx.where((dsarkn_rz1 + sarkn_rz[2:-2, 2:-2, 1, :] > 0) & (msarkn_rz[2:-2, 2:-2, 1, :] > 0), msarkn_rz[2:-2, 2:-2, 1, :] * (sarkn_rz[2:-2, 2:-2, 1, :] / (dsarkn_rz1 + sarkn_rz[2:-2, 2:-2, 1, :])), 0) + npx.where(dsarkn_rz1 + sarkn_rz[2:-2, 2:-2, 1, :] > 0, dmsarkn_rz1 * (dsarkn_rz1 / (dsarkn_rz1 + sarkn_rz[2:-2, 2:-2, 1, :])), 0),
        )
        msarkn_rz = update(
            msarkn_rz,
            at[2:-2, 2:-2, 1, :], npx.where((dsarkn_rz1 > 0) & (msarkn_rz[2:-2, 2:-2, 1, :] <= 0), dmsarkn_rz1, msarkn_rz[2:-2, 2:-2, 1, :]),
        )
        msarkn_ss = update(
            msarkn_ss,
            at[2:-2, 2:-2, 1, :], npx.where(dsarkn_ss1 + sarkn_ss[2:-2, 2:-2, 1, :] > 0, msarkn_ss[2:-2, 2:-2, 1, :] * (sarkn_ss[2:-2, 2:-2, 1, :] / (dsarkn_ss1 + sarkn_ss[2:-2, 2:-2, 1, :])), 0) + npx.where(dsarkn_ss1 + sarkn_ss[2:-2, 2:-2, 1, :] > 0, dmsarkn_ss1 * (dsarkn_ss1 / (dsarkn_ss1 + sarkn_ss[2:-2, 2:-2, 1, :])), 0),
        )
        msarkn_ss = update(
            msarkn_ss,
            at[2:-2, 2:-2, 1, :], npx.where((dsarkn_ss1 > 0) & (msarkn_ss[2:-2, 2:-2, 1, :] <= 0), dmsarkn_ss1, msarkn_ss[2:-2, 2:-2, 1, :]),
        )
        sarkn_rz = update_add(
            sarkn_rz,
            at[2:-2, 2:-2, 1, :], dsarkn_rz,
        )
        sarkn_ss = update_add(
            sarkn_ss,
            at[2:-2, 2:-2, 1, :], dsarkn_ss,
        )
        msarkn_rz = update(
            msarkn_rz,
            at[2:-2, 2:-2, 1, :], npx.where(sarkn_rz[2:-2, 2:-2, 1, :] <= 0, 0, msarkn_rz[2:-2, 2:-2, 1, :]),
        )
        msarkn_ss = update(
            msarkn_ss,
            at[2:-2, 2:-2, 1, :], npx.where(sarkn_ss[2:-2, 2:-2, 1, :] <= 0, 0, msarkn_ss[2:-2, 2:-2, 1, :]),
        )

    elif settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate:
        # impose nonegative constrain to numerical solution
        dsarkn_rz = (vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 0] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * ttrkn_evap_soil[2:-2, 2:-2, :, 0] - vs.transp[2:-2, 2:-2, npx.newaxis] * ttrkn_transp[2:-2, 2:-2, :, 0] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 0]) * settings.h/2
        dsarkn_rz = npx.where(sarkn_rz[2:-2, 2:-2, 1, :] + dsarkn_rz < 0, -sarkn_rz[2:-2, 2:-2, 1, :], dsarkn_rz)
        dsarkn_ss = (vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 0] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 0] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * ttrkn_q_ss[2:-2, 2:-2, :, 0]) * settings.h/2
        dsarkn_ss = npx.where(sarkn_ss[2:-2, 2:-2, 1, :] + dsarkn_ss < 0, -sarkn_ss[2:-2, 2:-2, 1, :], dsarkn_ss)
        sarkn_rz = update_add(
            sarkn_rz,
            at[2:-2, 2:-2, 1, :], dsarkn_rz,
        )
        sarkn_ss = update_add(
            sarkn_ss,
            at[2:-2, 2:-2, 1, :], dsarkn_ss,
        )
        dmsarkn_rz = (npx.where(npx.isnan(mttrkn_cpr_rz[2:-2, 2:-2, :, 0]), 0, mttrkn_cpr_rz[2:-2, 2:-2, :, 0]) - npx.where(npx.isnan(mttrkn_evap_soil[2:-2, 2:-2, :, 0]), 0, mttrkn_evap_soil[2:-2, 2:-2, :, 0]) - npx.where(npx.isnan(mttrkn_transp[2:-2, 2:-2, :, 0]), 0, mttrkn_transp[2:-2, 2:-2, :, 0]) - npx.where(npx.isnan(mttrkn_q_rz[2:-2, 2:-2, :, 0]), 0, mttrkn_q_rz[2:-2, 2:-2, :, 0]))
        dmsarkn_ss = (npx.where(npx.isnan(mttrkn_q_rz[2:-2, 2:-2, :, 0]), 0, mttrkn_q_rz[2:-2, 2:-2, :, 0]) - npx.where(npx.isnan(mttrkn_cpr_rz[2:-2, 2:-2, :, 0]), 0, mttrkn_cpr_rz[2:-2, 2:-2, :, 0]) - npx.where(npx.isnan(mttrkn_q_ss[2:-2, 2:-2, :, 0]), 0, mttrkn_q_ss[2:-2, 2:-2, :, 0]))
        dmsarkn_rz = npx.where(vs.msa_rz[2:-2, 2:-2, 1, :] + dmsarkn_rz < 0, -msarkn_rz[2:-2, 2:-2, 1, :], dmsarkn_rz)
        dmsarkn_ss = npx.where(vs.msa_ss[2:-2, 2:-2, 1, :] + dmsarkn_ss < 0, -msarkn_ss[2:-2, 2:-2, 1, :], dmsarkn_ss)
        msarkn_rz = update_add(
            msarkn_rz,
            at[2:-2, 2:-2, 1, :], dmsarkn_rz,
        )
        msarkn_ss = update_add(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], dmsarkn_ss,
        )

    SArkn_rz = update(
        SArkn_rz,
        at[2:-2, 2:-2, 1, 1:], npx.cumsum(sarkn_rz[2:-2, 2:-2, 1, :], axis=-1),
    )
    SArkn_ss = update(
        SArkn_ss,
        at[2:-2, 2:-2, 1, 1:], npx.cumsum(sarkn_ss[2:-2, 2:-2, 1, :], axis=-1),
    )

    # step 2 (second approximation point)
    TTrkn_evap_soil = update(
        TTrkn_evap_soil,
        at[2:-2, 2:-2, :, 1], calc_TT_num(state, SArkn_rz, vs.sas_params_evap_soil, vs.evap_soil * settings.h/2)[2:-2, 2:-2, :],
    )
    TTrkn_evap_soil = update(
        TTrkn_evap_soil,
        at[2:-2, 2:-2, :, 1], calc_TT_num_nonneg(state, SArkn_rz, TTrkn_evap_soil[..., 1], vs.evap_soil * settings.h/2)[2:-2, 2:-2, :],
    )
    ttrkn_evap_soil = update(
        ttrkn_evap_soil,
        at[2:-2, 2:-2, :, 1], npx.where(npx.diff(TTrkn_evap_soil[2:-2, 2:-2, :, 1], axis=-1) >= 0, npx.diff(TTrkn_evap_soil[2:-2, 2:-2, :, 1], axis=-1), 0),
    )
    if settings.enable_oxygen18 or settings.enable_deuterium:
        mttrkn_evap_soil = update(
            mttrkn_evap_soil,
            at[2:-2, 2:-2, :, 1], calc_mtt(state, sarkn_rz, ttrkn_evap_soil[:, :, :, 1], vs.evap_soil * settings.h/2, msarkn_rz, vs.alpha_q)[2:-2, 2:-2, :],
        )
    TTrkn_transp = update(
        TTrkn_transp,
        at[2:-2, 2:-2, :, 1], calc_TT_num(state, SArkn_rz, vs.sas_params_transp, vs.transp * settings.h/2)[2:-2, 2:-2, :],
    )
    TTrkn_transp = update(
        TTrkn_transp,
        at[2:-2, 2:-2, :, 1], calc_TT_num_nonneg(state, SArkn_rz, TTrkn_transp[..., 1], vs.transp * settings.h/2)[2:-2, 2:-2, :],
    )
    ttrkn_transp = update(
        ttrkn_transp,
        at[2:-2, 2:-2, :, 1], npx.where(npx.diff(TTrkn_transp[2:-2, 2:-2, :, 1], axis=-1) >= 0, npx.diff(TTrkn_transp[2:-2, 2:-2, :, 1], axis=-1), 0),
    )
    mttrkn_transp = update(
        mttrkn_transp,
        at[2:-2, 2:-2, :, 1], calc_mtt(state, sarkn_rz, ttrkn_transp[:, :, :, 1], vs.transp * settings.h/2, msarkn_rz, vs.alpha_transp)[2:-2, 2:-2, :],
    )
    TTrkn_q_rz = update(
        TTrkn_q_rz,
        at[2:-2, 2:-2, :, 1], calc_TT_num(state, SArkn_rz, vs.sas_params_q_rz, vs.q_rz * settings.h/2)[2:-2, 2:-2, :],
    )
    TTrkn_q_rz = update(
        TTrkn_q_rz,
        at[2:-2, 2:-2, :, 1], calc_TT_num_nonneg(state, SArkn_rz, TTrkn_q_rz[..., 1], vs.q_rz * settings.h/2)[2:-2, 2:-2, :],
    )
    ttrkn_q_rz = update(
        ttrkn_q_rz,
        at[2:-2, 2:-2, :, 1], npx.where(npx.diff(TTrkn_q_rz[2:-2, 2:-2, :, 1], axis=-1) >= 0, npx.diff(TTrkn_q_rz[2:-2, 2:-2, :, 1], axis=-1), 0),
    )
    mttrkn_q_rz = update(
        mttrkn_q_rz,
        at[2:-2, 2:-2, :, 1], calc_mtt(state, sarkn_rz, ttrkn_q_rz[:, :, :, 1], vs.q_rz * settings.h/2, msarkn_rz, vs.alpha_q)[2:-2, 2:-2, :],
    )
    TTrkn_cpr_rz = update(
        TTrkn_cpr_rz,
        at[2:-2, 2:-2, :, 1], calc_TT_num(state, SArkn_ss, vs.sas_params_cpr_rz, vs.cpr_rz * settings.h/2)[2:-2, 2:-2, :],
    )
    TTrkn_cpr_rz = update(
        TTrkn_cpr_rz,
        at[2:-2, 2:-2, :, 1], calc_TT_num_nonneg(state, SArkn_ss, TTrkn_cpr_rz[..., 1], vs.cpr_rz * settings.h/2)[2:-2, 2:-2, :],
    )
    ttrkn_cpr_rz = update(
        ttrkn_cpr_rz,
        at[2:-2, 2:-2, :, 1], npx.where(npx.diff(TTrkn_cpr_rz[2:-2, 2:-2, :, 1], axis=-1) >= 0, npx.diff(TTrkn_cpr_rz[2:-2, 2:-2, :, 1], axis=-1), 0),
    )
    mttrkn_cpr_rz = update(
        mttrkn_cpr_rz,
        at[2:-2, 2:-2, :, 1], calc_mtt(state, sarkn_ss, ttrkn_cpr_rz[:, :, :, 1], vs.cpr_rz * settings.h/2, msarkn_ss, vs.alpha_q)[2:-2, 2:-2, :],
    )
    TTrkn_q_ss = update(
        TTrkn_q_ss,
        at[2:-2, 2:-2, :, 1], calc_TT_num(state, SArkn_ss, vs.sas_params_q_ss, vs.q_ss * settings.h/2)[2:-2, 2:-2, :],
    )
    TTrkn_q_ss = update(
        TTrkn_q_ss,
        at[2:-2, 2:-2, :, 1], calc_TT_num_nonneg(state, SArkn_ss, TTrkn_q_ss[..., 1], vs.q_ss * settings.h/2)[2:-2, 2:-2, :],
    )
    ttrkn_q_ss = update(
        ttrkn_q_ss,
        at[2:-2, 2:-2, :, 1], npx.where(npx.diff(TTrkn_q_ss[2:-2, 2:-2, :, 1], axis=-1) >= 0, npx.diff(TTrkn_q_ss[2:-2, 2:-2, :, 1], axis=-1), 0),
    )
    mttrkn_q_ss = update(
        mttrkn_q_ss,
        at[2:-2, 2:-2, :, 1], calc_mtt(state, sarkn_ss, ttrkn_q_ss[:, :, :, 1], vs.q_ss * settings.h/2, msarkn_ss, vs.alpha_q)[2:-2, 2:-2, :],
    )

    if settings.enable_oxygen18 or settings.enable_deuterium:
        # impose nonegative constrain to numerical solution
        dsarkn_rz = (vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 1] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * ttrkn_evap_soil[2:-2, 2:-2, :, 1] - vs.transp[2:-2, 2:-2, npx.newaxis] * ttrkn_transp[2:-2, 2:-2, :, 1] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 1]) * settings.h/2
        dsarkn_rz = npx.where(sarkn_rz[2:-2, 2:-2, 1, :] + dsarkn_rz < 0, -sarkn_rz[2:-2, 2:-2, 1, :], dsarkn_rz)
        dsarkn_ss = (vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 1] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 1] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * ttrkn_q_ss[2:-2, 2:-2, :, 1]) * settings.h/2
        dsarkn_ss = npx.where(sarkn_ss[2:-2, 2:-2, 1, :] + dsarkn_ss < 0, -sarkn_ss[2:-2, 2:-2, 1, :], dsarkn_ss)
        sarkn_rz = update_add(
            sarkn_rz,
            at[2:-2, 2:-2, 1, :], dsarkn_rz,
        )
        sarkn_ss = update_add(
            sarkn_ss,
            at[2:-2, 2:-2, 1, :], dsarkn_ss,
        )
        dmsarkn_rz1 = (npx.where(npx.isnan(mttrkn_cpr_rz[2:-2, 2:-2, :, 1]), 0, mttrkn_cpr_rz[2:-2, 2:-2, :, 1]))
        dmsarkn_rz2 = (npx.where(npx.isnan(mttrkn_evap_soil[2:-2, 2:-2, :, 1]), 0, mttrkn_evap_soil[2:-2, 2:-2, :, 1]) + npx.where(npx.isnan(mttrkn_transp[2:-2, 2:-2, :, 1]), 0, mttrkn_transp[2:-2, 2:-2, :, 1]) + npx.where(npx.isnan(mttrkn_q_rz[2:-2, 2:-2, :, 1]), 0, mttrkn_q_rz[2:-2, 2:-2, :, 1]))
        dmsarkn_ss1 = (npx.where(npx.isnan(mttrkn_q_rz[2:-2, 2:-2, :, 1]), 0, mttrkn_q_rz[2:-2, 2:-2, :, 1])) * settings.h/2
        dmsarkn_ss2 = (npx.where(npx.isnan(mttrkn_cpr_rz[2:-2, 2:-2, :, 1]), 0, mttrkn_cpr_rz[2:-2, 2:-2, :, 1]) + npx.where(npx.isnan(mttrkn_q_ss[2:-2, 2:-2, :, 1]), 0, mttrkn_q_ss[2:-2, 2:-2, :, 1])) * settings.h/2
        dmsarkn_rz = npx.where((dmsarkn_rz1 < 0) & (dmsarkn_rz2 >= 0), dmsarkn_rz1 + dmsarkn_rz2, dmsarkn_rz1 - dmsarkn_rz2)
        dmsarkn_ss = npx.where((dmsarkn_ss1 < 0) & (dmsarkn_ss2 >= 0), dmsarkn_ss1 + dmsarkn_ss2, dmsarkn_ss1 - dmsarkn_ss2)
        dmsarkn_rz = npx.where((dmsarkn_rz > 0), npx.where(msarkn_rz[2:-2, 2:-2, 1, :] + dmsarkn_rz > 0, -msarkn_rz[2:-2, 2:-2, 1, :], dmsarkn_rz), dmsarkn_rz)
        dmsarkn_ss = npx.where((dmsarkn_ss > 0), npx.where(msarkn_ss[2:-2, 2:-2, 1, :] + dmsarkn_ss > 0, -msarkn_ss[2:-2, 2:-2, 1, :], dmsarkn_ss), dmsarkn_ss)
        msarkn_rz = update_add(
            msarkn_rz,
            at[2:-2, 2:-2, 1, :], dmsarkn_rz,
        )
        msarkn_ss = update_add(
            msarkn_ss,
            at[2:-2, 2:-2, 1, :], dmsarkn_ss,
        )
    elif settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate:
        # impose nonegative constrain to numerical solution
        dsarkn_rz = (vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 1] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * ttrkn_evap_soil[2:-2, 2:-2, :, 1] - vs.transp[2:-2, 2:-2, npx.newaxis] * ttrkn_transp[2:-2, 2:-2, :, 1] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 1]) * settings.h/2
        dsarkn_rz = npx.where(sarkn_rz[2:-2, 2:-2, 1, :] + dsarkn_rz < 0, -sarkn_rz[2:-2, 2:-2, 1, :], dsarkn_rz)
        dsarkn_ss = (vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 1] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 1] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * ttrkn_q_ss[2:-2, 2:-2, :, 1]) * settings.h/2
        dsarkn_ss = npx.where(sarkn_ss[2:-2, 2:-2, 1, :] + dsarkn_ss < 0, -sarkn_ss[2:-2, 2:-2, 1, :], dsarkn_ss)
        sarkn_rz = update_add(
            sarkn_rz,
            at[2:-2, 2:-2, 1, :], dsarkn_rz,
        )
        sarkn_ss = update_add(
            sarkn_ss,
            at[2:-2, 2:-2, 1, :], dsarkn_ss,
        )
        dmsarkn_rz = (npx.where(npx.isnan(mttrkn_cpr_rz[2:-2, 2:-2, :, 1]), 0, mttrkn_cpr_rz[2:-2, 2:-2, :, 1]) - npx.where(npx.isnan(mttrkn_evap_soil[2:-2, 2:-2, :, 1]), 0, mttrkn_evap_soil[2:-2, 2:-2, :, 1]) - npx.where(npx.isnan(mttrkn_transp[2:-2, 2:-2, :, 1]), 0, mttrkn_transp[2:-2, 2:-2, :, 1]) - npx.where(npx.isnan(mttrkn_q_rz[2:-2, 2:-2, :, 1]), 0, mttrkn_q_rz[2:-2, 2:-2, :, 1]))
        dmsarkn_ss = (npx.where(npx.isnan(mttrkn_q_rz[2:-2, 2:-2, :, 1]), 0, mttrkn_q_rz[2:-2, 2:-2, :, 1]) - npx.where(npx.isnan(mttrkn_cpr_rz[2:-2, 2:-2, :, 1]), 0, mttrkn_cpr_rz[2:-2, 2:-2, :, 1]) - npx.where(npx.isnan(mttrkn_q_ss[2:-2, 2:-2, :, 1]), 0, mttrkn_q_ss[2:-2, 2:-2, :, 1]))
        dmsarkn_rz = npx.where(vs.msa_rz[2:-2, 2:-2, 1, :] + dmsarkn_rz < 0, -msarkn_rz[2:-2, 2:-2, 1, :], dmsarkn_rz)
        dmsarkn_ss = npx.where(vs.msa_ss[2:-2, 2:-2, 1, :] + dmsarkn_ss < 0, -msarkn_ss[2:-2, 2:-2, 1, :], dmsarkn_ss)
        msarkn_rz = update_add(
            msarkn_rz,
            at[2:-2, 2:-2, 1, :], dmsarkn_rz,
        )
        msarkn_ss = update_add(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], dmsarkn_ss,
        )
    SArkn_rz = update(
        SArkn_rz,
        at[2:-2, 2:-2, 1, 1:], npx.cumsum(sarkn_rz[2:-2, 2:-2, 1, :], axis=-1),
    )
    SArkn_ss = update(
        SArkn_ss,
        at[2:-2, 2:-2, 1, 1:], npx.cumsum(sarkn_ss[2:-2, 2:-2, 1, :], axis=-1),
    )

    # step 3 (third approximation point)
    TTrkn_evap_soil = update(
        TTrkn_evap_soil,
        at[2:-2, 2:-2, :, 2], calc_TT_num(state, SArkn_rz, vs.sas_params_evap_soil, vs.evap_soil * settings.h/2)[2:-2, 2:-2, :],
    )
    TTrkn_evap_soil = update(
        TTrkn_evap_soil,
        at[2:-2, 2:-2, :, 2], calc_TT_num_nonneg(state, SArkn_rz, TTrkn_evap_soil[..., 2], vs.evap_soil * settings.h/2)[2:-2, 2:-2, :],
    )
    ttrkn_evap_soil = update(
        ttrkn_evap_soil,
        at[2:-2, 2:-2, :, 2], npx.where(npx.diff(TTrkn_evap_soil[2:-2, 2:-2, :, 2], axis=-1) >= 0, npx.diff(TTrkn_evap_soil[2:-2, 2:-2, :, 2], axis=-1), 0),
    )
    if settings.enable_oxygen18 or settings.enable_deuterium:
        mttrkn_evap_soil = update(
            mttrkn_evap_soil,
            at[2:-2, 2:-2, :, 2], calc_mtt(state, sarkn_rz, ttrkn_evap_soil[:, :, :, 2], vs.evap_soil * settings.h/2, msarkn_rz, vs.alpha_q)[2:-2, 2:-2, :],
        )
    TTrkn_transp = update(
        TTrkn_transp,
        at[2:-2, 2:-2, :, 2], calc_TT_num(state, SArkn_rz, vs.sas_params_transp, vs.transp * settings.h/2)[2:-2, 2:-2, :],
    )
    TTrkn_transp = update(
        TTrkn_transp,
        at[2:-2, 2:-2, :, 2], calc_TT_num_nonneg(state, SArkn_rz, TTrkn_transp[..., 2], vs.transp * settings.h/2)[2:-2, 2:-2, :],
    )
    ttrkn_transp = update(
        ttrkn_transp,
        at[2:-2, 2:-2, :, 2], npx.where(npx.diff(TTrkn_transp[2:-2, 2:-2, :, 2], axis=-1) >= 0, npx.diff(TTrkn_transp[2:-2, 2:-2, :, 2], axis=-1), 0),
    )
    mttrkn_transp = update(
        mttrkn_transp,
        at[2:-2, 2:-2, :, 2], calc_mtt(state, sarkn_rz, ttrkn_transp[:, :, :, 2], vs.transp * settings.h/2, msarkn_rz, vs.alpha_transp)[2:-2, 2:-2, :],
    )
    TTrkn_q_rz = update(
        TTrkn_q_rz,
        at[2:-2, 2:-2, :, 2], calc_TT_num(state, SArkn_rz, vs.sas_params_q_rz, vs.q_rz * settings.h/2)[2:-2, 2:-2, :],
    )
    TTrkn_q_rz = update(
        TTrkn_q_rz,
        at[2:-2, 2:-2, :, 2], calc_TT_num_nonneg(state, SArkn_rz, TTrkn_q_rz[..., 2], vs.q_rz * settings.h/2)[2:-2, 2:-2, :],
    )
    ttrkn_q_rz = update(
        ttrkn_q_rz,
        at[2:-2, 2:-2, :, 2], npx.where(npx.diff(TTrkn_q_rz[2:-2, 2:-2, :, 2], axis=-1) >= 0, npx.diff(TTrkn_q_rz[2:-2, 2:-2, :, 2], axis=-1), 0),
    )
    mttrkn_q_rz = update(
        mttrkn_q_rz,
        at[2:-2, 2:-2, :, 2], calc_mtt(state, sarkn_rz, ttrkn_q_rz[:, :, :, 2], vs.q_rz * settings.h/2, msarkn_rz, vs.alpha_q)[2:-2, 2:-2, :],
    )
    TTrkn_cpr_rz = update(
        TTrkn_cpr_rz,
        at[2:-2, 2:-2, :, 2], calc_TT_num(state, SArkn_ss, vs.sas_params_cpr_rz, vs.cpr_rz * settings.h/2)[2:-2, 2:-2, :],
    )
    TTrkn_cpr_rz = update(
        TTrkn_cpr_rz,
        at[2:-2, 2:-2, :, 2], calc_TT_num_nonneg(state, SArkn_ss, TTrkn_cpr_rz[..., 2], vs.cpr_rz * settings.h/2)[2:-2, 2:-2, :],
    )
    ttrkn_cpr_rz = update(
        ttrkn_cpr_rz,
        at[2:-2, 2:-2, :, 2], npx.where(npx.diff(TTrkn_cpr_rz[2:-2, 2:-2, :, 2], axis=-1) >= 0, npx.diff(TTrkn_cpr_rz[2:-2, 2:-2, :, 2], axis=-1), 0),
    )
    mttrkn_cpr_rz = update(
        mttrkn_cpr_rz,
        at[2:-2, 2:-2, :, 2], calc_mtt(state, sarkn_ss, ttrkn_cpr_rz[:, :, :, 2], vs.cpr_rz * settings.h/2, msarkn_ss, vs.alpha_q)[2:-2, 2:-2, :],
    )
    TTrkn_q_ss = update(
        TTrkn_q_ss,
        at[2:-2, 2:-2, :, 2], calc_TT_num(state, SArkn_ss, vs.sas_params_q_ss, vs.q_ss * settings.h/2)[2:-2, 2:-2, :],
    )
    TTrkn_q_ss = update(
        TTrkn_q_ss,
        at[2:-2, 2:-2, :, 2], calc_TT_num_nonneg(state, SArkn_ss, TTrkn_q_ss[..., 2], vs.q_ss * settings.h/2)[2:-2, 2:-2, :],
    )
    ttrkn_q_ss = update(
        ttrkn_q_ss,
        at[2:-2, 2:-2, :, 2], npx.where(npx.diff(TTrkn_q_ss[2:-2, 2:-2, :, 2], axis=-1) >= 0, npx.diff(TTrkn_q_ss[2:-2, 2:-2, :, 2], axis=-1), 0),
    )
    mttrkn_q_ss = update(
        mttrkn_q_ss,
        at[2:-2, 2:-2, :, 2], calc_mtt(state, sarkn_ss, ttrkn_q_ss[:, :, :, 2], vs.q_ss * settings.h/2, msarkn_ss, vs.alpha_q)[2:-2, 2:-2, :],
    )

    if settings.enable_oxygen18 or settings.enable_deuterium:
        # impose nonegative constrain to numerical solution
        dsarkn_rz = (vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 2] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * ttrkn_evap_soil[2:-2, 2:-2, :, 2] - vs.transp[2:-2, 2:-2, npx.newaxis] * ttrkn_transp[2:-2, 2:-2, :, 2] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 2]) * settings.h/2
        dsarkn_rz = npx.where(sarkn_rz[2:-2, 2:-2, 1, :] - dsarkn_rz < 0, -sarkn_rz[2:-2, 2:-2, 1, :], dsarkn_rz)
        dsarkn_ss = (vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 2] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 2] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * ttrkn_q_ss[2:-2, 2:-2, :, 2]) * settings.h/2
        dsarkn_ss = npx.where(sarkn_ss[2:-2, 2:-2, 1, :] - dsarkn_ss < 0, -sarkn_ss[2:-2, 2:-2, 1, :], dsarkn_ss)
        sarkn_rz = update_add(
            sarkn_rz,
            at[2:-2, 2:-2, 1, :], dsarkn_rz,
        )
        sarkn_ss = update_add(
            sarkn_ss,
            at[2:-2, 2:-2, 1, :], dsarkn_ss,
        )
        dmsarkn_rz1 = (npx.where(npx.isnan(mttrkn_cpr_rz[2:-2, 2:-2, :, 2]), 0, mttrkn_cpr_rz[2:-2, 2:-2, :, 2]))
        dmsarkn_rz2 = (npx.where(npx.isnan(mttrkn_evap_soil[2:-2, 2:-2, :, 2]), 0, mttrkn_evap_soil[2:-2, 2:-2, :, 2]) + npx.where(npx.isnan(mttrkn_transp[2:-2, 2:-2, :, 2]), 0, mttrkn_transp[2:-2, 2:-2, :, 2]) + npx.where(npx.isnan(mttrkn_q_rz[2:-2, 2:-2, :, 2]), 0, mttrkn_q_rz[2:-2, 2:-2, :, 2]))
        dmsarkn_ss1 = (npx.where(npx.isnan(mttrkn_q_rz[2:-2, 2:-2, :, 2]), 0, mttrkn_q_rz[2:-2, 2:-2, :, 2]))
        dmsarkn_ss2 = (npx.where(npx.isnan(mttrkn_cpr_rz[2:-2, 2:-2, :, 2]), 0, mttrkn_cpr_rz[2:-2, 2:-2, :, 2]) + npx.where(npx.isnan(mttrkn_q_ss[2:-2, 2:-2, :, 2]), 0, mttrkn_q_ss[2:-2, 2:-2, :, 2]))
        dmsarkn_rz = npx.where((dmsarkn_rz1 < 0) & (dmsarkn_rz2 >= 0), dmsarkn_rz1 + dmsarkn_rz2, dmsarkn_rz1 - dmsarkn_rz2)
        dmsarkn_ss = npx.where((dmsarkn_ss1 < 0) & (dmsarkn_ss2 >= 0), dmsarkn_ss1 + dmsarkn_ss2, dmsarkn_ss1 - dmsarkn_ss2)
        dmsarkn_rz = npx.where((dmsarkn_rz > 0), npx.where(msarkn_rz[2:-2, 2:-2, 1, :] + dmsarkn_rz > 0, -msarkn_rz[2:-2, 2:-2, 1, :], dmsarkn_rz), dmsarkn_rz)
        dmsarkn_ss = npx.where((dmsarkn_ss > 0), npx.where(msarkn_ss[2:-2, 2:-2, 1, :] + dmsarkn_ss > 0, -msarkn_ss[2:-2, 2:-2, 1, :], dmsarkn_ss), dmsarkn_ss)
        msarkn_rz = update_add(
            msarkn_rz,
            at[2:-2, 2:-2, 1, :], dmsarkn_rz,
        )
        msarkn_ss = update_add(
            msarkn_ss,
            at[2:-2, 2:-2, 1, :], dmsarkn_ss,
        )
    elif settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate:
        # impose nonegative constrain to numerical solution
        dmsarkn_rz = (npx.where(npx.isnan(mttrkn_cpr_rz[2:-2, 2:-2, :, 2]), 0, mttrkn_cpr_rz[2:-2, 2:-2, :, 2]) - npx.where(npx.isnan(mttrkn_evap_soil[2:-2, 2:-2, :, 2]), 0, mttrkn_evap_soil[2:-2, 2:-2, :, 2]) - npx.where(npx.isnan(mttrkn_transp[2:-2, 2:-2, :, 2]), 0, mttrkn_transp[2:-2, 2:-2, :, 2]) - npx.where(npx.isnan(mttrkn_q_rz[2:-2, 2:-2, :, 2]), 0, mttrkn_q_rz[2:-2, 2:-2, :, 2]))
        dmsarkn_ss = (npx.where(npx.isnan(mttrkn_q_rz[2:-2, 2:-2, :, 1]), 0, mttrkn_q_rz[2:-2, 2:-2, :, 1]) - npx.where(npx.isnan(mttrkn_cpr_rz[2:-2, 2:-2, :, 1]), 0, mttrkn_cpr_rz[2:-2, 2:-2, :, 1]) - npx.where(npx.isnan(mttrkn_q_ss[2:-2, 2:-2, :, 1]), 0, mttrkn_q_ss[2:-2, 2:-2, :, 1]))
        dmsarkn_rz = npx.where(msarkn_rz[2:-2, 2:-2, 1, :] - dmsarkn_rz < 0, -msarkn_rz[2:-2, 2:-2, 1, :], dmsarkn_rz)
        dmsarkn_ss = npx.where(msarkn_ss[2:-2, 2:-2, 1, :] - dmsarkn_ss < 0, -msarkn_ss[2:-2, 2:-2, 1, :], dmsarkn_ss)
        msarkn_rz = update_add(
            msarkn_rz,
            at[2:-2, 2:-2, 1, :], dmsarkn_rz,
        )
        msarkn_ss = update_add(
            msarkn_ss,
            at[2:-2, 2:-2, 1, :], dmsarkn_ss,
        )
    SArkn_rz = update(
        SArkn_rz,
        at[2:-2, 2:-2, 1, 1:], npx.cumsum(sarkn_rz[2:-2, 2:-2, 1, :], axis=-1),
    )
    SArkn_ss = update(
        SArkn_ss,
        at[2:-2, 2:-2, 1, 1:], npx.cumsum(sarkn_ss[2:-2, 2:-2, 1, :], axis=-1),
    )

    # step 4 (fourth approximation point)
    TTrkn_evap_soil = update(
        TTrkn_evap_soil,
        at[2:-2, 2:-2, :, 3], calc_TT_num(state, SArkn_rz, vs.sas_params_evap_soil, vs.evap_soil * settings.h)[2:-2, 2:-2, :],
    )
    TTrkn_evap_soil = update(
        TTrkn_evap_soil,
        at[2:-2, 2:-2, :, 3], calc_TT_num_nonneg(state, SArkn_rz, TTrkn_evap_soil[..., 3], vs.evap_soil * settings.h)[2:-2, 2:-2, :],
    )
    ttrkn_evap_soil = update(
        ttrkn_evap_soil,
        at[2:-2, 2:-2, :, 3], npx.where(npx.diff(TTrkn_evap_soil[2:-2, 2:-2, :, 3], axis=-1) >= 0, npx.diff(TTrkn_evap_soil[2:-2, 2:-2, :, 3], axis=-1), 0),
    )
    if settings.enable_oxygen18 or settings.enable_deuterium:
        mttrkn_evap_soil = update(
            mttrkn_evap_soil,
            at[2:-2, 2:-2, :, 3], calc_mtt(state, sarkn_rz, ttrkn_evap_soil[:, :, :, 3], vs.evap_soil * settings.h, msarkn_rz, vs.alpha_q)[2:-2, 2:-2, :],
        )
    TTrkn_transp = update(
        TTrkn_transp,
        at[2:-2, 2:-2, :, 3], calc_TT_num(state, SArkn_rz, vs.sas_params_transp, vs.transp * settings.h)[2:-2, 2:-2, :],
    )
    TTrkn_transp = update(
        TTrkn_transp,
        at[2:-2, 2:-2, :, 3], calc_TT_num_nonneg(state, SArkn_rz, TTrkn_transp[..., 3], vs.transp * settings.h)[2:-2, 2:-2, :],
    )
    ttrkn_transp = update(
        ttrkn_transp,
        at[2:-2, 2:-2, :, 3], npx.where(npx.diff(TTrkn_transp[2:-2, 2:-2, :, 3], axis=-1) >= 0, npx.diff(TTrkn_transp[2:-2, 2:-2, :, 3], axis=-1), 0),
    )
    mttrkn_transp = update(
        mttrkn_transp,
        at[2:-2, 2:-2, :, 3], calc_mtt(state, sarkn_rz, ttrkn_transp[:, :, :, 3], vs.transp * settings.h, msarkn_rz, vs.alpha_transp)[2:-2, 2:-2, :],
    )
    TTrkn_q_rz = update(
        TTrkn_q_rz,
        at[2:-2, 2:-2, :, 3], calc_TT_num(state, SArkn_rz, vs.sas_params_q_rz, vs.q_rz * settings.h)[2:-2, 2:-2, :],
    )
    TTrkn_q_rz = update(
        TTrkn_q_rz,
        at[2:-2, 2:-2, :, 3], calc_TT_num_nonneg(state, SArkn_rz, TTrkn_q_rz[..., 3], vs.q_rz * settings.h)[2:-2, 2:-2, :],
    )
    ttrkn_q_rz = update(
        ttrkn_q_rz,
        at[2:-2, 2:-2, :, 3], npx.where(npx.diff(TTrkn_q_rz[2:-2, 2:-2, :, 3], axis=-1) >= 0, npx.diff(TTrkn_q_rz[2:-2, 2:-2, :, 3], axis=-1), 0),
    )
    mttrkn_q_rz = update(
        mttrkn_q_rz,
        at[2:-2, 2:-2, :, 3], calc_mtt(state, sarkn_rz, ttrkn_q_rz[:, :, :, 3], vs.q_rz * settings.h, msarkn_rz, vs.alpha_q)[2:-2, 2:-2, :],
    )
    TTrkn_cpr_rz = update(
        TTrkn_cpr_rz,
        at[2:-2, 2:-2, :, 3], calc_TT_num(state, SArkn_ss, vs.sas_params_cpr_rz, vs.cpr_rz * settings.h)[2:-2, 2:-2, :],
    )
    TTrkn_cpr_rz = update(
        TTrkn_cpr_rz,
        at[2:-2, 2:-2, :, 3], calc_TT_num_nonneg(state, SArkn_ss, TTrkn_cpr_rz[..., 3], vs.cpr_rz * settings.h)[2:-2, 2:-2, :],
    )
    ttrkn_cpr_rz = update(
        ttrkn_cpr_rz,
        at[2:-2, 2:-2, :, 3], npx.where(npx.diff(TTrkn_cpr_rz[2:-2, 2:-2, :, 3], axis=-1) >= 0, npx.diff(TTrkn_cpr_rz[2:-2, 2:-2, :, 3], axis=-1), 0),
    )
    mttrkn_cpr_rz = update(
        mttrkn_cpr_rz,
        at[2:-2, 2:-2, :, 3], calc_mtt(state, sarkn_ss, ttrkn_cpr_rz[:, :, :, 3], vs.cpr_rz * settings.h, msarkn_ss, vs.alpha_q)[2:-2, 2:-2, :],
    )
    TTrkn_q_ss = update(
        TTrkn_q_ss,
        at[2:-2, 2:-2, :, 3], calc_TT_num(state, SArkn_ss, vs.sas_params_q_ss, vs.q_ss * settings.h)[2:-2, 2:-2, :],
    )
    TTrkn_q_ss = update(
        TTrkn_q_ss,
        at[2:-2, 2:-2, :, 3], calc_TT_num_nonneg(state, SArkn_ss, TTrkn_q_ss[..., 3], vs.q_ss * settings.h)[2:-2, 2:-2, :],
    )
    ttrkn_q_ss = update(
        ttrkn_q_ss,
        at[2:-2, 2:-2, :, 3], npx.where(npx.diff(TTrkn_q_ss[2:-2, 2:-2, :, 3], axis=-1) >= 0, npx.diff(TTrkn_q_ss[2:-2, 2:-2, :, 3], axis=-1), 0),
    )
    mttrkn_q_ss = update(
        mttrkn_q_ss,
        at[2:-2, 2:-2, :, 3], calc_mtt(state, sarkn_ss, ttrkn_q_ss[:, :, :, 3], vs.q_ss * settings.h, msarkn_ss, vs.alpha_q)[2:-2, 2:-2, :],
    )

    vs.tt_evap_soil = update(
        vs.tt_evap_soil,
        at[2:-2, 2:-2, :], (ttrkn_evap_soil[2:-2, 2:-2, :, 0] + 2*ttrkn_evap_soil[2:-2, 2:-2, :, 1] + 2*ttrkn_evap_soil[2:-2, 2:-2, :, 2] + ttrkn_evap_soil[2:-2, 2:-2, :, 3]) / 6.,
    )
    vs.tt_transp = update(
        vs.tt_transp,
        at[2:-2, 2:-2, :], (ttrkn_transp[2:-2, 2:-2, :, 0] + 2*ttrkn_transp[2:-2, 2:-2, :, 1] + 2*ttrkn_transp[2:-2, 2:-2, :, 2] + ttrkn_transp[2:-2, 2:-2, :, 3]) / 6.,
    )
    vs.tt_q_rz = update(
        vs.tt_q_rz,
        at[2:-2, 2:-2, :], (ttrkn_q_rz[2:-2, 2:-2, :, 0] + 2*ttrkn_q_rz[2:-2, 2:-2, :, 1] + 2*ttrkn_q_rz[2:-2, 2:-2, :, 2] + ttrkn_q_rz[2:-2, 2:-2, :, 3]) / 6.,
    )
    vs.tt_cpr_rz = update(
        vs.tt_cpr_rz,
        at[2:-2, 2:-2, :], (ttrkn_cpr_rz[2:-2, 2:-2, :, 0] + 2*ttrkn_cpr_rz[2:-2, 2:-2, :, 1] + 2*ttrkn_cpr_rz[2:-2, 2:-2, :, 2] + ttrkn_cpr_rz[2:-2, 2:-2, :, 3]) / 6.,
    )
    vs.tt_q_ss = update(
        vs.tt_q_ss,
        at[2:-2, 2:-2, :], (ttrkn_q_ss[2:-2, 2:-2, :, 0] + 2*ttrkn_q_ss[2:-2, 2:-2, :, 1] + 2*ttrkn_q_ss[2:-2, 2:-2, :, 2] + ttrkn_q_ss[2:-2, 2:-2, :, 3]) / 6.,
    )
    vs.TT_evap_soil = update(
        vs.TT_evap_soil,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_evap_soil, axis=-1)[2:-2, 2:-2, :],
    )
    vs.TT_transp = update(
        vs.TT_transp,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_transp, axis=-1)[2:-2, 2:-2, :],
    )
    vs.TT_q_rz = update(
        vs.TT_q_rz,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_rz, axis=-1)[2:-2, 2:-2, :],
    )
    vs.TT_cpr_rz = update(
        vs.TT_cpr_rz,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_cpr_rz, axis=-1)[2:-2, 2:-2, :],
    )
    vs.TT_q_ss = update(
        vs.TT_q_ss,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_ss, axis=-1)[2:-2, 2:-2, :],
    )

    if settings.enable_oxygen18 or settings.enable_deuterium:
        vs.mtt_evap_soil = update(
            vs.mtt_evap_soil,
            at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_rz, vs.tt_evap_soil, vs.evap_soil * settings.h, vs.msa_rz, vs.alpha_q)[2:-2, 2:-2, :],
        )
    vs.mtt_transp = update(
        vs.mtt_transp,
        at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_rz, vs.tt_transp, vs.transp * settings.h, vs.msa_rz, vs.alpha_transp)[2:-2, 2:-2, :],
    )
    vs.mtt_q_rz = update(
        vs.mtt_q_rz,
        at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_rz, vs.tt_q_rz, vs.q_rz * settings.h, vs.msa_rz, vs.alpha_q)[2:-2, 2:-2, :],
    )
    vs.mtt_cpr_rz = update(
        vs.mtt_cpr_rz,
        at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_ss, vs.tt_cpr_rz, vs.cpr_rz * settings.h, vs.msa_ss, vs.alpha_q)[2:-2, 2:-2, :],
    )
    vs.mtt_q_ss = update(
        vs.mtt_q_ss,
        at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_ss, vs.tt_q_ss, vs.q_ss * settings.h, vs.msa_ss, vs.alpha_q)[2:-2, 2:-2, :],
    )

    # update StorAge
    if settings.enable_oxygen18 or settings.enable_deuterium:
        # impose nonegative constrain to numerical solution
        dsa_rz = (vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :] - vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :]) * settings.h
        dsa_rz = npx.where(vs.sa_rz[2:-2, 2:-2, 1, :] + dsa_rz < 0, -vs.sa_rz[2:-2, 2:-2, 1, :], dsa_rz)
        dsa_ss = (vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :]) * settings.h
        dsa_ss = npx.where(vs.sa_ss[2:-2, 2:-2, 1, :] + dsa_ss < 0, -vs.sa_ss[2:-2, 2:-2, 1, :], dsa_ss)
        dsa_rz1 = (vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :]) * settings.h
        dmsa_rz1 = npx.where(npx.isnan(vs.mtt_cpr_rz[2:-2, 2:-2, :]), 0, vs.mtt_cpr_rz[2:-2, 2:-2, :]) * npx.where(dsa_rz1 > 0, ((vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] * settings.h) / dsa_rz1), 0)
        dsa_ss1 = (vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :]) * settings.h
        dmsa_ss1 = npx.where(npx.isnan(vs.mtt_q_rz[2:-2, 2:-2, :]), 0, vs.mtt_q_rz[2:-2, 2:-2, :]) * npx.where(dsa_ss1 > 0, ((vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] * settings.h) / dsa_ss1), 0)
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :] > 0) & (vs.msa_rz[2:-2, 2:-2, 1, :] > 0), vs.msa_rz[2:-2, 2:-2, 1, :] * (vs.sa_rz[2:-2, 2:-2, 1, :] / (dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :])), 0) + npx.where(dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :] > 0, dmsa_rz1 * (dsa_rz1 / (dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :])), 0),
        )
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_rz1 > 0) & (vs.msa_rz[2:-2, 2:-2, 1, :] <= 0), dmsa_rz1, vs.msa_rz[2:-2, 2:-2, 1, :]),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where(dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :] > 0, vs.msa_ss[2:-2, 2:-2, 1, :] * (vs.sa_ss[2:-2, 2:-2, 1, :] / (dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :])), 0) + npx.where(dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :] > 0, dmsa_ss1 * (dsa_ss1 / (dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :])), 0),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_ss1 > 0) & (vs.msa_ss[2:-2, 2:-2, 1, :] <= 0), dmsa_ss1, vs.msa_ss[2:-2, 2:-2, 1, :]),
        )
        vs.sa_rz = update_add(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :], dsa_rz,
        )
        vs.sa_ss = update_add(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :], dsa_ss,
        )
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where(vs.sa_rz[2:-2, 2:-2, 1, :] <= 0, 0, vs.msa_rz[2:-2, 2:-2, 1, :]),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where(vs.sa_ss[2:-2, 2:-2, 1, :] <= 0, 0, vs.msa_ss[2:-2, 2:-2, 1, :]),
        )
    elif settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate:
        # impose nonegative constrain of numerical solution
        dsa_rz = (vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :] - vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :]) * settings.h
        dsa_rz = npx.where(vs.sa_rz[2:-2, 2:-2, 1, :] + dsa_rz < 0, -vs.sa_rz[2:-2, 2:-2, 1, :], dsa_rz)
        dsa_ss = (vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :]) * settings.h
        dsa_ss = npx.where(vs.sa_ss[2:-2, 2:-2, 1, :] + dsa_ss < 0, -vs.sa_ss[2:-2, 2:-2, 1, :], dsa_ss)
        vs.sa_rz = update_add(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :], dsa_rz,
        )
        vs.sa_ss = update_add(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :], dsa_ss,
        )
        dmsa_rz = npx.where(npx.isnan(vs.mtt_cpr_rz[2:-2, 2:-2, :]), 0, vs.mtt_cpr_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_transp[2:-2, 2:-2, :]), 0, vs.mtt_transp[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_q_rz[2:-2, 2:-2, :]), 0, vs.mtt_q_rz[2:-2, 2:-2, :])
        dmsa_ss = npx.where(npx.isnan(vs.mtt_q_rz[2:-2, 2:-2, :]), 0, vs.mtt_q_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_cpr_rz[2:-2, 2:-2, :]), 0, vs.mtt_cpr_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_q_ss[2:-2, 2:-2, :]), 0, vs.mtt_q_ss[2:-2, 2:-2, :])
        dmsa_rz = npx.where(vs.msa_rz[2:-2, 2:-2, 1, :] + dmsa_rz < 0, 0, dmsa_rz)
        dmsa_ss = npx.where(vs.msa_ss[2:-2, 2:-2, 1, :] + dmsa_ss < 0, 0, dmsa_ss)
        vs.msa_rz = update_add(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], dmsa_rz,
        )
        vs.msa_ss = update_add(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], dmsa_ss,
        )

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, 1, 1:], npx.cumsum(vs.sa_rz[2:-2, 2:-2, 1, :], axis=-1),
    )
    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, 1, 1:], npx.cumsum(vs.sa_ss[2:-2, 2:-2, 1, :], axis=-1),
    )

    # calculate solute concentrations
    if settings.enable_oxygen18 or settings.enable_deuterium:
        vs.C_inf_mat_rz = update(
            vs.C_inf_mat_rz,
            at[2:-2, 2:-2], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_inf_pf_rz = update(
            vs.C_inf_pf_rz,
            at[2:-2, 2:-2], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_inf_pf_ss = update(
            vs.C_inf_pf_ss,
            at[2:-2, 2:-2], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_evap_soil = update(
            vs.C_evap_soil,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_evap_soil, vs.tt_evap_soil, vs.evap_soil)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_transp = update(
            vs.C_transp,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_transp, vs.tt_transp, vs.transp)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_q_rz = update(
            vs.C_q_rz,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_q_rz, vs.tt_q_rz, vs.q_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_cpr_rz = update(
            vs.C_cpr_rz,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_cpr_rz, vs.tt_cpr_rz, vs.cpr_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_q_ss = update(
            vs.C_q_ss,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_q_ss, vs.tt_q_ss, vs.q_ss)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )

    elif settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate:
        vs.C_inf_mat_rz = update(
            vs.C_inf_mat_rz,
            at[2:-2, 2:-2], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_inf_pf_rz = update(
            vs.C_inf_pf_rz,
            at[2:-2, 2:-2], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_inf_pf_ss = update(
            vs.C_inf_pf_ss,
            at[2:-2, 2:-2], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_transp = update(
            vs.C_transp,
            at[2:-2, 2:-2], npx.where(vs.transp > 0, npx.sum(vs.mtt_transp, axis=2) / vs.transp, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_q_rz = update(
            vs.C_q_rz,
            at[2:-2, 2:-2], npx.where(vs.q_rz > 0, npx.sum(vs.mtt_q_rz, axis=2) / vs.q_rz, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_cpr_rz = update(
            vs.C_cpr_rz,
            at[2:-2, 2:-2], npx.where(vs.cpr_rz > 0, npx.sum(vs.mtt_cpr_rz, axis=2) / vs.cpr_rz, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_q_ss = update(
            vs.C_q_ss,
            at[2:-2, 2:-2], npx.where(vs.q_ss > 0, npx.sum(vs.mtt_q_ss, axis=2) / vs.q_ss, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )

    return KernelOutput(tt_evap_soil=vs.tt_evap_soil, tt_transp=vs.tt_transp, tt_q_rz=vs.tt_q_rz, tt_cpr_rz=vs.tt_cpr_rz, tt_q_ss=vs.tt_q_ss,
                        TT_evap_soil=vs.TT_evap_soil, TT_transp=vs.TT_transp, TT_q_rz=vs.TT_q_rz, TT_cpr_rz=vs.TT_cpr_rz, TT_q_ss=vs.TT_q_ss,
                        mtt_evap_soil=vs.mtt_evap_soil, mtt_transp=vs.mtt_transp, mtt_q_rz=vs.mtt_q_rz, mtt_cpr_rz=vs.mtt_cpr_rz, mtt_q_ss=vs.mtt_q_ss,
                        C_inf_mat_rz=vs.C_inf_mat_rz, C_inf_pf_rz=vs.C_inf_pf_rz, C_inf_pf_ss=vs.C_inf_pf_ss, C_evap_soil=vs.C_evap_soil, C_transp=vs.C_transp, C_q_rz=vs.C_q_rz, C_cpr_rz=vs.C_cpr_rz, C_q_ss=vs.C_q_ss,
                        sa_rz=vs.sa_rz, sa_ss=vs.sa_ss, msa_rz=vs.msa_rz, msa_ss=vs.msa_ss)


@roger_kernel
def svat_lbc_transport_model_rk4(state):
    """Calculates svat transport model with Runge-Kutta method
    """
    pass


@roger_kernel
def svat_crop_transport_model_rk4(state):
    """Calculates solute transport model with Runge-Kutta method
    """
    pass


@roger_kernel
def svat_transport_model_euler(state):
    """Calculates solute transport model with explicit Euler method
    """
    vs = state.variables
    settings = state.settings

    # upper boundary condition
    vs.tt_inf_mat_rz = update(
        vs.tt_inf_mat_rz,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.tt_inf_pf_rz = update(
        vs.tt_inf_pf_rz,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.tt_inf_pf_ss = update(
        vs.tt_inf_pf_ss,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    if settings.enable_deuterium or settings.enable_oxygen18:
        vs.mtt_inf_mat_rz = update(
            vs.mtt_inf_mat_rz,
            at[2:-2, 2:-2, 0], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_pf_rz = update(
            vs.mtt_inf_pf_rz,
            at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_pf_ss = update(
            vs.mtt_inf_pf_ss,
            at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
    else:
        vs.mtt_inf_pf_ss = update(
            vs.mtt_inf_pf_ss,
            at[2:-2, 2:-2, 0], vs.C_in[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_mat_rz = update(
            vs.mtt_inf_mat_rz,
            at[2:-2, 2:-2, 0], vs.C_in[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_pf_rz = update(
            vs.mtt_inf_pf_rz,
            at[2:-2, 2:-2, 0], vs.C_in[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )

    # update StorAge with upper boundary condition
    if settings.enable_oxygen18 or settings.enable_deuterium:
        dsa_rz = (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :]) * settings.h
        dsa_ss = (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :]) * settings.h
        dsa_rz1 = (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :]) * settings.h
        dmsa_rz1 = npx.where(npx.isnan(vs.mtt_inf_mat_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_mat_rz[2:-2, 2:-2, :]) * npx.where(dsa_rz1 > 0, ((vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] * settings.h) / dsa_rz1), 0) + npx.where(npx.isnan(vs.mtt_inf_pf_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_rz[2:-2, 2:-2, :]) * npx.where(dsa_rz1 > 0, ((vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] * settings.h) / dsa_rz1), 0)
        dsa_ss1 = (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :]) * settings.h
        dmsa_ss1 = npx.where(npx.isnan(vs.mtt_inf_pf_ss[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_ss[2:-2, 2:-2, :]) * npx.where(dsa_ss1 > 0, ((vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] * settings.h) / dsa_ss1), 0)
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :] > 0) & (vs.msa_rz[2:-2, 2:-2, 1, :] > 0), vs.msa_rz[2:-2, 2:-2, 1, :] * (vs.sa_rz[2:-2, 2:-2, 1, :] / (dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :])), 0) + npx.where(dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :] > 0, dmsa_rz1 * (dsa_rz1 / (dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :])), 0),
        )
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_rz1 > 0) & (vs.msa_rz[2:-2, 2:-2, 1, :] <= 0), dmsa_rz1, vs.msa_rz[2:-2, 2:-2, 1, :]),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where(dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :] > 0, vs.msa_ss[2:-2, 2:-2, 1, :] * (vs.sa_ss[2:-2, 2:-2, 1, :] / (dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :])), 0) + npx.where(dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :] > 0, dmsa_ss1 * (dsa_ss1 / (dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :])), 0),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_ss1 > 0) & (vs.msa_ss[2:-2, 2:-2, 1, :] <= 0), dmsa_ss1, vs.msa_ss[2:-2, 2:-2, 1, :]),
        )
        vs.sa_rz = update_add(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :], dsa_rz,
        )
        vs.sa_ss = update_add(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :], dsa_ss,
        )
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where(vs.sa_rz[2:-2, 2:-2, 1, :] <= 0, 0, vs.msa_rz[2:-2, 2:-2, 1, :]),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where(vs.sa_ss[2:-2, 2:-2, 1, :] <= 0, 0, vs.msa_ss[2:-2, 2:-2, 1, :]),
        )
    elif settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate:
        dsa_rz = (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :]) * settings.h
        dsa_ss = (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :]) * settings.h
        vs.sa_rz = update_add(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :], dsa_rz,
        )
        vs.sa_ss = update_add(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :], dsa_ss,
        )
        dmsa_rz = npx.where(npx.isnan(vs.mtt_inf_mat_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_mat_rz[2:-2, 2:-2, :]) * settings.h + npx.where(npx.isnan(vs.mtt_inf_pf_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_rz[2:-2, 2:-2, :]) * settings.h
        dmsa_ss = npx.where(npx.isnan(vs.mtt_inf_pf_ss[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_ss[2:-2, 2:-2, :]) * settings.h
        vs.msa_rz = update_add(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], dmsa_rz,
        )
        vs.msa_ss = update_add(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], dmsa_ss,
        )

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :], calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    # calculate SAS
    vs.TT_evap_soil = update(
        vs.TT_evap_soil,
        at[2:-2, 2:-2, :], calc_TT_num(state, vs.SA_rz, vs.sas_params_evap_soil, vs.evap_soil * settings.h)[2:-2, 2:-2, :],
    )
    vs.TT_evap_soil = update(
        vs.TT_evap_soil,
        at[2:-2, 2:-2, :], calc_TT_num_nonneg(state, vs.SA_rz, vs.TT_evap_soil, vs.evap_soil * settings.h)[2:-2, 2:-2, :],
    )
    vs.tt_evap_soil = update(
        vs.tt_evap_soil,
        at[2:-2, 2:-2, :], npx.where(npx.diff(vs.TT_evap_soil[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(vs.TT_evap_soil[2:-2, 2:-2, :], axis=-1), 0),
    )
    if settings.enable_oxygen18 or settings.enable_deuterium:
        vs.mtt_evap_soil = update(
            vs.mtt_evap_soil,
            at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_rz, vs.tt_evap_soil, vs.evap_soil * settings.h, vs.msa_rz, vs.alpha_q)[2:-2, 2:-2, :],
        )
    vs.TT_transp = update(
        vs.TT_transp,
        at[2:-2, 2:-2, :], calc_TT_num(state, vs.SA_rz, vs.sas_params_transp, vs.transp * settings.h)[2:-2, 2:-2, :],
    )
    vs.TT_transp = update(
        vs.TT_transp,
        at[2:-2, 2:-2, :], calc_TT_num_nonneg(state, vs.SA_rz, vs.TT_transp, vs.transp * settings.h)[2:-2, 2:-2, :],
    )
    vs.tt_transp = update(
        vs.tt_transp,
        at[2:-2, 2:-2, :], npx.where(npx.diff(vs.TT_transp[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(vs.TT_transp[2:-2, 2:-2, :], axis=-1), 0),
    )
    vs.mtt_transp = update(
        vs.mtt_transp,
        at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_rz, vs.tt_transp, vs.transp * settings.h, vs.msa_rz, vs.alpha_transp)[2:-2, 2:-2, :],
    )
    vs.TT_q_rz = update(
        vs.TT_q_rz,
        at[2:-2, 2:-2, :], calc_TT_num(state, vs.SA_rz, vs.sas_params_q_rz, vs.q_rz * settings.h)[2:-2, 2:-2, :],
    )
    vs.TT_q_rz = update(
        vs.TT_q_rz,
        at[2:-2, 2:-2, :], calc_TT_num_nonneg(state, vs.SA_rz, vs.TT_q_rz, vs.q_rz * settings.h)[2:-2, 2:-2, :],
    )
    vs.tt_q_rz = update(
        vs.tt_q_rz,
        at[2:-2, 2:-2, :], npx.where(npx.diff(vs.TT_q_rz[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(vs.TT_q_rz[2:-2, 2:-2, :], axis=-1), 0),
    )
    vs.mtt_q_rz = update(
        vs.mtt_q_rz,
        at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_rz, vs.tt_q_rz, vs.q_rz * settings.h, vs.msa_rz, vs.alpha_q)[2:-2, 2:-2, :],
    )
    vs.TT_cpr_rz = update(
        vs.TT_cpr_rz,
        at[2:-2, 2:-2, :], calc_TT_num(state, vs.SA_ss, vs.sas_params_cpr_rz, vs.cpr_rz * settings.h)[2:-2, 2:-2, :],
    )
    vs.TT_cpr_rz = update(
        vs.TT_cpr_rz,
        at[2:-2, 2:-2, :], calc_TT_num_nonneg(state, vs.SA_ss, vs.TT_cpr_rz, vs.cpr_rz * settings.h)[2:-2, 2:-2, :],
    )
    vs.tt_cpr_rz = update(
        vs.tt_cpr_rz,
        at[2:-2, 2:-2, :], npx.where(npx.diff(vs.TT_cpr_rz[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(vs.TT_cpr_rz[2:-2, 2:-2, :], axis=-1), 0),
    )
    vs.mtt_cpr_rz = update(
        vs.mtt_cpr_rz,
        at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_ss, vs.tt_cpr_rz, vs.cpr_rz * settings.h, vs.msa_ss, vs.alpha_q)[2:-2, 2:-2, :],
    )
    vs.TT_q_ss = update(
        vs.TT_q_ss,
        at[2:-2, 2:-2, :], calc_TT_num(state, vs.SA_ss, vs.sas_params_q_ss, vs.q_ss * settings.h)[2:-2, 2:-2, :],
    )
    vs.TT_q_ss = update(
        vs.TT_q_ss,
        at[2:-2, 2:-2, :], calc_TT_num_nonneg(state, vs.SA_ss, vs.TT_q_ss, vs.q_ss * settings.h)[2:-2, 2:-2, :],
    )
    vs.tt_q_ss = update(
        vs.tt_q_ss,
        at[2:-2, 2:-2, :], npx.where(npx.diff(vs.TT_q_ss[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(vs.TT_q_ss[2:-2, 2:-2, :], axis=-1), 0),
    )
    vs.mtt_q_ss = update(
        vs.mtt_q_ss,
        at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_ss, vs.tt_q_ss, vs.q_ss * settings.h, vs.msa_ss, vs.alpha_q)[2:-2, 2:-2, :],
    )

    # update StorAge
    if settings.enable_oxygen18 or settings.enable_deuterium:
        # impose nonegative constrain to numerical solution
        dsa_rz = (vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :] - vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :]) * settings.h
        dsa_rz = npx.where(vs.sa_rz[2:-2, 2:-2, 1, :] + dsa_rz < 0, -vs.sa_rz[2:-2, 2:-2, 1, :], dsa_rz)
        dsa_ss = (vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :]) * settings.h
        dsa_ss = npx.where(vs.sa_ss[2:-2, 2:-2, 1, :] + dsa_ss < 0, -vs.sa_ss[2:-2, 2:-2, 1, :], dsa_ss)
        dsa_rz1 = (vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :]) * settings.h
        dmsa_rz1 = npx.where(npx.isnan(vs.mtt_cpr_rz[2:-2, 2:-2, :]), 0, vs.mtt_cpr_rz[2:-2, 2:-2, :]) * npx.where(dsa_rz1 > 0, ((vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] * settings.h) / dsa_rz1), 0)
        dsa_ss1 = (vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :]) * settings.h
        dmsa_ss1 = npx.where(npx.isnan(vs.mtt_q_rz[2:-2, 2:-2, :]), 0, vs.mtt_q_rz[2:-2, 2:-2, :]) * npx.where(dsa_ss1 > 0, ((vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] * settings.h) / dsa_ss1), 0)
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :] > 0) & (vs.msa_rz[2:-2, 2:-2, 1, :] > 0), vs.msa_rz[2:-2, 2:-2, 1, :] * (vs.sa_rz[2:-2, 2:-2, 1, :] / (dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :])), 0) + npx.where(dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :] > 0, dmsa_rz1 * (dsa_rz1 / (dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :])), 0),
        )
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_rz1 > 0) & (vs.msa_rz[2:-2, 2:-2, 1, :] <= 0), dmsa_rz1, vs.msa_rz[2:-2, 2:-2, 1, :]),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where(dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :] > 0, vs.msa_ss[2:-2, 2:-2, 1, :] * (vs.sa_ss[2:-2, 2:-2, 1, :] / (dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :])), 0) + npx.where(dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :] > 0, dmsa_ss1 * (dsa_ss1 / (dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :])), 0),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_ss1 > 0) & (vs.msa_ss[2:-2, 2:-2, 1, :] <= 0), dmsa_ss1, vs.msa_ss[2:-2, 2:-2, 1, :]),
        )
        vs.sa_rz = update_add(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :], dsa_rz,
        )
        vs.sa_ss = update_add(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :], dsa_ss,
        )
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where(vs.sa_rz[2:-2, 2:-2, 1, :] <= 0, 0, vs.msa_rz[2:-2, 2:-2, 1, :]),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where(vs.sa_ss[2:-2, 2:-2, 1, :] <= 0, 0, vs.msa_ss[2:-2, 2:-2, 1, :]),
        )
    elif settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate:
        # impose nonegative constrain of numerical solution
        dsa_rz = (vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :] - vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :]) * settings.h
        dsa_rz = npx.where(vs.sa_rz[2:-2, 2:-2, 1, :] + dsa_rz < 0, -vs.sa_rz[2:-2, 2:-2, 1, :], dsa_rz)
        dsa_ss = (vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :]) * settings.h
        dsa_ss = npx.where(vs.sa_ss[2:-2, 2:-2, 1, :] + dsa_ss < 0, -vs.sa_ss[2:-2, 2:-2, 1, :], dsa_ss)
        vs.sa_rz = update_add(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :], dsa_rz,
        )
        vs.sa_ss = update_add(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :], dsa_ss,
        )
        dmsa_rz = npx.where(npx.isnan(vs.mtt_cpr_rz[2:-2, 2:-2, :]), 0, vs.mtt_cpr_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_transp[2:-2, 2:-2, :]), 0, vs.mtt_transp[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_q_rz[2:-2, 2:-2, :]), 0, vs.mtt_q_rz[2:-2, 2:-2, :])
        dmsa_ss = npx.where(npx.isnan(vs.mtt_q_rz[2:-2, 2:-2, :]), 0, vs.mtt_q_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_cpr_rz[2:-2, 2:-2, :]), 0, vs.mtt_cpr_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_q_ss[2:-2, 2:-2, :]), 0, vs.mtt_q_ss[2:-2, 2:-2, :])
        dmsa_rz = npx.where(vs.msa_rz[2:-2, 2:-2, 1, :] + dmsa_rz < 0, 0, dmsa_rz)
        dmsa_ss = npx.where(vs.msa_ss[2:-2, 2:-2, 1, :] + dmsa_ss < 0, 0, dmsa_ss)
        vs.msa_rz = update_add(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], dmsa_rz,
        )
        vs.msa_ss = update_add(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], dmsa_ss,
        )

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, 1, 1:], npx.cumsum(vs.sa_rz[2:-2, 2:-2, 1, :], axis=-1),
    )
    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, 1, 1:], npx.cumsum(vs.sa_ss[2:-2, 2:-2, 1, :], axis=-1),
    )

    # calculate solute concentrations
    if settings.enable_oxygen18 or settings.enable_deuterium:
        vs.C_inf_mat_rz = update(
            vs.C_inf_mat_rz,
            at[2:-2, 2:-2], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_inf_pf_rz = update(
            vs.C_inf_pf_rz,
            at[2:-2, 2:-2], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_inf_pf_ss = update(
            vs.C_inf_pf_ss,
            at[2:-2, 2:-2], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_evap_soil = update(
            vs.C_evap_soil,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_evap_soil, vs.tt_evap_soil, vs.evap_soil)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_transp = update(
            vs.C_transp,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_transp, vs.tt_transp, vs.transp)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_q_rz = update(
            vs.C_q_rz,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_q_rz, vs.tt_q_rz, vs.q_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_cpr_rz = update(
            vs.C_cpr_rz,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_cpr_rz, vs.tt_cpr_rz, vs.cpr_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_q_ss = update(
            vs.C_q_ss,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_q_ss, vs.tt_q_ss, vs.q_ss)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )

    elif settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate:
        vs.C_inf_mat_rz = update(
            vs.C_inf_mat_rz,
            at[2:-2, 2:-2], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_inf_pf_rz = update(
            vs.C_inf_pf_rz,
            at[2:-2, 2:-2], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_inf_pf_ss = update(
            vs.C_inf_pf_ss,
            at[2:-2, 2:-2], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_transp = update(
            vs.C_transp,
            at[2:-2, 2:-2], npx.where(vs.transp > 0, npx.sum(vs.mtt_transp, axis=2) / vs.transp, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_q_rz = update(
            vs.C_q_rz,
            at[2:-2, 2:-2], npx.where(vs.q_rz > 0, npx.sum(vs.mtt_q_rz, axis=2) / vs.q_rz, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_cpr_rz = update(
            vs.C_cpr_rz,
            at[2:-2, 2:-2], npx.where(vs.cpr_rz > 0, npx.sum(vs.mtt_cpr_rz, axis=2) / vs.cpr_rz, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_q_ss = update(
            vs.C_q_ss,
            at[2:-2, 2:-2], npx.where(vs.q_ss > 0, npx.sum(vs.mtt_q_ss, axis=2) / vs.q_ss, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )

    return KernelOutput(tt_evap_soil=vs.tt_evap_soil, tt_transp=vs.tt_transp, tt_q_rz=vs.tt_q_rz, tt_cpr_rz=vs.tt_cpr_rz, tt_q_ss=vs.tt_q_ss,
                        TT_evap_soil=vs.TT_evap_soil, TT_transp=vs.TT_transp, TT_q_rz=vs.TT_q_rz, TT_cpr_rz=vs.TT_cpr_rz, TT_q_ss=vs.TT_q_ss,
                        mtt_evap_soil=vs.mtt_evap_soil, mtt_transp=vs.mtt_transp, mtt_q_rz=vs.mtt_q_rz, mtt_cpr_rz=vs.mtt_cpr_rz, mtt_q_ss=vs.mtt_q_ss,
                        C_inf_mat_rz=vs.C_inf_mat_rz, C_inf_pf_rz=vs.C_inf_pf_rz, C_inf_pf_ss=vs.C_inf_pf_ss, C_evap_soil=vs.C_evap_soil, C_transp=vs.C_transp, C_q_rz=vs.C_q_rz, C_cpr_rz=vs.C_cpr_rz, C_q_ss=vs.C_q_ss,
                        sa_rz=vs.sa_rz, sa_ss=vs.sa_ss, msa_rz=vs.msa_rz, msa_ss=vs.msa_ss)


@roger_kernel
def svat_lbc_transport_model_euler(state):
    """Calculates solute transport model with explicit Euler method
    """
    vs = state.variables
    settings = state.settings

    # upper boundary condition
    vs.tt_inf_mat_rz = update(
        vs.tt_inf_mat_rz,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.tt_inf_pf_rz = update(
        vs.tt_inf_pf_rz,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.tt_inf_pf_ss = update(
        vs.tt_inf_pf_ss,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    # lower boundary condition
    vs.tt_cpr_ss = update(
        vs.tt_cpr_ss,
        at[2:-2, 2:-2, -1], npx.where(vs.cpr_ss[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    if settings.enable_deuterium or settings.enable_oxygen18:
        vs.mtt_inf_mat_rz = update(
            vs.mtt_inf_mat_rz,
            at[2:-2, 2:-2, 0], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_pf_rz = update(
            vs.mtt_inf_pf_rz,
            at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_pf_ss = update(
            vs.mtt_inf_pf_ss,
            at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_cpr_ss = update(
            vs.mtt_cpr_ss,
            at[2:-2, 2:-2, :], npx.where(vs.inf_cpr_ss[2:-2, 2:-2] * vs.tt_cpr_ss[2:-2, 2:-2, :] > 0, vs.C_cpr_ss[2:-2, 2:-2, npx.newaxis], 0) * vs.maskCatch[2:-2, 2:-2],
        )
    else:
        vs.mtt_inf_pf_ss = update(
            vs.mtt_inf_pf_ss,
            at[2:-2, 2:-2, 0], vs.inf_pf_ss[2:-2, 2:-2] * vs.C_in[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_mat_rz = update(
            vs.mtt_inf_mat_rz,
            at[2:-2, 2:-2, 0], vs.inf_mat_rz[2:-2, 2:-2] * vs.C_in[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_pf_rz = update(
            vs.mtt_inf_pf_rz,
            at[2:-2, 2:-2, 0], vs.inf_pf_rz[2:-2, 2:-2] * vs.C_in[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_cpr_ss = update(
            vs.mtt_cpr_ss,
            at[2:-2, 2:-2, :], vs.tt_cpr_ss[2:-2, 2:-2, :] * vs.C_cpr_ss[2:-2, 2:-2, npx.newaxis] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
        )

    # update StorAge with upper boundary condition
    if settings.enable_oxygen18 or settings.enable_deuterium:
        dsa_rz = (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :]) * settings.h
        dsa_ss = (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :]) * settings.h
        dsa_rz1 = (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :]) * settings.h
        dmsa_rz1 = npx.where(npx.isnan(vs.mtt_inf_mat_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_mat_rz[2:-2, 2:-2, :]) * npx.where(dsa_rz1 > 0, ((vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] * settings.h) / dsa_rz1), 0) + npx.where(npx.isnan(vs.mtt_inf_pf_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_rz[2:-2, 2:-2, :]) * npx.where(dsa_rz1 > 0, ((vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] * settings.h) / dsa_rz1), 0)
        dsa_ss1 = (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :]) * settings.h
        dmsa_ss1 = npx.where(npx.isnan(vs.mtt_inf_pf_ss[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_ss[2:-2, 2:-2, :]) * npx.where(dsa_ss1 > 0, ((vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] * settings.h) / dsa_ss1), 0)
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :] > 0) & (vs.msa_rz[2:-2, 2:-2, 1, :] > 0), vs.msa_rz[2:-2, 2:-2, 1, :] * (vs.sa_rz[2:-2, 2:-2, 1, :] / (dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :])), 0) + npx.where(dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :] > 0, dmsa_rz1 * (dsa_rz1 / (dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :])), 0),
        )
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_rz1 > 0) & (vs.msa_rz[2:-2, 2:-2, 1, :] <= 0), dmsa_rz1, vs.msa_rz[2:-2, 2:-2, 1, :]),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where(dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :] > 0, vs.msa_ss[2:-2, 2:-2, 1, :] * (vs.sa_ss[2:-2, 2:-2, 1, :] / (dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :])), 0) + npx.where(dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :] > 0, dmsa_ss1 * (dsa_ss1 / (dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :])), 0),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_ss1 > 0) & (vs.msa_ss[2:-2, 2:-2, 1, :] <= 0), dmsa_ss1, vs.msa_ss[2:-2, 2:-2, 1, :]),
        )
        vs.sa_rz = update_add(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :], dsa_rz,
        )
        vs.sa_ss = update_add(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :], dsa_ss,
        )
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where(vs.sa_rz[2:-2, 2:-2, 1, :] <= 0, 0, vs.msa_rz[2:-2, 2:-2, 1, :]),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where(vs.sa_ss[2:-2, 2:-2, 1, :] <= 0, 0, vs.msa_ss[2:-2, 2:-2, 1, :]),
        )
    elif settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate:
        dsa_rz = (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :]) * settings.h
        dsa_ss = (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :]) * settings.h
        vs.sa_rz = update_add(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :], dsa_rz,
        )
        vs.sa_ss = update_add(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :], dsa_ss,
        )
        dmsa_rz = npx.where(npx.isnan(vs.mtt_inf_mat_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_mat_rz[2:-2, 2:-2, :]) * settings.h + npx.where(npx.isnan(vs.mtt_inf_pf_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_rz[2:-2, 2:-2, :]) * settings.h
        dmsa_ss = npx.where(npx.isnan(vs.mtt_inf_pf_ss[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_ss[2:-2, 2:-2, :]) * settings.h
        vs.msa_rz = update_add(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], dmsa_rz,
        )
        vs.msa_ss = update_add(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], dmsa_ss,
        )

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :], calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    # calculate SAS
    vs.TT_evap_soil = update(
        vs.TT_evap_soil,
        at[2:-2, 2:-2, :], calc_TT_num(state, vs.SA_rz, vs.sas_params_evap_soil, vs.evap_soil * settings.h)[2:-2, 2:-2, :],
    )
    vs.tt_evap_soil = update(
        vs.tt_evap_soil,
        at[2:-2, 2:-2, :], npx.where(npx.diff(vs.TT_evap_soil[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(vs.TT_evap_soil[2:-2, 2:-2, :], axis=-1), 0),
    )
    if settings.enable_oxygen18 or settings.enable_deuterium:
        vs.mtt_evap_soil = update(
            vs.mtt_evap_soil,
            at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_rz, vs.tt_evap_soil, vs.evap_soil * settings.h, vs.msa_rz, vs.alpha_q)[2:-2, 2:-2, :],
        )
    vs.TT_transp = update(
        vs.TT_transp,
        at[2:-2, 2:-2, :], calc_TT_num(state, vs.SA_rz, vs.sas_params_transp, vs.transp * settings.h)[2:-2, 2:-2, :],
    )
    vs.tt_transp = update(
        vs.tt_transp,
        at[2:-2, 2:-2, :], npx.where(npx.diff(vs.TT_transp[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(vs.TT_transp[2:-2, 2:-2, :], axis=-1), 0),
    )
    vs.mtt_transp = update(
        vs.mtt_transp,
        at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_rz, vs.tt_transp, vs.transp * settings.h, vs.msa_rz, vs.alpha_transp)[2:-2, 2:-2, :],
    )
    vs.TT_q_rz = update(
        vs.TT_q_rz,
        at[2:-2, 2:-2, :], calc_TT_num(state, vs.SA_rz, vs.sas_params_q_rz, vs.q_rz * settings.h)[2:-2, 2:-2, :],
    )
    vs.tt_q_rz = update(
        vs.tt_q_rz,
        at[2:-2, 2:-2, :], npx.where(npx.diff(vs.TT_q_rz[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(vs.TT_q_rz[2:-2, 2:-2, :], axis=-1), 0),
    )
    vs.mtt_q_rz = update(
        vs.mtt_q_rz,
        at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_rz, vs.tt_q_rz, vs.q_rz * settings.h, vs.msa_rz, vs.alpha_q)[2:-2, 2:-2, :],
    )
    vs.TT_cpr_rz = update(
        vs.TT_cpr_rz,
        at[2:-2, 2:-2, :], calc_TT_num(state, vs.SA_ss, vs.sas_params_cpr_rz, vs.cpr_rz * settings.h)[2:-2, 2:-2, :],
    )
    vs.tt_cpr_rz = update(
        vs.tt_cpr_rz,
        at[2:-2, 2:-2, :], npx.where(npx.diff(vs.TT_cpr_rz[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(vs.TT_cpr_rz[2:-2, 2:-2, :], axis=-1), 0),
    )
    vs.mtt_cpr_rz = update(
        vs.mtt_cpr_rz,
        at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_ss, vs.tt_cpr_rz, vs.cpr_rz * settings.h, vs.msa_ss, vs.alpha_q)[2:-2, 2:-2, :],
    )
    vs.TT_q_ss = update(
        vs.TT_q_ss,
        at[2:-2, 2:-2, :], calc_TT_num(state, vs.SA_ss, vs.sas_params_q_ss, vs.q_ss * settings.h)[2:-2, 2:-2, :],
    )
    vs.tt_q_ss = update(
        vs.tt_q_ss,
        at[2:-2, 2:-2, :], npx.where(npx.diff(vs.TT_q_ss[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(vs.TT_q_ss[2:-2, 2:-2, :], axis=-1), 0),
    )
    vs.mtt_q_ss = update(
        vs.mtt_q_ss,
        at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_ss, vs.tt_q_ss, vs.q_ss * settings.h, vs.msa_ss, vs.alpha_q)[2:-2, 2:-2, :],
    )

    # update StorAge
    if settings.enable_oxygen18 or settings.enable_deuterium:
        # impose nonegative constrain to numerical solution
        dsa_rz = (vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :] - vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :]) * settings.h
        dsa_rz = npx.where(vs.sa_rz[2:-2, 2:-2, 1, :] + dsa_rz < 0, -vs.sa_rz[2:-2, 2:-2, 1, :], dsa_rz)
        dsa_ss = (vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] + vs.cpr_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_ss[2:-2, 2:-2, :] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :]) * settings.h
        dsa_ss = npx.where(vs.sa_ss[2:-2, 2:-2, 1, :] + dsa_ss < 0, -vs.sa_ss[2:-2, 2:-2, 1, :], dsa_ss)
        dsa_rz1 = (vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :]) * settings.h
        dmsa_rz1 = npx.where(npx.isnan(vs.mtt_cpr_rz[2:-2, 2:-2, :]), 0, vs.mtt_cpr_rz[2:-2, 2:-2, :]) * npx.where(dsa_rz1 > 0, ((vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] * settings.h) / dsa_rz1), 0)
        dsa_ss1 = (vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] + vs.cpr_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_ss[2:-2, 2:-2, :]) * settings.h
        dmsa_ss1 = npx.where(npx.isnan(vs.mtt_q_rz[2:-2, 2:-2, :]), 0, vs.mtt_q_rz[2:-2, 2:-2, :]) * npx.where(dsa_ss1 > 0, ((vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] * settings.h) / dsa_ss1), 0) + npx.where(npx.isnan(vs.mtt_cpr_ss[2:-2, 2:-2, :]), 0, vs.mtt_cpr_ss[2:-2, 2:-2, :]) * npx.where(dsa_ss1 > 0, ((vs.cpr_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_ss[2:-2, 2:-2, :] * settings.h) / dsa_ss1), 0)
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :] > 0) & (vs.msa_rz[2:-2, 2:-2, 1, :] > 0), vs.msa_rz[2:-2, 2:-2, 1, :] * (vs.sa_rz[2:-2, 2:-2, 1, :] / (dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :])), 0) + npx.where(dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :] > 0, dmsa_rz1 * (dsa_rz1 / (dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :])), 0),
        )
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_rz1 > 0) & (vs.msa_rz[2:-2, 2:-2, 1, :] <= 0), dmsa_rz1, vs.msa_rz[2:-2, 2:-2, 1, :]),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where(dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :] > 0, vs.msa_ss[2:-2, 2:-2, 1, :] * (vs.sa_ss[2:-2, 2:-2, 1, :] / (dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :])), 0) + npx.where(dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :] > 0, dmsa_ss1 * (dsa_ss1 / (dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :])), 0),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_ss1 > 0) & (vs.msa_ss[2:-2, 2:-2, 1, :] <= 0), dmsa_ss1, vs.msa_ss[2:-2, 2:-2, 1, :]),
        )
        vs.sa_rz = update_add(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :], dsa_rz,
        )
        vs.sa_ss = update_add(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :], dsa_ss,
        )
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where(vs.sa_rz[2:-2, 2:-2, 1, :] <= 0, 0, vs.msa_rz[2:-2, 2:-2, 1, :]),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where(vs.sa_ss[2:-2, 2:-2, 1, :] <= 0, 0, vs.msa_ss[2:-2, 2:-2, 1, :]),
        )
    elif settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate:
        # impose nonegative constrain of numerical solution
        dsa_rz = (vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :] - vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :]) * settings.h
        dsa_rz = npx.where(vs.sa_rz[2:-2, 2:-2, 1, :] + dsa_rz < 0, -vs.sa_rz[2:-2, 2:-2, 1, :], dsa_rz)
        dsa_ss = (vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] + vs.cpr_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_ss[2:-2, 2:-2, :] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :]) * settings.h
        dsa_ss = npx.where(vs.sa_ss[2:-2, 2:-2, 1, :] + dsa_ss < 0, -vs.sa_ss[2:-2, 2:-2, 1, :], dsa_ss)
        vs.sa_rz = update_add(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :], dsa_rz,
        )
        vs.sa_ss = update_add(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :], dsa_ss,
        )
        dmsa_rz = npx.where(npx.isnan(vs.mtt_cpr_rz[2:-2, 2:-2, :]), 0, vs.mtt_cpr_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_transp[2:-2, 2:-2, :]), 0, vs.mtt_transp[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_q_rz[2:-2, 2:-2, :]), 0, vs.mtt_q_rz[2:-2, 2:-2, :])
        dmsa_ss = npx.where(npx.isnan(vs.mtt_cpr_ss[2:-2, 2:-2, :]), 0, vs.mtt_cpr_ss[2:-2, 2:-2, :]) * settings.h + npx.where(npx.isnan(vs.mtt_q_rz[2:-2, 2:-2, :]), 0, vs.mtt_q_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_cpr_rz[2:-2, 2:-2, :]), 0, vs.mtt_cpr_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_q_ss[2:-2, 2:-2, :]), 0, vs.mtt_q_ss[2:-2, 2:-2, :])
        dmsa_rz = npx.where(vs.msa_rz[2:-2, 2:-2, 1, :] + dmsa_rz < 0, 0, dmsa_rz)
        dmsa_ss = npx.where(vs.msa_ss[2:-2, 2:-2, 1, :] + dmsa_ss < 0, 0, dmsa_ss)
        vs.msa_rz = update_add(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], dmsa_rz,
        )
        vs.msa_ss = update_add(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], dmsa_ss,
        )
    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, 1, 1:], npx.cumsum(vs.sa_rz[2:-2, 2:-2, 1, :], axis=-1),
    )
    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, 1, 1:], npx.cumsum(vs.sa_ss[2:-2, 2:-2, 1, :], axis=-1),
    )

    # calculate solute concentrations
    if settings.enable_oxygen18 or settings.enable_deuterium:
        vs.C_inf_mat_rz = update(
            vs.C_inf_mat_rz,
            at[2:-2, 2:-2], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_inf_pf_rz = update(
            vs.C_inf_pf_rz,
            at[2:-2, 2:-2], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_inf_pf_ss = update(
            vs.C_inf_pf_ss,
            at[2:-2, 2:-2], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_evap_soil = update(
            vs.C_evap_soil,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_evap_soil, vs.tt_evap_soil, vs.evap_soil)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_transp = update(
            vs.C_transp,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_transp, vs.tt_transp, vs.transp)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_q_rz = update(
            vs.C_q_rz,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_q_rz, vs.tt_q_rz, vs.q_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_cpr_rz = update(
            vs.C_cpr_rz,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_cpr_rz, vs.tt_cpr_rz, vs.cpr_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_q_ss = update(
            vs.C_q_ss,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_q_ss, vs.tt_q_ss, vs.q_ss)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )

    elif settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate:
        vs.C_inf_mat_rz = update(
            vs.C_inf_mat_rz,
            at[2:-2, 2:-2], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_inf_pf_rz = update(
            vs.C_inf_pf_rz,
            at[2:-2, 2:-2], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_inf_pf_ss = update(
            vs.C_inf_pf_ss,
            at[2:-2, 2:-2], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_transp = update(
            vs.C_transp,
            at[2:-2, 2:-2], npx.where(vs.transp > 0, npx.sum(vs.mtt_transp, axis=2) / vs.transp, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_q_rz = update(
            vs.C_q_rz,
            at[2:-2, 2:-2], npx.where(vs.q_rz > 0, npx.sum(vs.mtt_q_rz, axis=2) / vs.q_rz, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_cpr_rz = update(
            vs.C_cpr_rz,
            at[2:-2, 2:-2], npx.where(vs.cpr_rz > 0, npx.sum(vs.mtt_cpr_rz, axis=2) / vs.cpr_rz, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_q_ss = update(
            vs.C_q_ss,
            at[2:-2, 2:-2], npx.where(vs.q_ss > 0, npx.sum(vs.mtt_q_ss, axis=2) / vs.q_ss, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )

    return KernelOutput(tt_evap_soil=vs.tt_evap_soil, tt_transp=vs.tt_transp, tt_q_rz=vs.tt_q_rz, tt_cpr_rz=vs.tt_cpr_rz, tt_q_ss=vs.tt_q_ss,
                        TT_evap_soil=vs.TT_evap_soil, TT_transp=vs.TT_transp, TT_q_rz=vs.TT_q_rz, TT_cpr_rz=vs.TT_cpr_rz, TT_q_ss=vs.TT_q_ss,
                        mtt_evap_soil=vs.mtt_evap_soil, mtt_transp=vs.mtt_transp, mtt_q_rz=vs.mtt_q_rz, mtt_cpr_rz=vs.mtt_cpr_rz, mtt_q_ss=vs.mtt_q_ss,
                        C_inf_mat_rz=vs.C_inf_mat_rz, C_inf_pf_rz=vs.C_inf_pf_rz, C_inf_pf_ss=vs.C_inf_pf_ss, C_evap_soil=vs.C_evap_soil, C_transp=vs.C_transp, C_q_rz=vs.C_q_rz, C_cpr_rz=vs.C_cpr_rz, C_q_ss=vs.C_q_ss,
                        sa_rz=vs.sa_rz, sa_ss=vs.sa_ss, msa_rz=vs.msa_rz, msa_ss=vs.msa_ss)


@roger_kernel
def svat_crop_transport_model_euler(state):
    """Calculates solute transport model with explicit Euler method
    """
    vs = state.variables
    settings = state.settings

    # upper boundary condition
    vs.tt_inf_mat_rz = update(
        vs.tt_inf_mat_rz,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.tt_inf_pf_rz = update(
        vs.tt_inf_pf_rz,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.tt_inf_pf_ss = update(
        vs.tt_inf_pf_ss,
        at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, 1, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    if settings.enable_deuterium or settings.enable_oxygen18:
        vs.mtt_inf_mat_rz = update(
            vs.mtt_inf_mat_rz,
            at[2:-2, 2:-2, 0], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_pf_rz = update(
            vs.mtt_inf_pf_rz,
            at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_pf_ss = update(
            vs.mtt_inf_pf_ss,
            at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
    else:
        vs.mtt_inf_pf_ss = update(
            vs.mtt_inf_pf_ss,
            at[2:-2, 2:-2, 0], vs.C_in[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_mat_rz = update(
            vs.mtt_inf_mat_rz,
            at[2:-2, 2:-2, 0], vs.C_in[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_pf_rz = update(
            vs.mtt_inf_pf_rz,
            at[2:-2, 2:-2, 0], vs.C_in[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )

    # update StorAge with upper boundary condition
    if settings.enable_oxygen18 or settings.enable_deuterium:
        dsa_rz = (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :]) * settings.h
        dsa_ss = (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :]) * settings.h
        dsa_rz1 = (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :]) * settings.h
        dmsa_rz1 = npx.where(npx.isnan(vs.mtt_inf_mat_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_mat_rz[2:-2, 2:-2, :]) * npx.where(dsa_rz1 > 0, ((vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] * settings.h) / dsa_rz1), 0) + npx.where(npx.isnan(vs.mtt_inf_pf_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_rz[2:-2, 2:-2, :]) * npx.where(dsa_rz1 > 0, ((vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] * settings.h) / dsa_rz1), 0)
        dsa_ss1 = (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :]) * settings.h
        dmsa_ss1 = npx.where(npx.isnan(vs.mtt_inf_pf_ss[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_ss[2:-2, 2:-2, :]) * npx.where(dsa_ss1 > 0, ((vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] * settings.h) / dsa_ss1), 0)
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :] > 0) & (vs.msa_rz[2:-2, 2:-2, 1, :] > 0), vs.msa_rz[2:-2, 2:-2, 1, :] * (vs.sa_rz[2:-2, 2:-2, 1, :] / (dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :])), 0) + npx.where(dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :] > 0, dmsa_rz1 * (dsa_rz1 / (dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :])), 0),
        )
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_rz1 > 0) & (vs.msa_rz[2:-2, 2:-2, 1, :] <= 0), dmsa_rz1, vs.msa_rz[2:-2, 2:-2, 1, :]),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where(dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :] > 0, vs.msa_ss[2:-2, 2:-2, 1, :] * (vs.sa_ss[2:-2, 2:-2, 1, :] / (dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :])), 0) + npx.where(dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :] > 0, dmsa_ss1 * (dsa_ss1 / (dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :])), 0),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_ss1 > 0) & (vs.msa_ss[2:-2, 2:-2, 1, :] <= 0), dmsa_ss1, vs.msa_ss[2:-2, 2:-2, 1, :]),
        )
        vs.sa_rz = update_add(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :], dsa_rz,
        )
        vs.sa_ss = update_add(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :], dsa_ss,
        )
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where(vs.sa_rz[2:-2, 2:-2, 1, :] <= 0, 0, vs.msa_rz[2:-2, 2:-2, 1, :]),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where(vs.sa_ss[2:-2, 2:-2, 1, :] <= 0, 0, vs.msa_ss[2:-2, 2:-2, 1, :]),
        )
    elif settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate:
        dsa_rz = (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :]) * settings.h
        dsa_ss = (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :]) * settings.h
        vs.sa_rz = update_add(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :], dsa_rz,
        )
        vs.sa_ss = update_add(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :], dsa_ss,
        )
        dmsa_rz = npx.where(npx.isnan(vs.mtt_inf_mat_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_mat_rz[2:-2, 2:-2, :]) * settings.h + npx.where(npx.isnan(vs.mtt_inf_pf_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_rz[2:-2, 2:-2, :]) * settings.h
        dmsa_ss = npx.where(npx.isnan(vs.mtt_inf_pf_ss[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_ss[2:-2, 2:-2, :]) * settings.h
        vs.msa_rz = update_add(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], dmsa_rz,
        )
        vs.msa_ss = update_add(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], dmsa_ss,
        )

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :], calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    # calculate SAS
    vs.TT_evap_soil = update(
        vs.TT_evap_soil,
        at[2:-2, 2:-2, :], calc_TT_num(state, vs.SA_rz, vs.sas_params_evap_soil, vs.evap_soil * settings.h)[2:-2, 2:-2, :],
    )
    vs.tt_evap_soil = update(
        vs.tt_evap_soil,
        at[2:-2, 2:-2, :], npx.where(npx.diff(vs.TT_evap_soil[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(vs.TT_evap_soil[2:-2, 2:-2, :], axis=-1), 0),
    )
    if settings.enable_oxygen18 or settings.enable_deuterium:
        vs.mtt_evap_soil = update(
            vs.mtt_evap_soil,
            at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_rz, vs.tt_evap_soil, vs.evap_soil * settings.h, vs.msa_rz, vs.alpha_q)[2:-2, 2:-2, :],
        )
    vs.TT_transp = update(
        vs.TT_transp,
        at[2:-2, 2:-2, :], calc_TT_num(state, vs.SA_rz, vs.sas_params_transp, vs.transp * settings.h)[2:-2, 2:-2, :],
    )
    vs.tt_transp = update(
        vs.tt_transp,
        at[2:-2, 2:-2, :], npx.where(npx.diff(vs.TT_transp[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(vs.TT_transp[2:-2, 2:-2, :], axis=-1), 0),
    )
    vs.mtt_transp = update(
        vs.mtt_transp,
        at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_rz, vs.tt_transp, vs.transp * settings.h, vs.msa_rz, vs.alpha_transp)[2:-2, 2:-2, :],
    )
    vs.TT_q_rz = update(
        vs.TT_q_rz,
        at[2:-2, 2:-2, :], calc_TT_num(state, vs.SA_rz, vs.sas_params_q_rz, vs.q_rz * settings.h)[2:-2, 2:-2, :],
    )
    vs.tt_q_rz = update(
        vs.tt_q_rz,
        at[2:-2, 2:-2, :], npx.where(npx.diff(vs.TT_q_rz[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(vs.TT_q_rz[2:-2, 2:-2, :], axis=-1), 0),
    )
    vs.mtt_q_rz = update(
        vs.mtt_q_rz,
        at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_rz, vs.tt_q_rz, vs.q_rz * settings.h, vs.msa_rz, vs.alpha_q)[2:-2, 2:-2, :],
    )
    vs.TT_cpr_rz = update(
        vs.TT_cpr_rz,
        at[2:-2, 2:-2, :], calc_TT_num(state, vs.SA_ss, vs.sas_params_cpr_rz, vs.cpr_rz * settings.h)[2:-2, 2:-2, :],
    )
    vs.tt_cpr_rz = update(
        vs.tt_cpr_rz,
        at[2:-2, 2:-2, :], npx.where(npx.diff(vs.TT_cpr_rz[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(vs.TT_cpr_rz[2:-2, 2:-2, :], axis=-1), 0),
    )
    vs.mtt_cpr_rz = update(
        vs.mtt_cpr_rz,
        at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_ss, vs.tt_cpr_rz, vs.cpr_rz * settings.h, vs.msa_ss, vs.alpha_q)[2:-2, 2:-2, :],
    )
    vs.TT_q_ss = update(
        vs.TT_q_ss,
        at[2:-2, 2:-2, :], calc_TT_num(state, vs.SA_ss, vs.sas_params_q_ss, vs.q_ss * settings.h)[2:-2, 2:-2, :],
    )
    vs.tt_q_ss = update(
        vs.tt_q_ss,
        at[2:-2, 2:-2, :], npx.where(npx.diff(vs.TT_q_ss[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(vs.TT_q_ss[2:-2, 2:-2, :], axis=-1), 0),
    )
    vs.mtt_q_ss = update(
        vs.mtt_q_ss,
        at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_ss, vs.tt_q_ss, vs.q_ss * settings.h, vs.msa_ss, vs.alpha_q)[2:-2, 2:-2, :],
    )
    vs.TT_re_rg = update(
        vs.TT_re_rg,
        at[2:-2, 2:-2, :], calc_TT_num(state, vs.SA_ss, vs.sas_params_re_rg, vs.re_rg * settings.h)[2:-2, 2:-2, :],
    )
    vs.tt_re_rg = update(
        vs.tt_re_rg,
        at[2:-2, 2:-2, :], npx.where(npx.diff(vs.TT_re_rg[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(vs.TT_re_rg[2:-2, 2:-2, :], axis=-1), 0),
    )
    vs.mtt_re_rg = update(
        vs.mtt_re_rg,
        at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_ss, vs.tt_re_rg, vs.re_rg * settings.h, vs.msa_ss, vs.alpha_q)[2:-2, 2:-2, :],
    )
    vs.TT_re_rl = update(
        vs.TT_re_rl,
        at[2:-2, 2:-2, :], calc_TT_num(state, vs.SA_ss, vs.sas_params_re_rl, vs.re_rl * settings.h)[2:-2, 2:-2, :],
    )
    vs.tt_re_rl = update(
        vs.tt_re_rl,
        at[2:-2, 2:-2, :], npx.where(npx.diff(vs.TT_re_rl[2:-2, 2:-2, :], axis=-1) >= 0, npx.diff(vs.TT_re_rl[2:-2, 2:-2, :], axis=-1), 0),
    )
    vs.mtt_re_rl = update(
        vs.mtt_re_rl,
        at[2:-2, 2:-2, :], calc_mtt(state, vs.sa_ss, vs.tt_re_rl, vs.re_rl * settings.h, vs.msa_ss, vs.alpha_q)[2:-2, 2:-2, :],
    )

    # update StorAge
    if settings.enable_oxygen18 or settings.enable_deuterium:
        # impose nonegative constrain to numerical solution
        dsa_rz = (vs.re_rg[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rg[2:-2, 2:-2, :] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :] - vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] - vs.re_rl[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rl[2:-2, 2:-2, :]) * settings.h
        dsa_rz = npx.where(vs.sa_rz[2:-2, 2:-2, 1, :] + dsa_rz < 0, -vs.sa_rz[2:-2, 2:-2, 1, :], dsa_rz)
        dsa_ss = (vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] + vs.re_rl[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rl[2:-2, 2:-2, :] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :] - vs.re_rg[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rg[2:-2, 2:-2, :]) * settings.h
        dsa_ss = npx.where(vs.sa_ss[2:-2, 2:-2, 1, :] + dsa_ss < 0, -vs.sa_ss[2:-2, 2:-2, 1, :], dsa_ss)
        dsa_rz1 = (vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] + vs.re_rg[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rg[2:-2, 2:-2, :]) * settings.h
        dmsa_rz1 = npx.where(npx.isnan(vs.mtt_cpr_rz[2:-2, 2:-2, :]), 0, vs.mtt_cpr_rz[2:-2, 2:-2, :]) * npx.where(dsa_rz1 > 0, ((vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] * settings.h) / dsa_rz1), 0) + npx.where(npx.isnan(vs.mtt_re_rg[2:-2, 2:-2, :]), 0, vs.mtt_re_rg[2:-2, 2:-2, :]) * npx.where(dsa_rz1 > 0, ((vs.re_rg[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rg[2:-2, 2:-2, :] * settings.h) / dsa_rz1), 0)
        dsa_ss1 = (vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] + vs.re_rl[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rl[2:-2, 2:-2, :]) * settings.h
        dmsa_ss1 = npx.where(npx.isnan(vs.mtt_q_rz[2:-2, 2:-2, :]), 0, vs.mtt_q_rz[2:-2, 2:-2, :]) * npx.where(dsa_ss1 > 0, ((vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] * settings.h) / dsa_ss1), 0) + npx.where(npx.isnan(vs.mtt_re_rl[2:-2, 2:-2, :]), 0, vs.mtt_re_rl[2:-2, 2:-2, :]) * npx.where(dsa_ss1 > 0, ((vs.re_rl[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rl[2:-2, 2:-2, :] * settings.h) / dsa_ss1), 0)
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :] > 0) & (vs.msa_rz[2:-2, 2:-2, 1, :] > 0), vs.msa_rz[2:-2, 2:-2, 1, :] * (vs.sa_rz[2:-2, 2:-2, 1, :] / (dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :])), 0) + npx.where(dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :] > 0, dmsa_rz1 * (dsa_rz1 / (dsa_rz1 + vs.sa_rz[2:-2, 2:-2, 1, :])), 0),
        )
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_rz1 > 0) & (vs.msa_rz[2:-2, 2:-2, 1, :] <= 0), dmsa_rz1, vs.msa_rz[2:-2, 2:-2, 1, :]),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where(dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :] > 0, vs.msa_ss[2:-2, 2:-2, 1, :] * (vs.sa_ss[2:-2, 2:-2, 1, :] / (dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :])), 0) + npx.where(dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :] > 0, dmsa_ss1 * (dsa_ss1 / (dsa_ss1 + vs.sa_ss[2:-2, 2:-2, 1, :])), 0),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where((dsa_ss1 > 0) & (vs.msa_ss[2:-2, 2:-2, 1, :] <= 0), dmsa_ss1, vs.msa_ss[2:-2, 2:-2, 1, :]),
        )
        vs.sa_rz = update_add(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :], dsa_rz,
        )
        vs.sa_ss = update_add(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :], dsa_ss,
        )
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], npx.where(vs.sa_rz[2:-2, 2:-2, 1, :] <= 0, 0, vs.msa_rz[2:-2, 2:-2, 1, :]),
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], npx.where(vs.sa_ss[2:-2, 2:-2, 1, :] <= 0, 0, vs.msa_ss[2:-2, 2:-2, 1, :]),
        )
    elif settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate:
        # impose nonegative constrain of numerical solution
        dsa_rz = (vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] + vs.q_re_rg[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rg[2:-2, 2:-2, :] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :] - vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] - vs.q_re_rl[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rl[2:-2, 2:-2, :]) * settings.h
        dsa_rz = npx.where(vs.sa_rz[2:-2, 2:-2, 1, :] + dsa_rz < 0, -vs.sa_rz[2:-2, 2:-2, 1, :], dsa_rz)
        dsa_rz = npx.where(vs.sa_rz[2:-2, 2:-2, 1, :] + dsa_rz < 0, -vs.sa_rz[2:-2, 2:-2, 1, :], dsa_rz)
        dsa_ss = (vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] + vs.q_re_rl[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rl[2:-2, 2:-2, :] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :] - vs.q_re_rg[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rg[2:-2, 2:-2, :]) * settings.h
        dsa_ss = npx.where(vs.sa_ss[2:-2, 2:-2, 1, :] + dsa_ss < 0, -vs.sa_ss[2:-2, 2:-2, 1, :], dsa_ss)
        dsa_ss = npx.where(vs.sa_ss[2:-2, 2:-2, 1, :] + dsa_ss < 0, -vs.sa_ss[2:-2, 2:-2, 1, :], dsa_ss)
        vs.sa_rz = update_add(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :], dsa_rz,
        )
        vs.sa_ss = update_add(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :], dsa_ss,
        )
        dmsa_rz = npx.where(npx.isnan(vs.mtt_cpr_rz[2:-2, 2:-2, :]), 0, vs.mtt_cpr_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_re_rg[2:-2, 2:-2, :]), 0, vs.mtt_re_rg[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_transp[2:-2, 2:-2, :]), 0, vs.mtt_transp[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_q_rz[2:-2, 2:-2, :]), 0, vs.mtt_q_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_re_rl[2:-2, 2:-2, :]), 0, vs.mtt_re_rl[2:-2, 2:-2, :])
        dmsa_ss = npx.where(npx.isnan(vs.mtt_q_rz[2:-2, 2:-2, :]), 0, vs.mtt_q_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_re_rl[2:-2, 2:-2, :]), 0, vs.mtt_re_rl[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_cpr_rz[2:-2, 2:-2, :]), 0, vs.mtt_cpr_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_q_ss[2:-2, 2:-2, :]), 0, vs.mtt_q_ss[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_re_rg[2:-2, 2:-2, :]), 0, vs.mtt_re_rg[2:-2, 2:-2, :])
        dmsa_rz = npx.where(vs.msa_rz[2:-2, 2:-2, 1, :] + dmsa_rz < 0, 0, dmsa_rz)
        dmsa_ss = npx.where(vs.msa_ss[2:-2, 2:-2, 1, :] + dmsa_ss < 0, 0, dmsa_ss)
        vs.msa_rz = update_add(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], dmsa_rz,
        )
        vs.msa_ss = update_add(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], dmsa_ss,
        )
    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, 1, 1:], npx.cumsum(vs.sa_rz[2:-2, 2:-2, 1, :], axis=-1),
    )
    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, 1, 1:], npx.cumsum(vs.sa_ss[2:-2, 2:-2, 1, :], axis=-1),
    )

    # calculate solute concentrations
    if settings.enable_oxygen18 or settings.enable_deuterium:
        vs.C_inf_mat_rz = update(
            vs.C_inf_mat_rz,
            at[2:-2, 2:-2], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_inf_pf_rz = update(
            vs.C_inf_pf_rz,
            at[2:-2, 2:-2], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_inf_pf_ss = update(
            vs.C_inf_pf_ss,
            at[2:-2, 2:-2], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_evap_soil = update(
            vs.C_evap_soil,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_evap_soil, vs.tt_evap_soil, vs.evap_soil)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_transp = update(
            vs.C_transp,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_transp, vs.tt_transp, vs.transp)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_q_rz = update(
            vs.C_q_rz,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_q_rz, vs.tt_q_rz, vs.q_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_cpr_rz = update(
            vs.C_cpr_rz,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_cpr_rz, vs.tt_cpr_rz, vs.cpr_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_q_ss = update(
            vs.C_q_ss,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_q_ss, vs.tt_q_ss, vs.q_ss)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_re_rg = update(
            vs.C_re_rg,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_re_rg, vs.tt_re_rg, vs.re_rg)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_re_rl = update(
            vs.C_re_rl,
            at[2:-2, 2:-2], calc_conc_iso_flux(state, vs.mtt_re_rl, vs.tt_re_rl, vs.re_rl)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )

    elif settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate:
        vs.C_inf_mat_rz = update(
            vs.C_inf_mat_rz,
            at[2:-2, 2:-2], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_inf_pf_rz = update(
            vs.C_inf_pf_rz,
            at[2:-2, 2:-2], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_inf_pf_ss = update(
            vs.C_inf_pf_ss,
            at[2:-2, 2:-2], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, vs.C_in[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_transp = update(
            vs.C_transp,
            at[2:-2, 2:-2], npx.where(vs.transp > 0, npx.sum(vs.mtt_transp, axis=2) / vs.transp, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_q_rz = update(
            vs.C_q_rz,
            at[2:-2, 2:-2], npx.where(vs.q_rz > 0, npx.sum(vs.mtt_q_rz, axis=2) / vs.q_rz, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_cpr_rz = update(
            vs.C_cpr_rz,
            at[2:-2, 2:-2], npx.where(vs.cpr_rz > 0, npx.sum(vs.mtt_cpr_rz, axis=2) / vs.cpr_rz, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_q_ss = update(
            vs.C_q_ss,
            at[2:-2, 2:-2], npx.where(vs.q_ss > 0, npx.sum(vs.mtt_q_ss, axis=2) / vs.q_ss, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_re_rg = update(
            vs.C_re_rg,
            at[2:-2, 2:-2], npx.where(vs.re_rg > 0, npx.sum(vs.mtt_re_rg, axis=2) / vs.re_rg, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_re_rl = update(
            vs.C_re_rl,
            at[2:-2, 2:-2], npx.where(vs.re_rl > 0, npx.sum(vs.mtt_re_rl, axis=2) / vs.re_rl, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
        )

    return KernelOutput(tt_evap_soil=vs.tt_evap_soil, tt_transp=vs.tt_transp, tt_q_rz=vs.tt_q_rz, tt_cpr_rz=vs.tt_cpr_rz, tt_q_ss=vs.tt_q_ss, tt_re_rg=vs.tt_re_rg, tt_re_rl=vs.tt_re_rl,
                        TT_evap_soil=vs.TT_evap_soil, TT_transp=vs.TT_transp, TT_q_rz=vs.TT_q_rz, TT_cpr_rz=vs.TT_cpr_rz, TT_q_ss=vs.TT_q_ss, TT_re_rg=vs.TT_re_rg, TT_re_rl=vs.TT_re_rl,
                        mtt_evap_soil=vs.mtt_evap_soil, mtt_transp=vs.mtt_transp, mtt_q_rz=vs.mtt_q_rz, mtt_cpr_rz=vs.mtt_cpr_rz, mtt_q_ss=vs.mtt_q_ss, mtt_re_rg=vs.mtt_re_rg, mtt_re_rl=vs.mtt_re_rl,
                        C_inf_mat_rz=vs.C_inf_mat_rz, C_inf_pf_rz=vs.C_inf_pf_rz, C_inf_pf_ss=vs.C_inf_pf_ss, C_evap_soil=vs.C_evap_soil, C_transp=vs.C_transp, C_q_rz=vs.C_q_rz, C_cpr_rz=vs.C_cpr_rz, C_q_ss=vs.C_q_ss, C_re_rg=vs.C_re_rg, C_re_rl=vs.C_re_rl,
                        sa_rz=vs.sa_rz, sa_ss=vs.sa_ss, msa_rz=vs.msa_rz, msa_ss=vs.msa_ss)


@roger_routine
def calculate_storage_selection(state):
    """Calculates transport model
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_groundwater_boundary & settings.enable_crop_phenology) and not (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate):
        if settings.sas_solver == "Euler":
            # loop over substeps
            while vs.itt_substep < settings.sas_solver_substeps:
                vs.update(svat_transport_model_euler(state))
                root_zone.calculate_root_zone_transport(state)
                subsoil.calculate_subsoil_transport(state)
                soil.calculate_soil_transport(state)
                vs.time = vs.time + int(vs.dt_secs / settings.sas_solver_substeps)
                # collect data for output at end of substep
                with state.timers["diagnostics"]:
                    write_output(state)
                if (vs.time % (24 * 60 * 60) == 0):
                    calculate_ageing(state)
                    vs.update(after_substep_anion(state))
                else:
                    vs.update(after_substep_anion(state))
                vs.itt_substep = vs.itt_substep + 1
            vs.itt_substep = 0

        elif settings.sas_solver == "RK4":
            # loop over substeps
            while vs.itt_substep < settings.sas_solver_substeps:
                vs.update(svat_transport_model_rk4(state))
                root_zone.calculate_root_zone_transport(state)
                subsoil.calculate_subsoil_transport(state)
                soil.calculate_soil_transport(state)
                vs.time = vs.time + int(vs.dt_secs / settings.sas_solver_substeps)
                # collect data for output at end of substep
                with state.timers["diagnostics"]:
                    write_output(state)
                if (vs.time % (24 * 60 * 60) == 0):
                    calculate_ageing(state)
                    vs.update(after_substep_anion(state))
                else:
                    vs.update(after_substep_anion(state))
                vs.itt_substep = vs.itt_substep + 1
            vs.itt_substep = 0

        else:
            vs.update(svat_transport_model_deterministic(state))

    elif settings.enable_offline_transport and not (settings.enable_groundwater_boundary & settings.enable_crop_phenology) and (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate):
        if settings.sas_solver == "Euler":
            # loop over substeps
            while vs.itt_substep < settings.sas_solver_substeps:
                vs.update(svat_transport_model_euler(state))
                if (settings.enable_oxygen18 | settings.enable_deuterium):
                    # convert isotope concentrations to delta valus
                    vs.update(delta_fluxes_svat(state))
                root_zone.calculate_root_zone_transport(state)
                subsoil.calculate_subsoil_transport(state)
                soil.calculate_soil_transport(state)
                vs.time = vs.time + int(vs.dt_secs / settings.sas_solver_substeps)
                # collect data for output at end of substep
                with state.timers["diagnostics"]:
                    write_output(state)
                if (vs.time % (24 * 60 * 60) == 0):
                    calculate_ageing(state)
                    if (settings.enable_oxygen18 | settings.enable_deuterium):
                        vs.update(after_substep_iso(state))
                    else:
                        vs.update(after_substep_anion(state))
                else:
                    if (settings.enable_oxygen18 | settings.enable_deuterium):
                        vs.update(after_substep_iso(state))
                    else:
                        vs.update(after_substep_anion(state))
                vs.itt_substep = vs.itt_substep + 1
            vs.itt_substep = 0

        elif settings.sas_solver == "RK4":
            # loop over substeps
            while vs.itt_substep < settings.sas_solver_substeps:
                vs.update(svat_transport_model_rk4(state))
                if (settings.enable_oxygen18 | settings.enable_deuterium):
                    # convert isotope concentrations to delta valus
                    vs.update(delta_fluxes_svat(state))
                root_zone.calculate_root_zone_transport(state)
                subsoil.calculate_subsoil_transport(state)
                soil.calculate_soil_transport(state)
                vs.time = vs.time + int(vs.dt_secs / settings.sas_solver_substeps)
                # collect data for output at end of substep
                with state.timers["diagnostics"]:
                    write_output(state)
                if (vs.time % (24 * 60 * 60) == 0):
                    calculate_ageing(state)
                    if (settings.enable_oxygen18 | settings.enable_deuterium):
                        vs.update(after_substep_iso(state))
                    else:
                        vs.update(after_substep_anion(state))
                else:
                    if (settings.enable_oxygen18 | settings.enable_deuterium):
                        vs.update(after_substep_iso(state))
                    else:
                        vs.update(after_substep_anion(state))
                vs.itt_substep = vs.itt_substep + 1
            vs.itt_substep = 0

        else:
            vs.update(svat_transport_model_deterministic(state))

    elif settings.enable_offline_transport and settings.enable_groundwater_boundary and not settings.enable_crop_phenology and (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate):
        if settings.sas_solver == "Euler":
            # loop over substeps
            while vs.itt_substep < settings.sas_solver_substeps:
                vs.update(svat_lbc_transport_model_euler(state))
                if (settings.enable_oxygen18 | settings.enable_deuterium):
                    # convert isotope concentrations to delta valus
                    vs.update(delta_fluxes_svat(state))
                root_zone.calculate_root_zone_transport(state)
                subsoil.calculate_subsoil_transport(state)
                soil.calculate_soil_transport(state)
                vs.time = vs.time + int(vs.dt_secs / settings.sas_solver_substeps)
                # collect data for output at end of substep
                with state.timers["diagnostics"]:
                    write_output(state)
                if (vs.time % (24 * 60 * 60) == 0):
                    calculate_ageing(state)
                    if (settings.enable_oxygen18 | settings.enable_deuterium):
                        vs.update(after_substep_iso(state))
                    else:
                        vs.update(after_substep_anion(state))
                else:
                    if (settings.enable_oxygen18 | settings.enable_deuterium):
                        vs.update(after_substep_iso(state))
                    else:
                        vs.update(after_substep_anion(state))
                vs.itt_substep = vs.itt_substep + 1
            vs.itt_substep = 0

        elif settings.sas_solver == "RK4":
            pass

        else:
            vs.update(svat_lbc_transport_model_deterministic(state))

    elif settings.enable_offline_transport and not settings.enable_groundwater_boundary and settings.enable_crop_phenology and (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate):
        if settings.sas_solver == "Euler":
            # loop over substeps
            while vs.itt_substep < settings.sas_solver_substeps:
                vs.update(svat_crop_transport_model_euler(state))
                if (settings.enable_oxygen18 | settings.enable_deuterium):
                    # convert isotope concentrations to delta valus
                    vs.update(delta_fluxes_svat_crop(state))
                root_zone.calculate_root_zone_transport(state)
                subsoil.calculate_subsoil_transport(state)
                soil.calculate_soil_transport(state)
                vs.time = vs.time + int(vs.dt_secs / settings.sas_solver_substeps)
                # collect data for output at end of substep
                with state.timers["diagnostics"]:
                    write_output(state)
                if (vs.time % (24 * 60 * 60) == 0):
                    calculate_ageing(state)
                    if (settings.enable_oxygen18 | settings.enable_deuterium):
                        vs.update(after_substep_iso(state))
                    else:
                        vs.update(after_substep_anion(state))
                else:
                    if (settings.enable_oxygen18 | settings.enable_deuterium):
                        vs.update(after_substep_iso(state))
                    else:
                        vs.update(after_substep_anion(state))
                vs.itt_substep = vs.itt_substep + 1
            vs.itt_substep = 0

        elif settings.sas_solver == "RK4":
            pass

        else:
            vs.update(svat_crop_transport_model_deterministic(state))


@roger_routine
def write_output(state):
    """Collect and write output
    """
    vs = state.variables
    settings = state.settings

    if not numerics.sanity_check(state):
        logger.warning(f"Solution diverged at iteration {vs.itt} at substep {vs.itt_substep}.\n An evaluation of the bias of the deterministic/numerical\n solution is highly recommended. The bias is written to\n the model output.")
    numerics.calculate_num_error(state)

    if settings.warmup_done:
        diagnostics.diagnose(state)
        diagnostics.output(state)


@roger_kernel
def after_substep_iso(state):
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, vs.taum1, :], vs.SA_rz[2:-2, 2:-2, vs.tau, :],
    )
    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, vs.taum1, :], vs.sa_rz[2:-2, 2:-2, vs.tau, :],
    )
    vs.msa_rz = update(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.taum1, :], vs.msa_rz[2:-2, 2:-2, vs.tau, :],
    )
    vs.C_rz = update(
        vs.C_rz,
        at[2:-2, 2:-2, vs.taum1], vs.C_rz[2:-2, 2:-2, vs.tau],
    )
    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, vs.taum1, :], vs.SA_ss[2:-2, 2:-2, vs.tau, :],
    )
    vs.sa_ss = update(
        vs.sa_ss,
        at[2:-2, 2:-2, vs.taum1, :], vs.sa_ss[2:-2, 2:-2, vs.tau, :],
    )
    vs.msa_ss = update(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.taum1, :], vs.msa_ss[2:-2, 2:-2, vs.tau, :],
    )
    vs.C_ss = update(
        vs.C_ss,
        at[2:-2, 2:-2, vs.taum1], vs.C_ss[2:-2, 2:-2, vs.tau],
    )
    vs.SA_s = update(
        vs.SA_s,
        at[2:-2, 2:-2, vs.taum1, :], vs.SA_s[2:-2, 2:-2, vs.tau, :],
    )
    vs.sa_s = update(
        vs.sa_s,
        at[2:-2, 2:-2, vs.taum1, :], vs.sa_s[2:-2, 2:-2, vs.tau, :],
    )
    vs.msa_s = update(
        vs.msa_s,
        at[2:-2, 2:-2, vs.taum1, :], vs.msa_s[2:-2, 2:-2, vs.tau, :],
    )
    vs.C_s = update(
        vs.C_s,
        at[2:-2, 2:-2, vs.taum1], vs.C_s[2:-2, 2:-2, vs.tau],
    )
    vs.C_iso_rz = update(
        vs.C_iso_rz,
        at[2:-2, 2:-2, vs.taum1], vs.C_iso_rz[2:-2, 2:-2, vs.tau],
    )
    vs.C_iso_ss = update(
        vs.C_iso_ss,
        at[2:-2, 2:-2, vs.taum1], vs.C_iso_ss[2:-2, 2:-2, vs.tau],
    )
    vs.C_iso_s = update(
        vs.C_iso_s,
        at[2:-2, 2:-2, vs.taum1], vs.C_iso_s[2:-2, 2:-2, vs.tau],
    )
    vs.C_iso_snow = update(
        vs.C_iso_snow,
        at[2:-2, 2:-2, vs.taum1], vs.C_iso_snow[2:-2, 2:-2, vs.tau],
    )
    vs.csa_rz = update(
        vs.csa_rz,
        at[2:-2, 2:-2, vs.taum1, :], vs.csa_rz[2:-2, 2:-2, vs.tau, :],
    )
    vs.csa_ss = update(
        vs.csa_ss,
        at[2:-2, 2:-2, vs.taum1, :], vs.csa_ss[2:-2, 2:-2, vs.tau, :],
    )
    vs.csa_s = update(
        vs.csa_s,
        at[2:-2, 2:-2, vs.taum1, :], vs.csa_s[2:-2, 2:-2, vs.tau, :],
    )

    return KernelOutput(
        SA_rz=vs.SA_rz,
        sa_rz=vs.sa_rz,
        msa_rz=vs.msa_rz,
        C_rz=vs.C_rz,
        SA_ss=vs.SA_ss,
        sa_ss=vs.sa_ss,
        msa_ss=vs.msa_ss,
        C_ss=vs.C_ss,
        SA_s=vs.SA_s,
        sa_s=vs.sa_s,
        msa_s=vs.msa_s,
        C_s=vs.C_s,
        C_iso_rz=vs.C_iso_rz,
        C_iso_ss=vs.C_iso_ss,
        C_iso_s=vs.C_iso_s,
        csa_s=vs.csa_s,
        csa_rz=vs.csa_rz,
        csa_ss=vs.csa_ss,
        )


@roger_kernel
def after_substep_anion(state):
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, vs.taum1, :], vs.SA_rz[2:-2, 2:-2, vs.tau, :],
    )
    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, vs.taum1, :], vs.sa_rz[2:-2, 2:-2, vs.tau, :],
    )
    vs.msa_rz = update(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.taum1, :], vs.msa_rz[2:-2, 2:-2, vs.tau, :],
    )
    vs.M_rz = update(
        vs.M_rz,
        at[2:-2, 2:-2, vs.taum1], vs.M_rz[2:-2, 2:-2, vs.tau],
    )
    vs.C_rz = update(
        vs.C_rz,
        at[2:-2, 2:-2, vs.taum1], vs.C_rz[2:-2, 2:-2, vs.tau],
    )
    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, vs.taum1, :], vs.SA_ss[2:-2, 2:-2, vs.tau, :],
    )
    vs.sa_ss = update(
        vs.sa_ss,
        at[2:-2, 2:-2, vs.taum1, :], vs.sa_ss[2:-2, 2:-2, vs.tau, :],
    )
    vs.msa_ss = update(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.taum1, :], vs.msa_ss[2:-2, 2:-2, vs.tau, :],
    )
    vs.M_ss = update(
        vs.M_ss,
        at[2:-2, 2:-2, vs.taum1], vs.M_ss[2:-2, 2:-2, vs.tau],
    )
    vs.C_ss = update(
        vs.C_ss,
        at[2:-2, 2:-2, vs.taum1], vs.C_ss[2:-2, 2:-2, vs.tau],
    )
    vs.SA_s = update(
        vs.SA_s,
        at[2:-2, 2:-2, vs.taum1, :], vs.SA_s[2:-2, 2:-2, vs.tau, :],
    )
    vs.sa_s = update(
        vs.sa_s,
        at[2:-2, 2:-2, vs.taum1, :], vs.sa_s[2:-2, 2:-2, vs.tau, :],
    )
    vs.msa_s = update(
        vs.msa_s,
        at[2:-2, 2:-2, vs.taum1, :], vs.msa_s[2:-2, 2:-2, vs.tau, :],
    )
    vs.M_s = update(
        vs.M_s,
        at[2:-2, 2:-2, vs.taum1], vs.M_s[2:-2, 2:-2, vs.tau],
    )
    vs.C_s = update(
        vs.C_s,
        at[2:-2, 2:-2, vs.taum1], vs.C_s[2:-2, 2:-2, vs.tau],
    )
    vs.csa_rz = update(
        vs.csa_rz,
        at[2:-2, 2:-2, vs.taum1, :], vs.csa_rz[2:-2, 2:-2, vs.tau, :],
    )
    vs.csa_ss = update(
        vs.csa_ss,
        at[2:-2, 2:-2, vs.taum1, :], vs.csa_ss[2:-2, 2:-2, vs.tau, :],
    )
    vs.csa_s = update(
        vs.csa_s,
        at[2:-2, 2:-2, vs.taum1, :], vs.csa_s[2:-2, 2:-2, vs.tau, :],
    )

    return KernelOutput(
        SA_rz=vs.SA_rz,
        sa_rz=vs.sa_rz,
        msa_rz=vs.msa_rz,
        M_rz=vs.M_rz,
        C_rz=vs.C_rz,
        SA_ss=vs.SA_ss,
        sa_ss=vs.sa_ss,
        msa_ss=vs.msa_ss,
        M_ss=vs.M_ss,
        C_ss=vs.C_ss,
        SA_s=vs.SA_s,
        sa_s=vs.sa_s,
        msa_s=vs.msa_s,
        M_s=vs.M_s,
        C_s=vs.C_s,
        csa_s=vs.csa_s,
        csa_rz=vs.csa_rz,
        csa_ss=vs.csa_ss,
        )


@roger_kernel
def delta_fluxes_svat(state):
    vs = state.variables

    vs.C_iso_inf_mat_rz = update(
        vs.C_iso_inf_mat_rz,
        at[2:-2, 2:-2], conc_to_delta(state, vs.C_inf_mat_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_inf_pf_rz = update(
        vs.C_iso_inf_pf_rz,
        at[2:-2, 2:-2], conc_to_delta(state, vs.C_inf_pf_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_inf_pf_ss = update(
        vs.C_iso_inf_pf_ss,
        at[2:-2, 2:-2], conc_to_delta(state, vs.C_inf_pf_ss)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_evap_soil = update(
        vs.C_iso_evap_soil,
        at[2:-2, 2:-2], conc_to_delta(state, vs.C_evap_soil)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_transp = update(
        vs.C_iso_transp,
        at[2:-2, 2:-2], conc_to_delta(state, vs.C_transp)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_cpr_rz = update(
        vs.C_iso_cpr_rz,
        at[2:-2, 2:-2], conc_to_delta(state, vs.C_cpr_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_q_rz = update(
        vs.C_iso_q_rz,
        at[2:-2, 2:-2], conc_to_delta(state, vs.C_q_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_q_ss = update(
        vs.C_iso_q_ss,
        at[2:-2, 2:-2], conc_to_delta(state, vs.C_q_ss)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(C_iso_inf_mat_rz=vs.C_iso_inf_mat_rz, C_iso_inf_pf_rz=vs.C_iso_inf_pf_rz, C_iso_inf_pf_ss=vs.C_iso_inf_pf_ss, C_iso_evap_soil=vs.C_iso_evap_soil, C_iso_transp=vs.C_iso_transp, C_iso_q_rz=vs.C_iso_q_rz, C_iso_cpr_rz=vs.C_iso_cpr_rz, C_iso_q_ss=vs.C_iso_q_ss)


@roger_kernel
def delta_fluxes_svat_crop(state):
    vs = state.variables

    vs.C_iso_inf_mat_rz = update(
        vs.C_iso_inf_mat_rz,
        at[2:-2, 2:-2], conc_to_delta(state, vs.C_inf_mat_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_inf_pf_rz = update(
        vs.C_iso_inf_pf_rz,
        at[2:-2, 2:-2], conc_to_delta(state, vs.C_inf_pf_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_inf_pf_ss = update(
        vs.C_iso_inf_pf_ss,
        at[2:-2, 2:-2], conc_to_delta(state, vs.C_inf_pf_ss)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_evap_soil = update(
        vs.C_iso_evap_soil,
        at[2:-2, 2:-2], conc_to_delta(state, vs.C_evap_soil)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_transp = update(
        vs.C_iso_transp,
        at[2:-2, 2:-2], conc_to_delta(state, vs.C_transp)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_cpr_rz = update(
        vs.C_iso_cpr_rz,
        at[2:-2, 2:-2], conc_to_delta(state, vs.C_cpr_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_q_rz = update(
        vs.C_iso_q_rz,
        at[2:-2, 2:-2], conc_to_delta(state, vs.C_q_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_q_ss = update(
        vs.C_iso_q_ss,
        at[2:-2, 2:-2], conc_to_delta(state, vs.C_q_ss)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_re_rg = update(
        vs.C_iso_re_rg,
        at[2:-2, 2:-2], conc_to_delta(state, vs.C_re_rg)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_re_rl = update(
        vs.C_iso_re_rl,
        at[2:-2, 2:-2], conc_to_delta(state, vs.C_re_rl)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(C_iso_inf_mat_rz=vs.C_iso_inf_mat_rz, C_iso_inf_pf_rz=vs.C_iso_inf_pf_rz, C_iso_inf_pf_ss=vs.C_iso_inf_pf_ss, C_iso_evap_soil=vs.C_iso_evap_soil, C_iso_transp=vs.C_iso_transp, C_iso_q_rz=vs.C_iso_q_rz, C_iso_cpr_rz=vs.C_iso_cpr_rz, C_iso_q_ss=vs.C_iso_q_ss, C_iso_re_rg=vs.C_iso_re_rg, C_iso_re_rl=vs.C_iso_re_rl)
