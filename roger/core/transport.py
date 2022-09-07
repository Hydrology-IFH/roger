from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at, scan
from roger.core import sas
from roger import runtime_settings as rs, logger


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
def calc_MSA(state, MSA, msa):
    """Calculate cumulative solute mass StorAge (residence time distribution).
    """
    vs = state.variables

    # cumulative StorAge (residence time distribution)
    MSA = update(
        MSA,
        at[2:-2, 2:-2, vs.tau, 1:], npx.cumsum(msa[2:-2, 2:-2, vs.tau, :], axis=-1),
    )
    MSA = update(
        MSA,
        at[2:-2, 2:-2, vs.tau, 0], 0,
    )

    return MSA


@roger_kernel
def calc_tt(state, SA, sa, flux, sas_params):
    """Calculates backward travel time distribution.
    """
    settings = state.settings

    if settings.sas_solver == "RK4":
        tt = calc_tt_rk4(state, SA, sa, flux, sas_params)

    else:
        tt = calc_tt_deterministic(state, SA, sa, flux, sas_params)

    return tt


@roger_kernel
def calc_tt_deterministic(state, SA, sa, flux, sas_params):
    """Calculates backward travel time distribution.
    """
    vs = state.variables

    TT = allocate(state.dimensions, ("x", "y", "nages"))
    tt = allocate(state.dimensions, ("x", "y", "ages"))
    # cumulative backward travel time distribution
    TT = update_add(
        TT,
        at[2:-2, 2:-2, :], sas.uniform(state, SA, sas_params)[2:-2, 2:-2, :],
    )
    TT = update_add(
        TT,
        at[2:-2, 2:-2, :], sas.dirac(state, sas_params)[2:-2, 2:-2, :],
    )
    Omega, sas_params = sas.kumaraswami(state, SA, sas_params)
    TT = update_add(
        TT,
        at[2:-2, 2:-2, :], Omega[2:-2, 2:-2, :],
    )
    TT = update_add(
        TT,
        at[2:-2, 2:-2, :], sas.gamma(state, SA, sas_params)[2:-2, 2:-2, :],
    )
    TT = update_add(
        TT,
        at[2:-2, 2:-2, :], sas.exponential(state, SA, sas_params)[2:-2, 2:-2, :],
    )
    TT = update_add(
        TT,
        at[2:-2, 2:-2, :], sas.power(state, SA, sas_params)[2:-2, 2:-2, :],
    )

    # travel time distribution
    mask_old = npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([1, 22, 32, 34, 52, 62])) | (npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([3, 35])) & (sas_params[:, :, 1, npx.newaxis] >= sas_params[:, :, 2, npx.newaxis]))
    mask_young = npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([21, 31, 33, 4, 41, 51, 61])) | (npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([3, 35])) & (sas_params[:, :, 1, npx.newaxis] < sas_params[:, :, 2, npx.newaxis]))

    tt = update(
        tt,
        at[2:-2, 2:-2, :], npx.diff(TT[2:-2, 2:-2, :], axis=-1),
    )
    # set travel time distribution for dirac function
    tt = update(
        tt,
        at[2:-2, 2:-2, :], npx.where(sas_params[2:-2, 2:-2, 0, npx.newaxis] == 21, 0, tt[2:-2, 2:-2, :]),
    )
    tt = update(
        tt,
        at[2:-2, 2:-2, 0], npx.where(sas_params[2:-2, 2:-2, 0] == 21, 1, tt[2:-2, 2:-2, 0]),
    )
    tt = update(
        tt,
        at[2:-2, 2:-2, :], npx.where(sas_params[2:-2, 2:-2, 0, npx.newaxis] == 22, 0, tt[2:-2, 2:-2, :]),
    )
    tt = update(
        tt,
        at[2:-2, 2:-2, -1], npx.where(sas_params[2:-2, 2:-2, 0] == 22, 1, tt[2:-2, 2:-2, -1]),
    )
    # outflux
    flux_tt = allocate(state.dimensions, ("x", "y", "ages"))
    flux_tt = update(
        flux_tt,
        at[2:-2, 2:-2, :], flux[2:-2, 2:-2, npx.newaxis] * tt[2:-2, 2:-2, :],
    )

    # set travel time distribution to zero if no outflux
    tt = update(
        tt,
        at[2:-2, 2:-2, :], npx.where(flux_tt[2:-2, 2:-2, :] > 0, tt[2:-2, 2:-2, :], 0),
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
        at[2:-2, 2:-2, :], flux_tt[2:-2, 2:-2, :],
    )
    diff_q = update(
        diff_q,
        at[2:-2, 2:-2, :], flux_tt[2:-2, 2:-2, :] - sa[2:-2, 2:-2, vs.tau, :],
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
        at[2:-2, 2:-2, :], npx.where((sa[2:-2, 2:-2, vs.tau, :] - flux_tt[2:-2, 2:-2, :] > 0), sa[2:-2, 2:-2, vs.tau, :] - flux_tt[2:-2, 2:-2, :], 0),
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
    flux_tt_re = update(
        flux_tt_re,
        at[2:-2, 2:-2, :], q_re_sum[2:-2, 2:-2, npx.newaxis] * omega_re[2:-2, 2:-2, :],
    )
    flux_tt = update(
        flux_tt,
        at[2:-2, 2:-2, :], npx.where((flux_tt_re[2:-2, 2:-2, :] > 0), flux_tt_re[2:-2, 2:-2, :] + flux_tt[2:-2, 2:-2, :], flux_tt[2:-2, 2:-2, :]),
    )
    flux_tt = update(
        flux_tt,
        at[2:-2, 2:-2, :], npx.where((diff_q[2:-2, 2:-2, :] > 0), flux_tt[2:-2, 2:-2, :] - diff_q[2:-2, 2:-2, :], flux_tt[2:-2, 2:-2, :]),
    )
    # recalculate travel time distribution
    tt = update(
        tt,
        at[2:-2, 2:-2, :], npx.where(npx.any(flux_tt_init[2:-2, 2:-2, :] > sa[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis], flux_tt[2:-2, 2:-2, :]/flux[2:-2, 2:-2, npx.newaxis], tt[2:-2, 2:-2, :]),
    )

    # set to zero if zero flux
    tt = update(
        tt,
        at[2:-2, 2:-2, :], npx.where(flux[2:-2, 2:-2, npx.newaxis] <= 0, 0, tt[2:-2, 2:-2, :]),
    )

    if rs.backend == 'numpy':
        # sanity check of SAS function (works only for numpy backend)
        mask = npx.isclose(npx.sum(tt, axis=-1) * flux, flux, atol=1e-02)
        if not npx.all(mask[2:-2, 2:-2]):
            if rs.loglevel == 'debug':
                logger.debug(f"Solution of SAS function diverged at iteration {vs.itt}")
            else:
                raise RuntimeError(f"Solution of SAS function diverged at iteration {vs.itt}")
        mask1 = (tt * flux[:, :, npx.newaxis] - sa[:, :, 1, :] > 1e-02)
        if npx.any(mask1[2:-2, 2:-2, :]):
            if rs.loglevel == 'debug':
                logger.debug(f"Solution of SAS function diverged at iteration {vs.itt}")
            else:
                raise RuntimeError(f"Solution of SAS function diverged at iteration {vs.itt}")
        if rs.loglevel == 'debug':
            rows = npx.where(mask[2:-2, 2:-2] == False)[0].tolist()
            if rows:
                logger.debug(f"Solution of SAS function diverged at {rows}")

    return tt


@roger_kernel
def calc_tt_rk4(state, SA, sa, flux, sas_params):
    """Calculates backward travel time distribution with Runge-Kutta method
    """
    vs = state.variables
    settings = state.settings

    if vs.itt >= 231:
        print('')

    dt_num = 1 / settings.sas_solver_substeps

    SAn = allocate(state.dimensions, ("x", "y", "timesteps", "nages"))
    san = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
    SAn = update(
        SAn,
        at[2:-2, 2:-2, :, :], SA[2:-2, 2:-2, :, :],
    )
    san = update(
        san,
        at[2:-2, 2:-2, :, :], sa[2:-2, 2:-2, :, :],
    )

    TT1 = allocate(state.dimensions, ("x", "y", "nages"))
    tt1 = allocate(state.dimensions, ("x", "y", "ages"))
    TT2 = allocate(state.dimensions, ("x", "y", "nages"))
    tt2 = allocate(state.dimensions, ("x", "y", "ages"))
    TT3 = allocate(state.dimensions, ("x", "y", "nages"))
    tt3 = allocate(state.dimensions, ("x", "y", "ages"))
    TT4 = allocate(state.dimensions, ("x", "y", "nages"))
    tt4 = allocate(state.dimensions, ("x", "y", "ages"))
    tt = allocate(state.dimensions, ("x", "y", "ages"))
    ttn = allocate(state.dimensions, ("x", "y", "ages"))

    def body(carry, x):
        state, sas_params, SAn, san, TT1, tt1, TT2, tt2, TT3, tt3, TT4, tt4, ttn, tt, dt_num = carry
        # cumulative backward travel time distribution
        TT1 = update(
            TT1,
            at[2:-2, 2:-2, :], 0,
        )
        TT1 = update_add(
            TT1,
            at[2:-2, 2:-2, :], sas.uniform(state, SAn, sas_params)[2:-2, 2:-2, :],
        )
        TT1 = update_add(
            TT1,
            at[2:-2, 2:-2, :], sas.dirac(state, sas_params)[2:-2, 2:-2, :],
        )
        TT1 = update(
            TT1,
            at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 21) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(flux[2:-2, 2:-2, npx.newaxis] * dt_num/2 <= SAn[2:-2, 2:-2, 1, :], SAn[2:-2, 2:-2, 1, :]/flux[2:-2, 2:-2, npx.newaxis], 0), TT1[2:-2, 2:-2, :]),
        )
        TT1 = update(
            TT1,
            at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 21) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(npx.max(TT1[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis] > 1, TT1[2:-2, 2:-2, :] * (1/npx.max(TT1[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis]), TT1[2:-2, 2:-2, :]), TT1[2:-2, 2:-2, :]),
        )
        TT1 = update(
            TT1,
            at[2:-2, 2:-2, 1:], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 22) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(flux[2:-2, 2:-2, npx.newaxis] * dt_num/2 <= npx.cumsum(san[2:-2, 2:-2, 1, ::-1], axis=-1)[:, :, ::-1], npx.cumsum(san[2:-2, 2:-2, 1, ::-1], axis=-1)[:, :, ::-1]/flux[2:-2, 2:-2, npx.newaxis], 0), TT1[2:-2, 2:-2, 1:]),
        )
        TT1 = update(
            TT1,
            at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 22) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(npx.max(TT1[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis] > 1, TT1[2:-2, 2:-2, :] * (1/npx.max(TT1[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis]), TT1[2:-2, 2:-2, :]), TT1[2:-2, 2:-2, :]),
        )
        Omega, sas_params = sas.kumaraswami(state, SAn, sas_params)
        TT1 = update_add(
            TT1,
            at[2:-2, 2:-2, :], Omega[2:-2, 2:-2, :],
        )
        TT1 = update_add(
            TT1,
            at[2:-2, 2:-2, :], sas.gamma(state, SAn, sas_params)[2:-2, 2:-2, :],
        )
        TT1 = update_add(
            TT1,
            at[2:-2, 2:-2, :], sas.exponential(state, SAn, sas_params)[2:-2, 2:-2, :],
        )
        TT1 = update_add(
            TT1,
            at[2:-2, 2:-2, :], sas.power(state, SAn, sas_params)[2:-2, 2:-2, :],
        )
        tt1 = update(
            tt1,
            at[2:-2, 2:-2, :], npx.diff(TT1[2:-2, 2:-2, :], axis=-1),
        )
        san = update_add(
            san,
            at[2:-2, 2:-2, 1, :], -flux[2:-2, 2:-2, npx.newaxis] * tt1[2:-2, 2:-2, :] * dt_num/2,
        )
        SAn = update(
            SAn,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(san[2:-2, 2:-2, 1, :], axis=-1),
        )

        TT2 = update(
            TT2,
            at[2:-2, 2:-2, :], 0,
        )
        TT2 = update_add(
            TT2,
            at[2:-2, 2:-2, :], sas.uniform(state, SAn, sas_params)[2:-2, 2:-2, :],
        )
        TT2 = update_add(
            TT2,
            at[2:-2, 2:-2, :], sas.dirac(state, sas_params)[2:-2, 2:-2, :],
        )
        TT2 = update(
            TT2,
            at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 21) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(flux[2:-2, 2:-2, npx.newaxis] * dt_num/2 <= SAn[2:-2, 2:-2, 1, :], SAn[2:-2, 2:-2, 1, :]/flux[2:-2, 2:-2, npx.newaxis], 0), TT2[2:-2, 2:-2, :]),
        )
        TT2 = update(
            TT2,
            at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 21) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(npx.max(TT2[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis] > 1, TT2[2:-2, 2:-2, :] * (1/npx.max(TT2[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis]), TT2[2:-2, 2:-2, :]), TT2[2:-2, 2:-2, :]),
        )
        TT2 = update(
            TT2,
            at[2:-2, 2:-2, 1:], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 22) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(flux[2:-2, 2:-2, npx.newaxis] * dt_num/2 <= npx.cumsum(san[2:-2, 2:-2, 1, ::-1], axis=-1)[:, :, ::-1], npx.cumsum(san[2:-2, 2:-2, 1, ::-1], axis=-1)[:, :, ::-1]/flux[2:-2, 2:-2, npx.newaxis], 0), TT2[2:-2, 2:-2, 1:]),
        )
        TT2 = update(
            TT2,
            at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 22) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(npx.max(TT2[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis] > 1, TT2[2:-2, 2:-2, :] * (1/npx.max(TT2[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis]), TT2[2:-2, 2:-2, :]), TT2[2:-2, 2:-2, :]),
        )
        Omega, sas_params = sas.kumaraswami(state, SAn, sas_params)
        TT2 = update_add(
            TT2,
            at[2:-2, 2:-2, :], Omega[2:-2, 2:-2, :],
        )
        TT2 = update_add(
            TT2,
            at[2:-2, 2:-2, :], sas.gamma(state, SAn, sas_params)[2:-2, 2:-2, :],
        )
        TT2 = update_add(
            TT2,
            at[2:-2, 2:-2, :], sas.exponential(state, SAn, sas_params)[2:-2, 2:-2, :],
        )
        TT2 = update_add(
            TT2,
            at[2:-2, 2:-2, :], sas.power(state, SAn, sas_params)[2:-2, 2:-2, :],
        )
        tt2 = update(
            tt2,
            at[2:-2, 2:-2, :], npx.diff(TT2[2:-2, 2:-2, :], axis=-1),
        )
        san = update_add(
            san,
            at[2:-2, 2:-2, 1, :], -flux[2:-2, 2:-2, npx.newaxis] * tt2[2:-2, 2:-2, :] * dt_num/2,
        )
        SAn = update(
            SAn,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(san[2:-2, 2:-2, 1, :], axis=-1),
        )

        TT3 = update(
            TT3,
            at[2:-2, 2:-2, :], 0,
        )
        TT3 = update_add(
            TT3,
            at[2:-2, 2:-2, :], sas.uniform(state, SAn, sas_params)[2:-2, 2:-2, :],
        )
        TT3 = update_add(
            TT3,
            at[2:-2, 2:-2, :], sas.dirac(state, sas_params)[2:-2, 2:-2, :],
        )
        TT3 = update(
            TT3,
            at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 21) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(flux[2:-2, 2:-2, npx.newaxis] * dt_num/2 <= SAn[2:-2, 2:-2, 1, :], SAn[2:-2, 2:-2, 1, :]/flux[2:-2, 2:-2, npx.newaxis], 0), TT3[2:-2, 2:-2, :]),
        )
        TT3 = update(
            TT3,
            at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 21) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(npx.max(TT3[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis] > 1, TT3[2:-2, 2:-2, :] * (1/npx.max(TT3[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis]), TT3[2:-2, 2:-2, :]), TT3[2:-2, 2:-2, :]),
        )
        TT3 = update(
            TT3,
            at[2:-2, 2:-2, 1:], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 22) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(flux[2:-2, 2:-2, npx.newaxis] * dt_num/2 <= npx.cumsum(san[2:-2, 2:-2, 1, ::-1], axis=-1)[:, :, ::-1], npx.cumsum(san[2:-2, 2:-2, 1, ::-1], axis=-1)[:, :, ::-1]/flux[2:-2, 2:-2, npx.newaxis], 0), TT3[2:-2, 2:-2, 1:]),
        )
        TT3 = update(
            TT3,
            at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 22) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(npx.max(TT3[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis] > 1, TT3[2:-2, 2:-2, :] * (1/npx.max(TT3[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis]), TT3[2:-2, 2:-2, :]), TT3[2:-2, 2:-2, :]),
        )
        Omega, sas_params = sas.kumaraswami(state, SAn, sas_params)
        TT3 = update_add(
            TT3,
            at[2:-2, 2:-2, :], Omega[2:-2, 2:-2, :],
        )
        TT3 = update_add(
            TT3,
            at[2:-2, 2:-2, :], sas.gamma(state, SAn, sas_params)[2:-2, 2:-2, :],
        )
        TT3 = update_add(
            TT3,
            at[2:-2, 2:-2, :], sas.exponential(state, SAn, sas_params)[2:-2, 2:-2, :],
        )
        TT3 = update_add(
            TT3,
            at[2:-2, 2:-2, :], sas.power(state, SAn, sas_params)[2:-2, 2:-2, :],
        )
        tt3 = update(
            tt3,
            at[2:-2, 2:-2, :], npx.diff(TT3[2:-2, 2:-2, :], axis=-1),
        )
        san = update_add(
            san,
            at[2:-2, 2:-2, 1, :], -flux[2:-2, 2:-2, npx.newaxis] * tt1[2:-2, 2:-2, :],
        )
        SAn = update(
            SAn,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(san[2:-2, 2:-2, 1, :], axis=-1),
        )

        TT4 = update(
            TT4,
            at[2:-2, 2:-2, :], 0,
        )
        TT4 = update_add(
            TT4,
            at[2:-2, 2:-2, :], sas.uniform(state, SAn, sas_params)[2:-2, 2:-2, :],
        )
        TT4 = update_add(
            TT4,
            at[2:-2, 2:-2, :], sas.dirac(state, sas_params)[2:-2, 2:-2, :],
        )
        TT4 = update(
            TT4,
            at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 21) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(flux[2:-2, 2:-2, npx.newaxis] * dt_num/2 <= SAn[2:-2, 2:-2, 1, :], SAn[2:-2, 2:-2, 1, :]/flux[2:-2, 2:-2, npx.newaxis], 0), TT4[2:-2, 2:-2, :]),
        )
        TT4 = update(
            TT4,
            at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 21) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(npx.max(TT4[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis] > 1, TT4[2:-2, 2:-2, :] * (1/npx.max(TT4[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis]), TT4[2:-2, 2:-2, :]), TT4[2:-2, 2:-2, :]),
        )
        TT4 = update(
            TT4,
            at[2:-2, 2:-2, 1:], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 22) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(flux[2:-2, 2:-2, npx.newaxis] * dt_num/2 <= npx.cumsum(san[2:-2, 2:-2, 1, ::-1], axis=-1)[:, :, ::-1], npx.cumsum(san[2:-2, 2:-2, 1, ::-1], axis=-1)[:, :, ::-1]/flux[2:-2, 2:-2, npx.newaxis], 0), TT4[2:-2, 2:-2, 1:]),
        )
        TT4 = update(
            TT4,
            at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 22) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(npx.max(TT4[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis] > 1, TT4[2:-2, 2:-2, :] * (1/npx.max(TT4[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis]), TT4[2:-2, 2:-2, :]), TT4[2:-2, 2:-2, :]),
        )
        Omega, sas_params = sas.kumaraswami(state, SAn, sas_params)
        TT4 = update_add(
            TT4,
            at[2:-2, 2:-2, :], Omega[2:-2, 2:-2, :],
        )
        TT4 = update_add(
            TT4,
            at[2:-2, 2:-2, :], sas.gamma(state, SAn, sas_params)[2:-2, 2:-2, :],
        )
        TT4 = update_add(
            TT4,
            at[2:-2, 2:-2, :], sas.exponential(state, SAn, sas_params)[2:-2, 2:-2, :],
        )
        TT4 = update_add(
            TT4,
            at[2:-2, 2:-2, :], sas.power(state, SAn, sas_params)[2:-2, 2:-2, :],
        )
        tt4 = update(
            tt4,
            at[2:-2, 2:-2, :], npx.diff(TT4[2:-2, 2:-2, :], axis=-1),
        )
        ttn = update(
            ttn,
            at[2:-2, 2:-2, :], (tt1[2:-2, 2:-2, :] + 2*tt2[2:-2, 2:-2, :] + 2*tt3[2:-2, 2:-2, :] + tt4[2:-2, 2:-2, :]) / 6.,
        )
        san = update_add(
            san,
            at[2:-2, 2:-2, 1, :], -flux[2:-2, 2:-2, npx.newaxis] * ttn[2:-2, 2:-2, :],
        )
        SAn = update(
            SAn,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(san[2:-2, 2:-2, 1, :], axis=-1),
        )
        tt = update_add(
            tt,
            at[2:-2, 2:-2, :], (tt1[2:-2, 2:-2, :] + 2*tt2[2:-2, 2:-2, :] + 2*tt3[2:-2, 2:-2, :] + tt4[2:-2, 2:-2, :]) / 6. / settings.sas_solver_substeps,
        )

        carry = (state, sas_params, SAn, san, TT1, tt1, TT2, tt2, TT3, tt3, TT4, tt4, ttn, tt, dt_num)

        return carry, None

    n_substeps = npx.arange(0, settings.sas_solver_substeps)
    carry = (state, sas_params, SAn, san, TT1, tt1, TT2, tt2, TT3, tt3, TT4, tt4, ttn, tt, dt_num)
    res, _ = scan(body, carry, n_substeps)
    tt = res[13]

    diff1 = npx.sum(tt) * npx.sum(flux) - npx.sum(flux)
    diff2 = npx.sum(npx.where(tt * flux[:, :, npx.newaxis] > sa[:, :, 1, :], tt * flux[:, :, npx.newaxis] - sa[:, :, 1, :], 0))

    if rs.backend == 'numpy':
        # sanity check of SAS function (works only for numpy backend)
        mask = npx.isclose(npx.sum(tt, axis=-1) * flux, flux, atol=1e-02)
        if not npx.all(mask[2:-2, 2:-2]):
            if rs.loglevel == 'debug':
                logger.debug(f"1Solution of SAS function diverged at iteration {vs.itt}")
                logger.debug(f"Difference: {diff1}")
            else:
                raise RuntimeError(f"Solution of SAS function diverged at iteration {vs.itt}")
        mask1 = (tt * flux[:, :, npx.newaxis] > sa[:, :, 1, :])
        if npx.any(mask1[2:-2, 2:-2, :]):
            if rs.loglevel == 'debug':
                logger.debug(f"2Solution of SAS function diverged at iteration {vs.itt}")
                logger.debug(f"Difference: {diff2}")
            else:
                raise RuntimeError(f"Solution of SAS function diverged at iteration {vs.itt}")
        if rs.loglevel == 'debug':
            rows = npx.where(mask[2:-2, 2:-2] == False)[0].tolist()
            if rows:
                logger.debug(f"Difference: {diff1}")
                logger.debug(f"3Solution of SAS function diverged at {rows}")

    return tt


@roger_kernel
def calc_conc_iso_flux(state, mtt, tt, flux):
    """Calculates isotope signal of hydrologic flux.
    """
    conc = allocate(state.dimensions, ("x", "y"))
    conc = update(
        conc,
        at[2:-2, 2:-2], npx.nansum(tt[2:-2, 2:-2, :] * flux[2:-2, 2:-2, npx.newaxis] * mtt[2:-2, 2:-2, :], axis=-1) / npx.sum(tt[2:-2, 2:-2, :] * flux[2:-2, 2:-2, npx.newaxis], axis=-1),
    )
    conc = update(
        conc,
        at[2:-2, 2:-2], npx.where(conc[2:-2, 2:-2] != 0, conc[2:-2, 2:-2], npx.nan),
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
        at[2:-2, 2:-2, :], flux[2:-2, 2:-2, npx.newaxis] * tt[2:-2, 2:-2, :],
    )
    sa1 = update(
        sa1,
        at[2:-2, 2:-2, :], sa[2:-2, 2:-2, vs.tau, :] + flux_in[2:-2, 2:-2, :],
    )
    # weighted average isotope ratios of influx and storage
    msa1 = update(
        msa1,
        at[2:-2, 2:-2, :], (msa[2:-2, 2:-2, vs.tau, :] * (sa[2:-2, 2:-2, vs.tau, :]/sa1[2:-2, 2:-2, :])) + (mtt[2:-2, 2:-2, :] * (flux_in[2:-2, 2:-2, :]/sa1[2:-2, 2:-2, :])),
    )
    msa2 = update(
        msa2,
        at[2:-2, 2:-2, :], npx.where(npx.isnan(msa1[2:-2, 2:-2, :]), msa[2:-2, 2:-2, vs.tau, :], msa1[2:-2, 2:-2, :]),
    )
    msa3 = update(
        msa3,
        at[2:-2, 2:-2, :], npx.where(npx.isnan(msa2[2:-2, 2:-2, :]), mtt[2:-2, 2:-2, :], msa2[2:-2, 2:-2, :]),
    )
    msa = update(
        msa,
        at[2:-2, 2:-2, vs.tau, :], msa3[2:-2, 2:-2, :],
    )
    msa = update(
        msa,
        at[2:-2, 2:-2, vs.tau, :], npx.where((msa[2:-2, 2:-2, vs.tau, :] != 0), msa[2:-2, 2:-2, vs.tau, :], npx.nan),
    )

    return msa


@roger_kernel
def calc_conc_iso_storage(state, sa, msa):
    """Calculates isotope signal of storage.
    """
    vs = state.variables

    conc = allocate(state.dimensions, ("x", "y"))
    conc = update(
        conc,
        at[2:-2, 2:-2], npx.nansum(sa[2:-2, 2:-2, vs.tau, :] * msa[2:-2, 2:-2, vs.tau, :], axis=-1) / npx.sum(sa[2:-2, 2:-2, vs.tau, :], axis=-1),
    )
    conc = update(
        conc,
        at[2:-2, 2:-2], npx.where(conc[2:-2, 2:-2] != 0, conc[2:-2, 2:-2], npx.nan),
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
            at[2:-2, 2:-2, :], npx.where(tt[2:-2, 2:-2, :] > 0, msa[2:-2, 2:-2, vs.tau, :], npx.nan),
        )

    else:
        # solute travel time distribution at current time step
        mask = (tt > 0) & (sa[:, :, vs.tau, :] > 0) & (msa[:, :, vs.tau, :] > 0)
        mtt = update(
            mtt,
            at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2, :], (msa[2:-2, 2:-2, vs.tau, :] / sa[2:-2, 2:-2, vs.tau, :]) * alpha[2:-2, 2:-2, npx.newaxis] * tt[2:-2, 2:-2, :] * flux[2:-2, 2:-2, npx.newaxis], 0.),
        )

        mask1 = (mtt > msa[:, :, vs.tau, :])
        mtt = update(
            mtt,
            at[2:-2, 2:-2, :], npx.where(mask1[2:-2, 2:-2, :], msa[2:-2, 2:-2, vs.tau, :], mtt[2:-2, 2:-2, :]),
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
    mask0 = (sa[:, :, vs.tau, :] > -0.001) & (sa[:, :, vs.tau, :] < 0)
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
def calc_ageing_iso(state, sa, msa):
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
    # isotope ageing
    msam1 = allocate(state.dimensions, ("x", "y", "ages"))
    msam1 = update(
        msam1,
        at[2:-2, 2:-2, :], msa[2:-2, 2:-2, vs.tau, :],
    )
    msa = update(
        msa,
        at[2:-2, 2:-2, vs.tau, 1:], msam1[2:-2, 2:-2, :-1],
    )
    # add youngest isotope input to isotope StorAge
    msa = update(
        msa,
        at[2:-2, 2:-2, vs.tau, 0], npx.nan,
    )
    # merge oldest isotopes
    sum_old = allocate(state.dimensions, ("x", "y"))
    iso_old1 = allocate(state.dimensions, ("x", "y"))
    iso_old2 = allocate(state.dimensions, ("x", "y"))
    sum_old = update(
        sum_old,
        at[2:-2, 2:-2], sam1[2:-2, 2:-2, -1] + sa[2:-2, 2:-2, vs.tau, -1],
    )
    iso_old1 = update(
        iso_old1,
        at[2:-2, 2:-2], (msam1[2:-2, 2:-2, -1] * (sam1[2:-2, 2:-2, -1]/sum_old[2:-2, 2:-2])) + (msa[2:-2, 2:-2, vs.tau, -1] * (sa[2:-2, 2:-2, vs.tau, -1]/sum_old[2:-2, 2:-2])),
    )
    iso_old2 = update(
        iso_old2,
        at[2:-2, 2:-2], npx.where(npx.isnan(iso_old1[2:-2, 2:-2]), npx.where(npx.isnan(msam1[2:-2, 2:-2, -1]), msa[2:-2, 2:-2, vs.tau, -1] * (sa[2:-2, 2:-2, vs.tau, -1]/sum_old[2:-2, 2:-2]), msam1[2:-2, 2:-2, -1] * (sam1[2:-2, 2:-2, -1]/sum_old[2:-2, 2:-2])), iso_old1[2:-2, 2:-2]),
    )
    msa = update(
        msa,
        at[2:-2, 2:-2, vs.tau, -1], iso_old2[2:-2, 2:-2],
    )
    # merge oldest water
    sa = update_add(
        sa,
        at[2:-2, 2:-2, vs.tau, -1], sam1[2:-2, 2:-2, -1],
    )

    return sa, msa


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
def calculate_ageing_iso_kernel(state):
    """Calculates ageing of solute StorAge
    """
    vs = state.variables

    sum_iso_rz = npx.nansum(vs.sa_rz[2:-2, 2:-2, vs.tau, :] * vs.msa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)

    sa_rz, msa_rz = calc_ageing_iso(state, vs.sa_rz, vs.msa_rz)

    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, vs.tau, :], sa_rz[2:-2, 2:-2, vs.tau, :],
    )

    vs.msa_rz = update(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, :], msa_rz[2:-2, 2:-2, vs.tau, :],
    )

    sum_iso_ss = npx.nansum(vs.sa_ss[2:-2, 2:-2, vs.tau, :] * vs.msa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)

    sa_ss, msa_ss = calc_ageing_iso(state, vs.sa_ss, vs.msa_ss)

    vs.sa_ss = update(
        vs.sa_ss,
        at[2:-2, 2:-2, vs.tau, :], sa_ss[2:-2, 2:-2, vs.tau, :],
    )

    vs.msa_ss = update(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.tau, :], msa_ss[2:-2, 2:-2, vs.tau, :],
    )

    # check for mass conservation after ageing
    sum_iso_rz1 = npx.nansum(vs.sa_rz[2:-2, 2:-2, vs.tau, :] * vs.msa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)
    sum_iso_ss1 = npx.nansum(vs.sa_ss[2:-2, 2:-2, vs.tau, :] * vs.msa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)
    check11 = npx.isclose(sum_iso_rz, sum_iso_rz1, atol=1e-02)
    check12 = npx.isclose(sum_iso_ss, sum_iso_ss1, atol=1e-02)

    if rs.loglevel == 'debug' and rs.backend == 'numpy':
        if not check11.all():
            logger.debug(f"Solution for ageing in root zone diverged at iteration {vs.itt}")

        if not check12.all():
            logger.debug(f"Solution for ageing in subsoil diverged at iteration {vs.itt}")

    return KernelOutput(sa_rz=vs.sa_rz, sa_ss=vs.sa_ss, msa_rz=vs.msa_rz, msa_ss=vs.msa_ss)


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
