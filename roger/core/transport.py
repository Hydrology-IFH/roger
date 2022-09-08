from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at, scan
from roger.core import sas, infiltration, evapotranspiration, subsurface_runoff, capillary_rise, crop
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
def calc_conc_iso_flux(state, mtt, tt, flux):
    """Calculates isotope signal of hydrologic flux.
    """
    conc = allocate(state.dimensions, ("x", "y"))
    conc = update(
        conc,
        at[2:-2, 2:-2], npx.nansum(mtt[2:-2, 2:-2, :], axis=-1) / npx.sum(tt[2:-2, 2:-2, :] * flux[2:-2, 2:-2, npx.newaxis], axis=-1),
    )
    conc = update(
        conc,
        at[2:-2, 2:-2], npx.where(conc[2:-2, 2:-2] != 0, conc[2:-2, 2:-2], npx.nan),
    )

    return conc


@roger_kernel
def calc_conc_iso_storage(state, sa, msa):
    """Calculates isotope signal of storage.
    """
    vs = state.variables

    conc = allocate(state.dimensions, ("x", "y"))
    conc = update(
        conc,
        at[2:-2, 2:-2], npx.nansum(msa[2:-2, 2:-2, vs.tau, :], axis=-1) / npx.sum(sa[2:-2, 2:-2, vs.tau, :], axis=-1),
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

    mtt = allocate(state.dimensions, ("x", "y", "ages"))

    # solute travel time distribution at current time step
    mtt = update(
        mtt,
        at[2:-2, 2:-2, :], (msa[2:-2, 2:-2, vs.tau, :] / sa[2:-2, 2:-2, vs.tau, :]) * alpha[2:-2, 2:-2, npx.newaxis] * tt[2:-2, 2:-2, :] * flux[2:-2, 2:-2, npx.newaxis],
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
        vs.update(calculate_ageing_sa_kernel(state))
        vs.update(calculate_ageing_msa_kernel(state))
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
def calc_omega_q(state, SA, sas_params, flux):
    """
    Calculates Omega
    """
    omega_q = allocate(state.dimensions, ("x", "y", "nages"))
    omega_q = update_add(
        omega_q,
        at[2:-2, 2:-2, :], sas.uniform(state, SA, sas_params)[2:-2, 2:-2, :],
    )
    omega_q = update_add(
        omega_q,
        at[2:-2, 2:-2, :], sas.dirac(state, sas_params)[2:-2, 2:-2, :],
    )
    omega_q = update(
        omega_q,
        at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 21) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(flux[2:-2, 2:-2, npx.newaxis] <= SA[2:-2, 2:-2, 1, :], SA[2:-2, 2:-2, 1, :]/flux[2:-2, 2:-2, npx.newaxis], 0), omega_q[2:-2, 2:-2, :]),
    )
    omega_q = update(
        omega_q,
        at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 21) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(npx.max(omega_q[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis] > 1, omega_q[2:-2, 2:-2, :] * (1/npx.max(omega_q[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis]), omega_q[2:-2, 2:-2, :]), omega_q[2:-2, 2:-2, :]),
    )
    omega_q = update(
        omega_q,
        at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 22) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(flux[2:-2, 2:-2, npx.newaxis] <= npx.cumsum(SA[2:-2, 2:-2, 1, ::-1], axis=-1)[:, :, ::-1], npx.cumsum(SA[2:-2, 2:-2, 1, ::-1], axis=-1)[:, :, ::-1]/flux[2:-2, 2:-2, npx.newaxis], 0), omega_q[2:-2, 2:-2, :]),
    )
    omega_q = update(
        omega_q,
        at[2:-2, 2:-2, :], npx.where((sas_params[2:-2, 2:-2, 0, npx.newaxis] == 22) & (flux[2:-2, 2:-2, npx.newaxis] > 0), npx.where(npx.max(omega_q[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis] > 1, omega_q[2:-2, 2:-2, :] * (1/npx.max(omega_q[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis]), omega_q[2:-2, 2:-2, :]), omega_q[2:-2, 2:-2, :]),
    )
    Omega, sas_params = sas.kumaraswami(state, SA, sas_params)
    omega_q = update_add(
        omega_q,
        at[2:-2, 2:-2, :], Omega[2:-2, 2:-2, :],
    )
    omega_q = update_add(
        omega_q,
        at[2:-2, 2:-2, :], sas.gamma(state, SA, sas_params)[2:-2, 2:-2, :],
    )
    omega_q = update_add(
        omega_q,
        at[2:-2, 2:-2, :], sas.exponential(state, SA, sas_params)[2:-2, 2:-2, :],
    )
    omega_q = update_add(
        omega_q,
        at[2:-2, 2:-2, :], sas.power(state, SA, sas_params)[2:-2, 2:-2, :],
    )

    return omega_q


@roger_kernel
def transport_model_deterministic(state):
    """Calculates water transport model with deterministic method (i.e. not numerically)
    """
    settings = state.settings

    if settings.enable_crop_phenology:
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
    if settings.enable_groundwater_boundary:
        with state.timers["capillary rise into subsoil"]:
            capillary_rise.calculate_capillary_rise_ss_transport(state)
    if settings.enable_groundwater:
        pass


@roger_kernel
def transport_model_rk4(state):
    """Calculates water transport model with Runge-Kutta method
    """
    vs = state.variables
    settings = state.settings
    h = 1 / settings.sas_solver_substeps

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :], calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    SAn_rz = allocate(state.dimensions, ("x", "y", "timesteps", "nages"))
    san_rz = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
    SAn_rz = update(
        SAn_rz,
        at[2:-2, 2:-2, :, :], vs.SA_rz[2:-2, 2:-2, :, :],
    )
    san_rz = update(
        san_rz,
        at[2:-2, 2:-2, :, :], vs.sa_rz[2:-2, 2:-2, :, :],
    )

    SAn_ss = allocate(state.dimensions, ("x", "y", "timesteps", "nages"))
    san_ss = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
    SAn_ss = update(
        SAn_ss,
        at[2:-2, 2:-2, :, :], vs.SA_ss[2:-2, 2:-2, :, :],
    )
    san_ss = update(
        san_ss,
        at[2:-2, 2:-2, :, :], vs.sa_ss[2:-2, 2:-2, :, :],
    )

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

    ttn_evap_soil = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    ttn_transp = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    ttn_cpr_rz = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    ttn_q_rz = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    ttn_q_ss = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))

    carry_rk = (TTrkn_evap_soil, ttrkn_evap_soil, TTrkn_transp, ttrkn_transp, TTrkn_cpr_rz, ttrkn_cpr_rz, TTrkn_q_rz, ttrkn_q_rz, TTrkn_q_ss, ttrkn_q_ss)
    carry_substeps = (ttn_evap_soil, ttn_transp, ttn_cpr_rz, ttn_q_rz, ttn_q_ss, SAn_rz, san_rz, SAn_ss, san_ss)

    # loop to solve SAS functions numerically
    def body(carry, i):
        state, carry_rk, carry_substeps = carry
        TTrkn_evap_soil, ttrkn_evap_soil, TTrkn_transp, ttrkn_transp, TTrkn_cpr_rz, ttrkn_cpr_rz, TTrkn_q_rz, ttrkn_q_rz, TTrkn_q_ss, ttrkn_q_ss = carry_rk
        ttn_evap_soil, ttn_transp, ttn_cpr_rz, ttn_q_rz, ttn_q_ss, SAn_rz, san_rz, SAn_ss, san_ss = carry_substeps
        vs = state.variables

        SArkn_rz = allocate(state.dimensions, ("x", "y", "timesteps", "nages"))
        sarkn_rz = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
        SArkn_ss = allocate(state.dimensions, ("x", "y", "timesteps", "nages"))
        sarkn_ss = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
        SArkn_rz = update(
            SArkn_rz,
            at[2:-2, 2:-2, :, :], SAn_rz[2:-2, 2:-2, :, :],
        )
        sarkn_rz = update(
            sarkn_rz,
            at[2:-2, 2:-2, :, :], san_rz[2:-2, 2:-2, :, :],
        )
        SArkn_ss = update(
            SArkn_ss,
            at[2:-2, 2:-2, :, :], SAn_ss[2:-2, 2:-2, :, :],
        )
        sarkn_ss = update(
            sarkn_ss,
            at[2:-2, 2:-2, :, :], san_ss[2:-2, 2:-2, :, :],
        )

        # step 1
        TTrkn_evap_soil = update(
            TTrkn_evap_soil,
            at[2:-2, 2:-2, :, 0], calc_omega_q(state, SArkn_rz, vs.sas_params_evap_soil, vs.evap_soil * h)[2:-2, 2:-2, :],
        )
        ttrkn_evap_soil = update(
            ttrkn_evap_soil,
            at[2:-2, 2:-2, :, 0], npx.diff(TTrkn_evap_soil[2:-2, 2:-2, :, 0], axis=-1),
        )
        TTrkn_transp = update(
            TTrkn_transp,
            at[2:-2, 2:-2, :, 0], calc_omega_q(state, SArkn_rz, vs.sas_params_transp, vs.transp * h)[2:-2, 2:-2, :],
        )
        ttrkn_transp = update(
            ttrkn_transp,
            at[2:-2, 2:-2, :, 0], npx.diff(TTrkn_transp[2:-2, 2:-2, :, 0], axis=-1),
        )
        TTrkn_q_rz = update(
            TTrkn_q_rz,
            at[2:-2, 2:-2, :, 0], calc_omega_q(state, SArkn_rz, vs.sas_params_q_rz, vs.q_rz * h)[2:-2, 2:-2, :],
        )
        ttrkn_q_rz = update(
            ttrkn_q_rz,
            at[2:-2, 2:-2, :, 0], npx.diff(TTrkn_q_rz[2:-2, 2:-2, :, 0], axis=-1),
        )
        TTrkn_cpr_rz = update(
            TTrkn_cpr_rz,
            at[2:-2, 2:-2, :, 0], calc_omega_q(state, SArkn_ss, vs.sas_params_cpr_rz, vs.cpr_rz * h)[2:-2, 2:-2, :],
        )
        ttrkn_cpr_rz = update(
            ttrkn_cpr_rz,
            at[2:-2, 2:-2, :, 0], npx.diff(TTrkn_cpr_rz[2:-2, 2:-2, :, 0], axis=-1),
        )
        TTrkn_q_ss = update(
            TTrkn_q_ss,
            at[2:-2, 2:-2, :, 0], calc_omega_q(state, SArkn_ss, vs.sas_params_q_ss, vs.q_ss * h)[2:-2, 2:-2, :],
        )
        ttrkn_q_ss = update(
            ttrkn_q_ss,
            at[2:-2, 2:-2, :, 0], npx.diff(TTrkn_q_ss[2:-2, 2:-2, :, 0], axis=-1),
        )

        sarkn_rz = update_add(
            sarkn_rz,
            at[2:-2, 2:-2, 1, :], (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] + vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 0] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * ttrkn_evap_soil[2:-2, 2:-2, :, 0] - vs.transp[2:-2, 2:-2, npx.newaxis] * ttrkn_transp[2:-2, 2:-2, :, 0] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 0]) * h/2,
        )
        sarkn_ss = update_add(
            sarkn_ss,
            at[2:-2, 2:-2, 1, :], (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] + vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 0] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 0] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * ttrkn_q_ss[2:-2, 2:-2, :, 0]) * h/2,
        )
        SArkn_rz = update(
            SArkn_rz,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(sarkn_rz[2:-2, 2:-2, 1, :], axis=-1),
        )
        SArkn_ss = update(
            SArkn_ss,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(sarkn_ss[2:-2, 2:-2, 1, :], axis=-1),
        )

        # step 2
        TTrkn_evap_soil = update(
            TTrkn_evap_soil,
            at[2:-2, 2:-2, :, 1], calc_omega_q(state, SArkn_rz, vs.sas_params_evap_soil, vs.evap_soil * h)[2:-2, 2:-2, :],
        )
        ttrkn_evap_soil = update(
            ttrkn_evap_soil,
            at[2:-2, 2:-2, :, 1], npx.diff(TTrkn_evap_soil[2:-2, 2:-2, :, 1], axis=-1),
        )
        TTrkn_transp = update(
            TTrkn_transp,
            at[2:-2, 2:-2, :, 1], calc_omega_q(state, SArkn_rz, vs.sas_params_transp, vs.transp * h)[2:-2, 2:-2, :],
        )
        ttrkn_transp = update(
            ttrkn_transp,
            at[2:-2, 2:-2, :, 1], npx.diff(TTrkn_transp[2:-2, 2:-2, :, 1], axis=-1),
        )
        TTrkn_q_rz = update(
            TTrkn_q_rz,
            at[2:-2, 2:-2, :, 1], calc_omega_q(state, SArkn_rz, vs.sas_params_q_rz, vs.q_rz * h)[2:-2, 2:-2, :],
        )
        ttrkn_q_rz = update(
            ttrkn_q_rz,
            at[2:-2, 2:-2, :, 1], npx.diff(TTrkn_q_rz[2:-2, 2:-2, :, 1], axis=-1),
        )
        TTrkn_cpr_rz = update(
            TTrkn_cpr_rz,
            at[2:-2, 2:-2, :, 1], calc_omega_q(state, SArkn_ss, vs.sas_params_cpr_rz, vs.cpr_rz * h)[2:-2, 2:-2, :],
        )
        ttrkn_cpr_rz = update(
            ttrkn_cpr_rz,
            at[2:-2, 2:-2, :, 1], npx.diff(TTrkn_cpr_rz[2:-2, 2:-2, :, 1], axis=-1),
        )
        TTrkn_q_ss = update(
            TTrkn_q_ss,
            at[2:-2, 2:-2, :, 1], calc_omega_q(state, SArkn_ss, vs.sas_params_q_ss, vs.q_ss * h)[2:-2, 2:-2, :],
        )
        ttrkn_q_ss = update(
            ttrkn_q_ss,
            at[2:-2, 2:-2, :, 1], npx.diff(TTrkn_q_ss[2:-2, 2:-2, :, 1], axis=-1),
        )

        sarkn_rz = update_add(
            sarkn_rz,
            at[2:-2, 2:-2, 1, :], (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] + vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 1] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * ttrkn_evap_soil[2:-2, 2:-2, :, 1] - vs.transp[2:-2, 2:-2, npx.newaxis] * ttrkn_transp[2:-2, 2:-2, :, 1] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 1]) * h/2,
        )
        sarkn_ss = update_add(
            sarkn_ss,
            at[2:-2, 2:-2, 1, :], (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] + vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 1] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 1] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * ttrkn_q_ss[2:-2, 2:-2, :, 1]) * h/2,
        )
        SArkn_rz = update(
            SArkn_rz,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(sarkn_rz[2:-2, 2:-2, 1, :], axis=-1),
        )
        SArkn_ss = update(
            SArkn_ss,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(sarkn_ss[2:-2, 2:-2, 1, :], axis=-1),
        )

        # step 3
        TTrkn_evap_soil = update(
            TTrkn_evap_soil,
            at[2:-2, 2:-2, :, 2], calc_omega_q(state, SArkn_rz, vs.sas_params_evap_soil, vs.evap_soil * h)[2:-2, 2:-2, :],
        )
        ttrkn_evap_soil = update(
            ttrkn_evap_soil,
            at[2:-2, 2:-2, :, 2], npx.diff(TTrkn_evap_soil[2:-2, 2:-2, :, 2], axis=-1),
        )
        TTrkn_transp = update(
            TTrkn_transp,
            at[2:-2, 2:-2, :, 2], calc_omega_q(state, SArkn_rz, vs.sas_params_transp, vs.transp * h)[2:-2, 2:-2, :],
        )
        ttrkn_transp = update(
            ttrkn_transp,
            at[2:-2, 2:-2, :, 2], npx.diff(TTrkn_transp[2:-2, 2:-2, :, 2], axis=-1),
        )
        TTrkn_q_rz = update(
            TTrkn_q_rz,
            at[2:-2, 2:-2, :, 2], calc_omega_q(state, SArkn_rz, vs.sas_params_q_rz, vs.q_rz * h)[2:-2, 2:-2, :],
        )
        ttrkn_q_rz = update(
            ttrkn_q_rz,
            at[2:-2, 2:-2, :, 2], npx.diff(TTrkn_q_rz[2:-2, 2:-2, :, 2], axis=-1),
        )
        TTrkn_cpr_rz = update(
            TTrkn_cpr_rz,
            at[2:-2, 2:-2, :, 2], calc_omega_q(state, SArkn_ss, vs.sas_params_cpr_rz, vs.cpr_rz * h)[2:-2, 2:-2, :],
        )
        ttrkn_cpr_rz = update(
            ttrkn_cpr_rz,
            at[2:-2, 2:-2, :, 2], npx.diff(TTrkn_cpr_rz[2:-2, 2:-2, :, 2], axis=-1),
        )
        TTrkn_q_ss = update(
            TTrkn_q_ss,
            at[2:-2, 2:-2, :, 2], calc_omega_q(state, SArkn_ss, vs.sas_params_q_ss, vs.q_ss * h)[2:-2, 2:-2, :],
        )
        ttrkn_q_ss = update(
            ttrkn_q_ss,
            at[2:-2, 2:-2, :, 2], npx.diff(TTrkn_q_ss[2:-2, 2:-2, :, 2], axis=-1),
        )

        sarkn_rz = update_add(
            sarkn_rz,
            at[2:-2, 2:-2, 1, :], (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] + vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 2] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * ttrkn_evap_soil[2:-2, 2:-2, :, 2] - vs.transp[2:-2, 2:-2, npx.newaxis] * ttrkn_transp[2:-2, 2:-2, :, 2] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 2]) * h,
        )
        sarkn_ss = update_add(
            sarkn_ss,
            at[2:-2, 2:-2, 1, :], (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] + vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 2] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 2] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * ttrkn_q_ss[2:-2, 2:-2, :, 2]) * h,
        )
        SArkn_rz = update(
            SArkn_rz,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(sarkn_rz[2:-2, 2:-2, 1, :], axis=-1),
        )
        SArkn_ss = update(
            SArkn_ss,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(sarkn_ss[2:-2, 2:-2, 1, :], axis=-1),
        )

        # step 4
        TTrkn_evap_soil = update(
            TTrkn_evap_soil,
            at[2:-2, 2:-2, :, 3], calc_omega_q(state, SArkn_rz, vs.sas_params_evap_soil, vs.evap_soil * h)[2:-2, 2:-2, :],
        )
        ttrkn_evap_soil = update(
            ttrkn_evap_soil,
            at[2:-2, 2:-2, :, 3], npx.diff(TTrkn_evap_soil[2:-2, 2:-2, :, 3], axis=-1),
        )
        TTrkn_transp = update(
            TTrkn_transp,
            at[2:-2, 2:-2, :, 3], calc_omega_q(state, SArkn_rz, vs.sas_params_transp, vs.transp * h)[2:-2, 2:-2, :],
        )
        ttrkn_transp = update(
            ttrkn_transp,
            at[2:-2, 2:-2, :, 3], npx.diff(TTrkn_transp[2:-2, 2:-2, :, 3], axis=-1),
        )
        TTrkn_q_rz = update(
            TTrkn_q_rz,
            at[2:-2, 2:-2, :, 3], calc_omega_q(state, SArkn_rz, vs.sas_params_q_rz, vs.q_rz * h)[2:-2, 2:-2, :],
        )
        ttrkn_q_rz = update(
            ttrkn_q_rz,
            at[2:-2, 2:-2, :, 3], npx.diff(TTrkn_q_rz[2:-2, 2:-2, :, 3], axis=-1),
        )
        TTrkn_cpr_rz = update(
            TTrkn_cpr_rz,
            at[2:-2, 2:-2, :, 3], calc_omega_q(state, SArkn_ss, vs.sas_params_cpr_rz, vs.cpr_rz * h)[2:-2, 2:-2, :],
        )
        ttrkn_cpr_rz = update(
            ttrkn_cpr_rz,
            at[2:-2, 2:-2, :, 3], npx.diff(TTrkn_cpr_rz[2:-2, 2:-2, :, 3], axis=-1),
        )
        TTrkn_q_ss = update(
            TTrkn_q_ss,
            at[2:-2, 2:-2, :, 3], calc_omega_q(state, SArkn_ss, vs.sas_params_q_ss, vs.q_ss * h)[2:-2, 2:-2, :],
        )
        ttrkn_q_ss = update(
            ttrkn_q_ss,
            at[2:-2, 2:-2, :, 3], npx.diff(TTrkn_q_ss[2:-2, 2:-2, :, 3], axis=-1),
        )

        ttn_evap_soil = update(
            ttn_evap_soil,
            at[2:-2, 2:-2, :, i], (ttrkn_evap_soil[2:-2, 2:-2, :, 0] + 2*ttrkn_evap_soil[2:-2, 2:-2, :, 1] + 2*ttrkn_evap_soil[2:-2, 2:-2, :, 2] + ttrkn_evap_soil[2:-2, 2:-2, :, 3]) / 6.,
        )
        ttn_transp = update(
            ttn_transp,
            at[2:-2, 2:-2, :, i], (ttrkn_transp[2:-2, 2:-2, :, 0] + 2*ttrkn_transp[2:-2, 2:-2, :, 1] + 2*ttrkn_transp[2:-2, 2:-2, :, 2] + ttrkn_transp[2:-2, 2:-2, :, 3]) / 6.,
        )
        ttn_q_rz = update(
            ttn_q_rz,
            at[2:-2, 2:-2, :, i], (ttrkn_q_rz[2:-2, 2:-2, :, 0] + 2*ttrkn_q_rz[2:-2, 2:-2, :, 1] + 2*ttrkn_q_rz[2:-2, 2:-2, :, 2] + ttrkn_q_rz[2:-2, 2:-2, :, 3]) / 6.,
        )
        ttn_cpr_rz = update(
            ttn_cpr_rz,
            at[2:-2, 2:-2, :, i], (ttrkn_cpr_rz[2:-2, 2:-2, :, 0] + 2*ttrkn_cpr_rz[2:-2, 2:-2, :, 1] + 2*ttrkn_cpr_rz[2:-2, 2:-2, :, 2] + ttrkn_cpr_rz[2:-2, 2:-2, :, 3]) / 6.,
        )
        ttn_q_ss = update(
            ttn_q_ss,
            at[2:-2, 2:-2, :, i], (ttrkn_q_ss[2:-2, 2:-2, :, 0] + 2*ttrkn_q_ss[2:-2, 2:-2, :, 1] + 2*ttrkn_q_ss[2:-2, 2:-2, :, 2] + ttrkn_q_ss[2:-2, 2:-2, :, 3]) / 6.,
        )

        san_rz = update_add(
            san_rz,
            at[2:-2, 2:-2, 1, :], (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] + vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttn_cpr_rz[2:-2, 2:-2, :, i] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * ttn_evap_soil[2:-2, 2:-2, :, i] - vs.transp[2:-2, 2:-2, npx.newaxis] * ttn_transp[2:-2, 2:-2, :, i] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttn_q_rz[2:-2, 2:-2, :, i]) * h,
        )
        san_ss = update_add(
            san_ss,
            at[2:-2, 2:-2, 1, :], (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] + vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttn_q_rz[2:-2, 2:-2, :, i] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttn_cpr_rz[2:-2, 2:-2, :, i] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * ttn_q_ss[2:-2, 2:-2, :, i]) * h,
        )
        SAn_rz = update(
            SAn_rz,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(san_rz[2:-2, 2:-2, 1, :], axis=-1),
        )
        SAn_ss = update(
            SAn_ss,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(san_ss[2:-2, 2:-2, 1, :], axis=-1),
        )

        carry_rk = (TTrkn_evap_soil, ttrkn_evap_soil, TTrkn_transp, ttrkn_transp, TTrkn_cpr_rz, ttrkn_cpr_rz, TTrkn_q_rz, ttrkn_q_rz, TTrkn_q_ss, ttrkn_q_ss)
        carry_substeps = (ttn_evap_soil, ttn_transp, ttn_cpr_rz, ttn_q_rz, ttn_q_ss, SAn_rz, san_rz, SAn_ss, san_ss)
        carry = (state, carry_rk, carry_substeps)

        return carry, None

    n_substeps = npx.arange(0, settings.sas_solver_substeps, dtype=int)

    carry = (state, carry_rk, carry_substeps)
    res, _ = scan(body, carry, n_substeps)
    res_substeps = res[1]
    ttn_evap_soil = res_substeps[1]
    ttn_transp = res_substeps[2]
    ttn_cpr_rz = res_substeps[3]
    ttn_q_rz = res_substeps[5]
    ttn_q_ss = res_substeps[6]

    # backward travel time distributions
    vs.tt_evap_soil = update(
        vs.tt_evap_soil,
        at[2:-2, 2:-2, :], npx.sum(ttn_transp / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.tt_transp = update(
        vs.tt_transp,
        at[2:-2, 2:-2, :], npx.sum(ttn_transp / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.tt_q_rz = update(
        vs.tt_q_rz,
        at[2:-2, 2:-2, :], npx.sum(ttn_transp / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.tt_cpr_rz = update(
        vs.tt_cpr_rz,
        at[2:-2, 2:-2, :], npx.sum(ttn_transp / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.tt_q_ss = update(
        vs.tt_q_ss,
        at[2:-2, 2:-2, :], npx.sum(ttn_transp / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    # cumulative backward travel time distributions
    vs.TT_evap_soil = update(
        vs.TT_evap_soil,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_evap_soil, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.TT_transp = update(
        vs.TT_transp,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_transp, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.TT_q_rz = update(
        vs.TT_q_rz,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_rz, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.TT_cpr_rz = update(
        vs.TT_cpr_rz,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_cpr_rz, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.TT_q_ss = update(
        vs.TT_q_ss,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_ss, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    # update StorAge
    vs.sa_rz = update_add(
        vs.sa_rz,
        at[2:-2, 2:-2, 1, :], vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] + vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :] - vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :],
    )

    vs.sa_ss = update_add(
        vs.sa_ss,
        at[2:-2, 2:-2, 1, :], vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] + vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :],
    )

    return KernelOutput(tt_evap_soil=vs.tt_evap_soil, tt_transp=vs.tt_transp, tt_q_rz=vs.tt_q_rz, tt_cpr_rz=vs.tt_cpr_rz, tt_q_ss=vs.tt_q_ss,
                        TT_evap_soil=vs.TT_evap_soil, TT_transp=vs.TT_transp, TT_q_rz=vs.TT_q_rz, TT_cpr_rz=vs.TT_cpr_rz, TT_q_ss=vs.TT_q_ss,
                        sa_rz=vs.sa_rz, sa_ss=vs.sa_ss)


@roger_kernel
def solute_transport_model_rk4(state):
    """Calculates solute transport model with Runge-Kutta method
    """
    vs = state.variables
    settings = state.settings
    h = 1 / settings.sas_solver_substeps

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :], calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    SAn_rz = allocate(state.dimensions, ("x", "y", "timesteps", "nages"))
    san_rz = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
    msan_rz = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
    SAn_rz = update(
        SAn_rz,
        at[2:-2, 2:-2, :, :], vs.SA_rz[2:-2, 2:-2, :, :],
    )
    san_rz = update(
        san_rz,
        at[2:-2, 2:-2, :, :], vs.sa_rz[2:-2, 2:-2, :, :],
    )
    msan_rz = update(
        msan_rz,
        at[2:-2, 2:-2, :, :], vs.msa_rz[2:-2, 2:-2, :, :],
    )

    SAn_ss = allocate(state.dimensions, ("x", "y", "timesteps", "nages"))
    san_ss = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
    msan_ss = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
    SAn_ss = update(
        SAn_ss,
        at[2:-2, 2:-2, :, :], vs.SA_ss[2:-2, 2:-2, :, :],
    )
    san_ss = update(
        san_ss,
        at[2:-2, 2:-2, :, :], vs.sa_ss[2:-2, 2:-2, :, :],
    )
    msan_ss = update(
        msan_ss,
        at[2:-2, 2:-2, :, :], vs.msa_ss[2:-2, 2:-2, :, :],
    )

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
            at[2:-2, 2:-2, 0], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, vs.inf_mat_rz[2:-2, 2:-2] * vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_mat_rz = update(
            vs.mtt_inf_mat_rz,
            at[2:-2, 2:-2, :], npx.where(vs.mtt_inf_mat_rz == 0, npx.nan, vs.mtt_inf_mat_rz)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
        )
        vs.mtt_inf_pf_rz = update(
            vs.mtt_inf_pf_rz,
            at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, vs.inf_pf_rz[2:-2, 2:-2] * vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_pf_rz = update(
            vs.mtt_inf_pf_rz,
            at[2:-2, 2:-2, :], npx.where(vs.mtt_inf_pf_rz == 0, npx.nan, vs.mtt_inf_pf_rz)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
        )
        vs.mtt_inf_pf_ss = update(
            vs.mtt_inf_pf_ss,
            at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, vs.inf_pf_ss[2:-2, 2:-2] * vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_pf_ss = update(
            vs.mtt_inf_pf_ss,
            at[2:-2, 2:-2, :], npx.where(vs.mtt_inf_pf_ss == 0, npx.nan, vs.mtt_inf_pf_ss)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
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

    ttn_evap_soil = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    ttn_transp = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    ttn_cpr_rz = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    ttn_q_rz = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    ttn_q_ss = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    mttn_evap_soil = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    mttn_transp = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    mttn_cpr_rz = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    mttn_q_rz = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    mttn_q_ss = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))

    carry_tt_rk = (TTrkn_evap_soil, ttrkn_evap_soil, TTrkn_transp, ttrkn_transp, TTrkn_cpr_rz, ttrkn_cpr_rz, TTrkn_q_rz, ttrkn_q_rz, TTrkn_q_ss, ttrkn_q_ss)
    carry_mtt_rk = (mttrkn_evap_soil, mttrkn_transp, mttrkn_cpr_rz, mttrkn_q_rz, mttrkn_q_ss)
    carry_tt_substeps = (ttn_evap_soil, ttn_transp, ttn_cpr_rz, ttn_q_rz, ttn_q_ss)
    carry_mtt_substeps = (mttn_evap_soil, mttn_transp, mttn_cpr_rz, mttn_q_rz, mttn_q_ss)
    carry_sa_substeps = (SAn_rz, san_rz, SAn_ss, san_ss, msan_rz, msan_ss)

    # loop to solve SAS functions numerically
    def body(carry, i):
        state, carry_tt_rk, carry_mtt_rk, carry_tt_substeps, carry_mtt_substeps, carry_sa_substeps = carry
        TTrkn_evap_soil, ttrkn_evap_soil, TTrkn_transp, ttrkn_transp, TTrkn_cpr_rz, ttrkn_cpr_rz, TTrkn_q_rz, ttrkn_q_rz, TTrkn_q_ss, ttrkn_q_ss = carry_tt_rk
        mttrkn_evap_soil, mttrkn_transp, mttrkn_cpr_rz, mttrkn_q_rz, mttrkn_q_ss = carry_mtt_rk
        ttn_evap_soil, ttn_transp, ttn_cpr_rz, ttn_q_rz, ttn_q_ss = carry_tt_substeps
        mttn_evap_soil, mttn_transp, mttn_cpr_rz, mttn_q_rz, mttn_q_ss = carry_mtt_substeps
        SAn_rz, san_rz, SAn_ss, san_ss, msan_rz, msan_ss = carry_sa_substeps
        vs = state.variables

        SArkn_rz = allocate(state.dimensions, ("x", "y", "timesteps", "nages"))
        sarkn_rz = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
        msarkn_rz = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
        SArkn_ss = allocate(state.dimensions, ("x", "y", "timesteps", "nages"))
        sarkn_ss = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
        msarkn_ss = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
        SArkn_rz = update(
            SArkn_rz,
            at[2:-2, 2:-2, :, :], SAn_rz[2:-2, 2:-2, :, :],
        )
        sarkn_rz = update(
            sarkn_rz,
            at[2:-2, 2:-2, :, :], san_rz[2:-2, 2:-2, :, :],
        )
        msarkn_rz = update(
            msarkn_rz,
            at[2:-2, 2:-2, :, :], msan_rz[2:-2, 2:-2, :, :],
        )
        SArkn_ss = update(
            SArkn_ss,
            at[2:-2, 2:-2, :, :], SAn_ss[2:-2, 2:-2, :, :],
        )
        sarkn_ss = update(
            sarkn_ss,
            at[2:-2, 2:-2, :, :], san_ss[2:-2, 2:-2, :, :],
        )
        msarkn_ss = update(
            msarkn_ss,
            at[2:-2, 2:-2, :, :], msan_ss[2:-2, 2:-2, :, :],
        )

        # step 1
        TTrkn_evap_soil = update(
            TTrkn_evap_soil,
            at[2:-2, 2:-2, :, 0], calc_omega_q(state, SArkn_rz, vs.sas_params_evap_soil, vs.evap_soil * h)[2:-2, 2:-2, :],
        )
        ttrkn_evap_soil = update(
            ttrkn_evap_soil,
            at[2:-2, 2:-2, :, 0], npx.diff(TTrkn_evap_soil[2:-2, 2:-2, :, 0], axis=-1),
        )
        if settings.enable_oxygen18 or settings.enable_deuterium:
            mttrkn_evap_soil = update(
                mttrkn_evap_soil,
                at[2:-2, 2:-2, :, 0], calc_mtt(state, sarkn_rz, ttrkn_evap_soil[:, :, :, 0], vs.evap_soil * h, msarkn_rz, vs.alpha_q)[2:-2, 2:-2, :],
            )
        TTrkn_transp = update(
            TTrkn_transp,
            at[2:-2, 2:-2, :, 0], calc_omega_q(state, SArkn_rz, vs.sas_params_transp, vs.transp * h)[2:-2, 2:-2, :],
        )
        ttrkn_transp = update(
            ttrkn_transp,
            at[2:-2, 2:-2, :, 0], npx.diff(TTrkn_transp[2:-2, 2:-2, :, 0], axis=-1),
        )
        mttrkn_transp = update(
            mttrkn_transp,
            at[2:-2, 2:-2, :, 0], calc_mtt(state, sarkn_rz, ttrkn_transp[:, :, :, 0], vs.transp * h, msarkn_rz, vs.alpha_transp)[2:-2, 2:-2, :],
        )
        TTrkn_q_rz = update(
            TTrkn_q_rz,
            at[2:-2, 2:-2, :, 0], calc_omega_q(state, SArkn_rz, vs.sas_params_q_rz, vs.q_rz * h)[2:-2, 2:-2, :],
        )
        ttrkn_q_rz = update(
            ttrkn_q_rz,
            at[2:-2, 2:-2, :, 0], npx.diff(TTrkn_q_rz[2:-2, 2:-2, :, 0], axis=-1),
        )
        mttrkn_q_rz = update(
            mttrkn_q_rz,
            at[2:-2, 2:-2, :, 0], calc_mtt(state, sarkn_rz, ttrkn_q_rz[:, :, :, 0], vs.q_rz * h, msarkn_rz, vs.alpha_q)[2:-2, 2:-2, :],
        )
        TTrkn_cpr_rz = update(
            TTrkn_cpr_rz,
            at[2:-2, 2:-2, :, 0], calc_omega_q(state, SArkn_ss, vs.sas_params_cpr_rz, vs.cpr_rz * h)[2:-2, 2:-2, :],
        )
        ttrkn_cpr_rz = update(
            ttrkn_cpr_rz,
            at[2:-2, 2:-2, :, 0], npx.diff(TTrkn_cpr_rz[2:-2, 2:-2, :, 0], axis=-1),
        )
        mttrkn_cpr_rz = update(
            mttrkn_cpr_rz,
            at[2:-2, 2:-2, :, 0], calc_mtt(state, sarkn_ss, ttrkn_cpr_rz[:, :, :, 0], vs.cpr_rz * h, msarkn_ss, vs.alpha_q)[2:-2, 2:-2, :],
        )
        TTrkn_q_ss = update(
            TTrkn_q_ss,
            at[2:-2, 2:-2, :, 0], calc_omega_q(state, SArkn_ss, vs.sas_params_q_ss, vs.q_ss * h)[2:-2, 2:-2, :],
        )
        ttrkn_q_ss = update(
            ttrkn_q_ss,
            at[2:-2, 2:-2, :, 0], npx.diff(TTrkn_q_ss[2:-2, 2:-2, :, 0], axis=-1),
        )
        mttrkn_q_ss = update(
            mttrkn_q_ss,
            at[2:-2, 2:-2, :, 0], calc_mtt(state, sarkn_ss, ttrkn_q_ss[:, :, :, 0], vs.q_ss * h, msarkn_ss, vs.alpha_q)[2:-2, 2:-2, :],
        )

        sarkn_rz = update_add(
            sarkn_rz,
            at[2:-2, 2:-2, 1, :], (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] + vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 0] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * ttrkn_evap_soil[2:-2, 2:-2, :, 0] - vs.transp[2:-2, 2:-2, npx.newaxis] * ttrkn_transp[2:-2, 2:-2, :, 0] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 0]) * h/2,
        )
        sarkn_ss = update_add(
            sarkn_ss,
            at[2:-2, 2:-2, 1, :], (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] + vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 0] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 0] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * ttrkn_q_ss[2:-2, 2:-2, :, 0]) * h/2,
        )
        msarkn_rz = update_add(
            msarkn_rz,
            at[2:-2, 2:-2, 1, :], (npx.where(npx.isnan(vs.mtt_inf_mat_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_mat_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_inf_pf_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(mttrkn_cpr_rz[2:-2, 2:-2, :, 0]), 0, mttrkn_cpr_rz[2:-2, 2:-2, :, 0]) - npx.where(npx.isnan(mttrkn_evap_soil[2:-2, 2:-2, :, 0]), 0, mttrkn_evap_soil[2:-2, 2:-2, :, 0]) - npx.where(npx.isnan(mttrkn_transp[2:-2, 2:-2, :, 0]), 0, mttrkn_transp[2:-2, 2:-2, :, 0]) - npx.where(npx.isnan(mttrkn_q_rz[2:-2, 2:-2, :, 0]), 0, mttrkn_q_rz[2:-2, 2:-2, :, 0])) * h/2,
        )
        msarkn_ss = update_add(
            msarkn_ss,
            at[2:-2, 2:-2, 1, :], (npx.where(npx.isnan(vs.mtt_inf_pf_ss[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_ss[2:-2, 2:-2, :]) + npx.where(npx.isnan(mttrkn_q_rz[2:-2, 2:-2, :, 0]), 0, mttrkn_q_rz[2:-2, 2:-2, :, 0]) - npx.where(npx.isnan(mttrkn_cpr_rz[2:-2, 2:-2, :, 0]), 0, mttrkn_cpr_rz[2:-2, 2:-2, :, 0]) - npx.where(npx.isnan(mttrkn_q_ss[2:-2, 2:-2, :, 0]), 0, mttrkn_q_ss[2:-2, 2:-2, :, 0])) * h/2,
        )
        SArkn_rz = update(
            SArkn_rz,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(sarkn_rz[2:-2, 2:-2, 1, :], axis=-1),
        )
        SArkn_ss = update(
            SArkn_ss,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(sarkn_ss[2:-2, 2:-2, 1, :], axis=-1),
        )

        # step 2
        TTrkn_evap_soil = update(
            TTrkn_evap_soil,
            at[2:-2, 2:-2, :, 1], calc_omega_q(state, SArkn_rz, vs.sas_params_evap_soil, vs.evap_soil * h)[2:-2, 2:-2, :],
        )
        ttrkn_evap_soil = update(
            ttrkn_evap_soil,
            at[2:-2, 2:-2, :, 1], npx.diff(TTrkn_evap_soil[2:-2, 2:-2, :, 1], axis=-1),
        )
        if settings.enable_oxygen18 or settings.enable_deuterium:
            mttrkn_evap_soil = update(
                mttrkn_evap_soil,
                at[2:-2, 2:-2, :, 1], calc_mtt(state, sarkn_rz, ttrkn_evap_soil[:, :, :, 1], vs.evap_soil * h, msarkn_rz, vs.alpha_q)[2:-2, 2:-2, :],
            )
        TTrkn_transp = update(
            TTrkn_transp,
            at[2:-2, 2:-2, :, 1], calc_omega_q(state, SArkn_rz, vs.sas_params_transp, vs.transp * h)[2:-2, 2:-2, :],
        )
        ttrkn_transp = update(
            ttrkn_transp,
            at[2:-2, 2:-2, :, 1], npx.diff(TTrkn_transp[2:-2, 2:-2, :, 1], axis=-1),
        )
        mttrkn_transp = update(
            mttrkn_transp,
            at[2:-2, 2:-2, :, 1], calc_mtt(state, sarkn_rz, ttrkn_transp[:, :, :, 1], vs.transp * h, msarkn_rz, vs.alpha_transp)[2:-2, 2:-2, :],
        )
        TTrkn_q_rz = update(
            TTrkn_q_rz,
            at[2:-2, 2:-2, :, 1], calc_omega_q(state, SArkn_rz, vs.sas_params_q_rz, vs.q_rz * h)[2:-2, 2:-2, :],
        )
        ttrkn_q_rz = update(
            ttrkn_q_rz,
            at[2:-2, 2:-2, :, 1], npx.diff(TTrkn_q_rz[2:-2, 2:-2, :, 1], axis=-1),
        )
        mttrkn_q_rz = update(
            mttrkn_q_rz,
            at[2:-2, 2:-2, :, 1], calc_mtt(state, sarkn_rz, ttrkn_q_rz[:, :, :, 1], vs.q_rz * h, msarkn_rz, vs.alpha_q)[2:-2, 2:-2, :],
        )
        TTrkn_cpr_rz = update(
            TTrkn_cpr_rz,
            at[2:-2, 2:-2, :, 1], calc_omega_q(state, SArkn_ss, vs.sas_params_cpr_rz, vs.cpr_rz * h)[2:-2, 2:-2, :],
        )
        ttrkn_cpr_rz = update(
            ttrkn_cpr_rz,
            at[2:-2, 2:-2, :, 1], npx.diff(TTrkn_cpr_rz[2:-2, 2:-2, :, 1], axis=-1),
        )
        mttrkn_cpr_rz = update(
            mttrkn_cpr_rz,
            at[2:-2, 2:-2, :, 1], calc_mtt(state, sarkn_ss, ttrkn_cpr_rz[:, :, :, 1], vs.cpr_rz * h, msarkn_ss, vs.alpha_q)[2:-2, 2:-2, :],
        )
        TTrkn_q_ss = update(
            TTrkn_q_ss,
            at[2:-2, 2:-2, :, 1], calc_omega_q(state, SArkn_ss, vs.sas_params_q_ss, vs.q_ss * h)[2:-2, 2:-2, :],
        )
        ttrkn_q_ss = update(
            ttrkn_q_ss,
            at[2:-2, 2:-2, :, 1], npx.diff(TTrkn_q_ss[2:-2, 2:-2, :, 1], axis=-1),
        )
        mttrkn_q_ss = update(
            mttrkn_q_ss,
            at[2:-2, 2:-2, :, 1], calc_mtt(state, sarkn_ss, ttrkn_q_ss[:, :, :, 1], vs.q_ss * h, msarkn_ss, vs.alpha_q)[2:-2, 2:-2, :],
        )

        sarkn_rz = update_add(
            sarkn_rz,
            at[2:-2, 2:-2, 1, :], (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] + vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 0] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * ttrkn_evap_soil[2:-2, 2:-2, :, 0] - vs.transp[2:-2, 2:-2, npx.newaxis] * ttrkn_transp[2:-2, 2:-2, :, 0] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 0]) * h/2,
        )

        sarkn_ss = update_add(
            sarkn_ss,
            at[2:-2, 2:-2, 1, :], (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] + vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 0] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 0] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * ttrkn_q_ss[2:-2, 2:-2, :, 0]) * h/2,
        )
        msarkn_rz = update_add(
            msarkn_rz,
            at[2:-2, 2:-2, 1, :], (npx.where(npx.isnan(vs.mtt_inf_mat_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_mat_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_inf_pf_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(mttrkn_cpr_rz[2:-2, 2:-2, :, 1]), 0, mttrkn_cpr_rz[2:-2, 2:-2, :, 1]) - npx.where(npx.isnan(mttrkn_evap_soil[2:-2, 2:-2, :, 1]), 0, mttrkn_evap_soil[2:-2, 2:-2, :, 1]) - npx.where(npx.isnan(mttrkn_transp[2:-2, 2:-2, :, 1]), 0, mttrkn_transp[2:-2, 2:-2, :, 1]) - npx.where(npx.isnan(mttrkn_q_rz[2:-2, 2:-2, :, 1]), 0, mttrkn_q_rz[2:-2, 2:-2, :, 1])) * h/2,
        )
        msarkn_ss = update_add(
            msarkn_ss,
            at[2:-2, 2:-2, 1, :], (npx.where(npx.isnan(vs.mtt_inf_pf_ss[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_ss[2:-2, 2:-2, :]) + npx.where(npx.isnan(mttrkn_q_rz[2:-2, 2:-2, :, 1]), 0, mttrkn_q_rz[2:-2, 2:-2, :, 1]) - npx.where(npx.isnan(mttrkn_cpr_rz[2:-2, 2:-2, :, 1]), 0, mttrkn_cpr_rz[2:-2, 2:-2, :, 1]) - npx.where(npx.isnan(mttrkn_q_ss[2:-2, 2:-2, :, 1]), 0, mttrkn_q_ss[2:-2, 2:-2, :, 1])) * h/2,
        )
        SArkn_rz = update(
            SArkn_rz,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(sarkn_rz[2:-2, 2:-2, 1, :], axis=-1),
        )
        SArkn_ss = update(
            SArkn_ss,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(sarkn_ss[2:-2, 2:-2, 1, :], axis=-1),
        )

        # step 3
        TTrkn_evap_soil = update(
            TTrkn_evap_soil,
            at[2:-2, 2:-2, :, 2], calc_omega_q(state, SArkn_rz, vs.sas_params_evap_soil, vs.evap_soil * h)[2:-2, 2:-2, :],
        )
        ttrkn_evap_soil = update(
            ttrkn_evap_soil,
            at[2:-2, 2:-2, :, 2], npx.diff(TTrkn_evap_soil[2:-2, 2:-2, :, 2], axis=-1),
        )
        if settings.enable_oxygen18 or settings.enable_deuterium:
            mttrkn_evap_soil = update(
                mttrkn_evap_soil,
                at[2:-2, 2:-2, :, 2], calc_mtt(state, sarkn_rz, ttrkn_evap_soil[:, :, :, 2], vs.evap_soil * h, msarkn_rz, vs.alpha_q)[2:-2, 2:-2, :],
            )
        TTrkn_transp = update(
            TTrkn_transp,
            at[2:-2, 2:-2, :, 2], calc_omega_q(state, SArkn_rz, vs.sas_params_transp, vs.transp * h)[2:-2, 2:-2, :],
        )
        ttrkn_transp = update(
            ttrkn_transp,
            at[2:-2, 2:-2, :, 2], npx.diff(TTrkn_transp[2:-2, 2:-2, :, 2], axis=-1),
        )
        mttrkn_transp = update(
            mttrkn_transp,
            at[2:-2, 2:-2, :, 2], calc_mtt(state, sarkn_rz, ttrkn_transp[:, :, :, 2], vs.transp * h, msarkn_rz, vs.alpha_transp)[2:-2, 2:-2, :],
        )
        TTrkn_q_rz = update(
            TTrkn_q_rz,
            at[2:-2, 2:-2, :, 2], calc_omega_q(state, SArkn_rz, vs.sas_params_q_rz, vs.q_rz * h)[2:-2, 2:-2, :],
        )
        ttrkn_q_rz = update(
            ttrkn_q_rz,
            at[2:-2, 2:-2, :, 2], npx.diff(TTrkn_q_rz[2:-2, 2:-2, :, 2], axis=-1),
        )
        mttrkn_q_rz = update(
            mttrkn_q_rz,
            at[2:-2, 2:-2, :, 2], calc_mtt(state, sarkn_rz, ttrkn_q_rz[:, :, :, 2], vs.q_rz * h, msarkn_rz, vs.alpha_q)[2:-2, 2:-2, :],
        )
        TTrkn_cpr_rz = update(
            TTrkn_cpr_rz,
            at[2:-2, 2:-2, :, 2], calc_omega_q(state, SArkn_ss, vs.sas_params_cpr_rz, vs.cpr_rz * h)[2:-2, 2:-2, :],
        )
        ttrkn_cpr_rz = update(
            ttrkn_cpr_rz,
            at[2:-2, 2:-2, :, 2], npx.diff(TTrkn_cpr_rz[2:-2, 2:-2, :, 2], axis=-1),
        )
        mttrkn_cpr_rz = update(
            mttrkn_cpr_rz,
            at[2:-2, 2:-2, :, 2], calc_mtt(state, sarkn_ss, ttrkn_cpr_rz[:, :, :, 2], vs.cpr_rz * h, msarkn_ss, vs.alpha_q)[2:-2, 2:-2, :],
        )
        TTrkn_q_ss = update(
            TTrkn_q_ss,
            at[2:-2, 2:-2, :, 2], calc_omega_q(state, SArkn_ss, vs.sas_params_q_ss, vs.q_ss * h)[2:-2, 2:-2, :],
        )
        ttrkn_q_ss = update(
            ttrkn_q_ss,
            at[2:-2, 2:-2, :, 2], npx.diff(TTrkn_q_ss[2:-2, 2:-2, :, 2], axis=-1),
        )
        mttrkn_q_ss = update(
            mttrkn_q_ss,
            at[2:-2, 2:-2, :, 2], calc_mtt(state, sarkn_ss, ttrkn_q_ss[:, :, :, 2], vs.q_ss * h, msarkn_ss, vs.alpha_q)[2:-2, 2:-2, :],
        )

        sarkn_rz = update_add(
            sarkn_rz,
            at[2:-2, 2:-2, 1, :], (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] + vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 0] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * ttrkn_evap_soil[2:-2, 2:-2, :, 0] - vs.transp[2:-2, 2:-2, npx.newaxis] * ttrkn_transp[2:-2, 2:-2, :, 0] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 0]) * h,
        )
        sarkn_ss = update_add(
            sarkn_ss,
            at[2:-2, 2:-2, 1, :], (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] + vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_q_rz[2:-2, 2:-2, :, 0] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttrkn_cpr_rz[2:-2, 2:-2, :, 0] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * ttrkn_q_ss[2:-2, 2:-2, :, 0]) * h,
        )
        msarkn_rz = update_add(
            msarkn_rz,
            at[2:-2, 2:-2, 1, :], (npx.where(npx.isnan(vs.mtt_inf_mat_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_mat_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_inf_pf_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(mttrkn_cpr_rz[2:-2, 2:-2, :, 2]), 0, mttrkn_cpr_rz[2:-2, 2:-2, :, 2]) - npx.where(npx.isnan(mttrkn_evap_soil[2:-2, 2:-2, :, 2]), 0, mttrkn_evap_soil[2:-2, 2:-2, :, 2]) - npx.where(npx.isnan(mttrkn_transp[2:-2, 2:-2, :, 2]), 0, mttrkn_transp[2:-2, 2:-2, :, 2]) - npx.where(npx.isnan(mttrkn_q_rz[2:-2, 2:-2, :, 2]), 0, mttrkn_q_rz[2:-2, 2:-2, :, 2])))
        msarkn_ss = update_add(
            msarkn_ss,
            at[2:-2, 2:-2, 1, :], (npx.where(npx.isnan(vs.mtt_inf_pf_ss[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_ss[2:-2, 2:-2, :]) + npx.where(npx.isnan(mttrkn_q_rz[2:-2, 2:-2, :, 2]), 0, mttrkn_q_rz[2:-2, 2:-2, :, 2]) - npx.where(npx.isnan(mttrkn_cpr_rz[2:-2, 2:-2, :, 2]), 0, mttrkn_cpr_rz[2:-2, 2:-2, :, 2]) - npx.where(npx.isnan(mttrkn_q_ss[2:-2, 2:-2, :, 2]), 0, mttrkn_q_ss[2:-2, 2:-2, :, 2])) * h,
        )
        SArkn_rz = update(
            SArkn_rz,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(sarkn_rz[2:-2, 2:-2, 1, :], axis=-1),
        )
        SArkn_ss = update(
            SArkn_ss,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(sarkn_ss[2:-2, 2:-2, 1, :], axis=-1),
        )

        # step 4
        TTrkn_evap_soil = update(
            TTrkn_evap_soil,
            at[2:-2, 2:-2, :, 3], calc_omega_q(state, SArkn_rz, vs.sas_params_evap_soil, vs.evap_soil * h)[2:-2, 2:-2, :],
        )
        ttrkn_evap_soil = update(
            ttrkn_evap_soil,
            at[2:-2, 2:-2, :, 3], npx.diff(TTrkn_evap_soil[2:-2, 2:-2, :, 3], axis=-1),
        )
        if settings.enable_oxygen18 or settings.enable_deuterium:
            mttrkn_evap_soil = update(
                mttrkn_evap_soil,
                at[2:-2, 2:-2, :, 3], calc_mtt(state, sarkn_rz, ttrkn_evap_soil[:, :, :, 3], vs.evap_soil * h, msarkn_rz, vs.alpha_q)[2:-2, 2:-2, :],
            )
        TTrkn_transp = update(
            TTrkn_transp,
            at[2:-2, 2:-2, :, 3], calc_omega_q(state, SArkn_rz, vs.sas_params_transp, vs.transp * h)[2:-2, 2:-2, :],
        )
        ttrkn_transp = update(
            ttrkn_transp,
            at[2:-2, 2:-2, :, 3], npx.diff(TTrkn_transp[2:-2, 2:-2, :, 3], axis=-1),
        )
        mttrkn_transp = update(
            mttrkn_transp,
            at[2:-2, 2:-2, :, 3], calc_mtt(state, sarkn_rz, ttrkn_transp[:, :, :, 3], vs.transp * h, msarkn_rz, vs.alpha_transp)[2:-2, 2:-2, :],
        )
        TTrkn_q_rz = update(
            TTrkn_q_rz,
            at[2:-2, 2:-2, :, 3], calc_omega_q(state, SArkn_rz, vs.sas_params_q_rz, vs.q_rz * h)[2:-2, 2:-2, :],
        )
        ttrkn_q_rz = update(
            ttrkn_q_rz,
            at[2:-2, 2:-2, :, 3], npx.diff(TTrkn_q_rz[2:-2, 2:-2, :, 3], axis=-1),
        )
        mttrkn_q_rz = update(
            mttrkn_q_rz,
            at[2:-2, 2:-2, :, 3], calc_mtt(state, sarkn_rz, ttrkn_q_rz[:, :, :, 3], vs.q_rz * h, msarkn_rz, vs.alpha_q)[2:-2, 2:-2, :],
        )
        TTrkn_cpr_rz = update(
            TTrkn_cpr_rz,
            at[2:-2, 2:-2, :, 3], calc_omega_q(state, SArkn_ss, vs.sas_params_cpr_rz, vs.cpr_rz * h)[2:-2, 2:-2, :],
        )
        ttrkn_cpr_rz = update(
            ttrkn_cpr_rz,
            at[2:-2, 2:-2, :, 3], npx.diff(TTrkn_cpr_rz[2:-2, 2:-2, :, 3], axis=-1),
        )
        mttrkn_cpr_rz = update(
            mttrkn_cpr_rz,
            at[2:-2, 2:-2, :, 3], calc_mtt(state, sarkn_ss, ttrkn_cpr_rz[:, :, :, 3], vs.cpr_rz * h, msarkn_ss, vs.alpha_q)[2:-2, 2:-2, :],
        )
        TTrkn_q_ss = update(
            TTrkn_q_ss,
            at[2:-2, 2:-2, :, 3], calc_omega_q(state, SArkn_ss, vs.sas_params_q_ss, vs.q_ss * h)[2:-2, 2:-2, :],
        )
        ttrkn_q_ss = update(
            ttrkn_q_ss,
            at[2:-2, 2:-2, :, 3], npx.diff(TTrkn_q_ss[2:-2, 2:-2, :, 3], axis=-1),
        )
        mttrkn_q_ss = update(
            mttrkn_q_ss,
            at[2:-2, 2:-2, :, 3], calc_mtt(state, sarkn_ss, ttrkn_q_ss[:, :, :, 3], vs.q_ss * h, msarkn_ss, vs.alpha_q)[2:-2, 2:-2, :],
        )

        ttn_evap_soil = update(
            ttn_evap_soil,
            at[2:-2, 2:-2, :, i], (ttrkn_evap_soil[2:-2, 2:-2, :, 0] + 2*ttrkn_evap_soil[2:-2, 2:-2, :, 1] + 2*ttrkn_evap_soil[2:-2, 2:-2, :, 2] + ttrkn_evap_soil[2:-2, 2:-2, :, 3]) / 6.,
        )
        ttn_transp = update(
            ttn_transp,
            at[2:-2, 2:-2, :, i], (ttrkn_transp[2:-2, 2:-2, :, 0] + 2*ttrkn_transp[2:-2, 2:-2, :, 1] + 2*ttrkn_transp[2:-2, 2:-2, :, 2] + ttrkn_transp[2:-2, 2:-2, :, 3]) / 6.,
        )
        ttn_q_rz = update(
            ttn_q_rz,
            at[2:-2, 2:-2, :, i], (ttrkn_q_rz[2:-2, 2:-2, :, 0] + 2*ttrkn_q_rz[2:-2, 2:-2, :, 1] + 2*ttrkn_q_rz[2:-2, 2:-2, :, 2] + ttrkn_q_rz[2:-2, 2:-2, :, 3]) / 6.,
        )
        ttn_cpr_rz = update(
            ttn_cpr_rz,
            at[2:-2, 2:-2, :, i], (ttrkn_cpr_rz[2:-2, 2:-2, :, 0] + 2*ttrkn_cpr_rz[2:-2, 2:-2, :, 1] + 2*ttrkn_cpr_rz[2:-2, 2:-2, :, 2] + ttrkn_cpr_rz[2:-2, 2:-2, :, 3]) / 6.,
        )
        ttn_q_ss = update(
            ttn_q_ss,
            at[2:-2, 2:-2, :, i], (ttrkn_q_ss[2:-2, 2:-2, :, 0] + 2*ttrkn_q_ss[2:-2, 2:-2, :, 1] + 2*ttrkn_q_ss[2:-2, 2:-2, :, 2] + ttrkn_q_ss[2:-2, 2:-2, :, 3]) / 6.,
        )

        mttn_evap_soil = update(
            mttn_evap_soil,
            at[2:-2, 2:-2, :, i], (mttrkn_evap_soil[2:-2, 2:-2, :, 0] + 2*mttrkn_evap_soil[2:-2, 2:-2, :, 1] + 2*mttrkn_evap_soil[2:-2, 2:-2, :, 2] + mttrkn_evap_soil[2:-2, 2:-2, :, 3]) / 6.,
        )
        mttn_transp = update(
            mttn_transp,
            at[2:-2, 2:-2, :, i], (mttrkn_transp[2:-2, 2:-2, :, 0] + 2*mttrkn_transp[2:-2, 2:-2, :, 1] + 2*mttrkn_transp[2:-2, 2:-2, :, 2] + mttrkn_transp[2:-2, 2:-2, :, 3]) / 6.,
        )
        mttn_q_rz = update(
            mttn_q_rz,
            at[2:-2, 2:-2, :, i], (mttrkn_q_rz[2:-2, 2:-2, :, 0] + 2*mttrkn_q_rz[2:-2, 2:-2, :, 1] + 2*mttrkn_q_rz[2:-2, 2:-2, :, 2] + mttrkn_q_rz[2:-2, 2:-2, :, 3]) / 6.,
        )
        mttn_cpr_rz = update(
            mttn_cpr_rz,
            at[2:-2, 2:-2, :, i], (mttrkn_cpr_rz[2:-2, 2:-2, :, 0] + 2*mttrkn_cpr_rz[2:-2, 2:-2, :, 1] + 2*mttrkn_cpr_rz[2:-2, 2:-2, :, 2] + mttrkn_cpr_rz[2:-2, 2:-2, :, 3]) / 6.,
        )
        mttn_q_ss = update(
            mttn_q_ss,
            at[2:-2, 2:-2, :, i], (mttrkn_q_ss[2:-2, 2:-2, :, 0] + 2*mttrkn_q_ss[2:-2, 2:-2, :, 1] + 2*mttrkn_q_ss[2:-2, 2:-2, :, 2] + mttrkn_q_ss[2:-2, 2:-2, :, 3]) / 6.,
        )

        san_rz = update_add(
            san_rz,
            at[2:-2, 2:-2, 1, :], (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] + vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttn_cpr_rz[2:-2, 2:-2, :, i] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * ttn_evap_soil[2:-2, 2:-2, :, i] - vs.transp[2:-2, 2:-2, npx.newaxis] * ttn_transp[2:-2, 2:-2, :, i] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttn_q_rz[2:-2, 2:-2, :, i]) * h,
        )
        san_ss = update_add(
            san_ss,
            at[2:-2, 2:-2, 1, :], (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] + vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttn_q_rz[2:-2, 2:-2, :, i] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttn_cpr_rz[2:-2, 2:-2, :, i] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * ttn_q_ss[2:-2, 2:-2, :, i]) * h,
        )
        SAn_rz = update(
            SAn_rz,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(san_rz[2:-2, 2:-2, 1, :], axis=-1),
        )
        SAn_ss = update(
            SAn_ss,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(san_ss[2:-2, 2:-2, 1, :], axis=-1),
        )
        msan_rz = update_add(
            msan_rz,
            at[2:-2, 2:-2, 1, :], (npx.where(npx.isnan(vs.mtt_inf_mat_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_mat_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_inf_pf_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(mttn_cpr_rz[2:-2, 2:-2, :, i]), 0, mttn_cpr_rz[2:-2, 2:-2, :, i]) - npx.where(npx.isnan(mttn_evap_soil[2:-2, 2:-2, :, i]), 0, mttn_evap_soil[2:-2, 2:-2, :, i]) - npx.where(npx.isnan(mttn_transp[2:-2, 2:-2, :, i]), 0, mttn_transp[2:-2, 2:-2, :, i]) - npx.where(npx.isnan(mttn_q_rz[2:-2, 2:-2, :, i]), 0, mttn_q_rz[2:-2, 2:-2, :, i])) * h,
        )
        msan_ss = update_add(
            msan_ss,
            at[2:-2, 2:-2, 1, :], (npx.where(npx.isnan(vs.mtt_inf_pf_ss[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_ss[2:-2, 2:-2, :]) + npx.where(npx.isnan(mttn_q_rz[2:-2, 2:-2, :, i]), 0, mttn_q_rz[2:-2, 2:-2, :, i]) - npx.where(npx.isnan(mttn_cpr_rz[2:-2, 2:-2, :, i]), 0, mttn_cpr_rz[2:-2, 2:-2, :, i]) - npx.where(npx.isnan(mttn_q_ss[2:-2, 2:-2, :, i]), 0, mttn_q_ss[2:-2, 2:-2, :, i])) * h,
        )

        carry_tt_rk = (TTrkn_evap_soil, ttrkn_evap_soil, TTrkn_transp, ttrkn_transp, TTrkn_cpr_rz, ttrkn_cpr_rz, TTrkn_q_rz, ttrkn_q_rz, TTrkn_q_ss, ttrkn_q_ss)
        carry_mtt_rk = (mttrkn_evap_soil, mttrkn_transp, mttrkn_cpr_rz, mttrkn_q_rz, mttrkn_q_ss)
        carry_tt_substeps = (ttn_evap_soil, ttn_transp, ttn_cpr_rz, ttn_q_rz, ttn_q_ss)
        carry_mtt_substeps = (mttn_evap_soil, mttn_transp, mttn_cpr_rz, mttn_q_rz, mttn_q_ss)
        carry_sa_substeps = (SAn_rz, san_rz, SAn_ss, san_ss, msan_rz, msan_ss)
        carry = (state, carry_tt_rk, carry_mtt_rk, carry_tt_substeps, carry_mtt_substeps, carry_sa_substeps)

        return carry, None

    n_substeps = npx.arange(0, settings.sas_solver_substeps, dtype=int)

    carry = (state, carry_tt_rk, carry_mtt_rk, carry_tt_substeps, carry_mtt_substeps, carry_sa_substeps)
    res, _ = scan(body, carry, n_substeps)
    res_tt = res[3]
    ttn_evap_soil = res_tt[0]
    ttn_transp = res_tt[1]
    ttn_cpr_rz = res_tt[2]
    ttn_q_rz = res_tt[3]
    ttn_q_ss = res_tt[4]
    res_mtt = res[4]
    mttn_evap_soil = res_mtt[0]
    mttn_transp = res_mtt[1]
    mttn_cpr_rz = res_mtt[2]
    mttn_q_rz = res_mtt[3]
    mttn_q_ss = res_mtt[4]

    # backward travel time distributions
    vs.tt_evap_soil = update(
        vs.tt_evap_soil,
        at[2:-2, 2:-2, :], npx.sum(ttn_transp / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.tt_transp = update(
        vs.tt_transp,
        at[2:-2, 2:-2, :], npx.sum(ttn_transp / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.tt_q_rz = update(
        vs.tt_q_rz,
        at[2:-2, 2:-2, :], npx.sum(ttn_transp / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.tt_cpr_rz = update(
        vs.tt_cpr_rz,
        at[2:-2, 2:-2, :], npx.sum(ttn_transp / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.tt_q_ss = update(
        vs.tt_q_ss,
        at[2:-2, 2:-2, :], npx.sum(ttn_transp / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.mtt_evap_soil = update(
        vs.mtt_evap_soil,
        at[2:-2, 2:-2, :], npx.sum(mttn_evap_soil, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.mtt_transp = update(
        vs.mtt_transp,
        at[2:-2, 2:-2, :], npx.sum(mttn_transp, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.mtt_q_rz = update(
        vs.mtt_q_rz,
        at[2:-2, 2:-2, :], npx.sum(mttn_q_rz, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.mtt_cpr_rz = update(
        vs.mtt_cpr_rz,
        at[2:-2, 2:-2, :], npx.sum(mttn_cpr_rz, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.mtt_q_ss = update(
        vs.mtt_q_ss,
        at[2:-2, 2:-2, :], npx.sum(mttn_q_ss, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    # cumulative backward travel time distributions
    vs.TT_evap_soil = update(
        vs.TT_evap_soil,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_evap_soil, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.TT_transp = update(
        vs.TT_transp,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_transp, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.TT_q_rz = update(
        vs.TT_q_rz,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_rz, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.TT_cpr_rz = update(
        vs.TT_cpr_rz,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_cpr_rz, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.TT_q_ss = update(
        vs.TT_q_ss,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_ss, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    # update StorAge
    vs.sa_rz = update_add(
        vs.sa_rz,
        at[2:-2, 2:-2, 1, :], (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] + vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :] - vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :]),
    )
    vs.sa_ss = update_add(
        vs.sa_ss,
        at[2:-2, 2:-2, 1, :], vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] + vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :],
    )
    vs.msa_rz = update_add(
        vs.msa_rz,
        at[2:-2, 2:-2, 1, :], npx.where(npx.isnan(vs.mtt_inf_mat_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_mat_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_inf_pf_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_cpr_rz[2:-2, 2:-2, :]), 0, vs.mtt_cpr_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_evap_soil[2:-2, 2:-2, :]), 0, vs.mtt_evap_soil[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_transp[2:-2, 2:-2, :]), 0, vs.mtt_transp[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_q_rz[2:-2, 2:-2, :]), 0, vs.mtt_q_rz[2:-2, 2:-2, :]),
    )
    vs.msa_ss = update_add(
        vs.msa_ss,
        at[2:-2, 2:-2, 1, :], npx.where(npx.isnan(vs.mtt_inf_pf_ss[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_ss[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_q_rz[2:-2, 2:-2, :]), 0, vs.mtt_q_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_cpr_rz[2:-2, 2:-2, :]), 0, vs.mtt_cpr_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_q_ss[2:-2, 2:-2, :]), 0, vs.mtt_q_ss[2:-2, 2:-2, :]),
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
def transport_model_euler(state):
    """Calculates water transport model with Euler method
    """
    vs = state.variables
    settings = state.settings
    h = 1 / settings.sas_solver_substeps

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :], calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    SAn_rz = allocate(state.dimensions, ("x", "y", "timesteps", "nages"))
    san_rz = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
    SAn_rz = update(
        SAn_rz,
        at[2:-2, 2:-2, :, :], vs.SA_rz[2:-2, 2:-2, :, :],
    )
    san_rz = update(
        san_rz,
        at[2:-2, 2:-2, :, :], vs.sa_rz[2:-2, 2:-2, :, :],
    )

    SAn_ss = allocate(state.dimensions, ("x", "y", "timesteps", "nages"))
    san_ss = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
    SAn_ss = update(
        SAn_ss,
        at[2:-2, 2:-2, :, :], vs.SA_ss[2:-2, 2:-2, :, :],
    )
    san_ss = update(
        san_ss,
        at[2:-2, 2:-2, :, :], vs.sa_ss[2:-2, 2:-2, :, :],
    )

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

    TTn_evap_soil = allocate(state.dimensions, ("x", "y", "nages", settings.sas_solver_substeps))
    TTn_transp = allocate(state.dimensions, ("x", "y", "nages", settings.sas_solver_substeps))
    TTn_cpr_rz = allocate(state.dimensions, ("x", "y", "nages", settings.sas_solver_substeps))
    TTn_q_rz = allocate(state.dimensions, ("x", "y", "nages", settings.sas_solver_substeps))
    TTn_q_ss = allocate(state.dimensions, ("x", "y", "nages", settings.sas_solver_substeps))
    ttn_evap_soil = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    ttn_transp = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    ttn_cpr_rz = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    ttn_q_rz = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    ttn_q_ss = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))

    carry_substeps_tt = (ttn_evap_soil, ttn_transp, ttn_cpr_rz, ttn_q_rz, ttn_q_ss)
    carry_substeps_TT = (TTn_evap_soil, TTn_transp, TTn_cpr_rz, TTn_q_rz, TTn_q_ss)
    carry_substeps_sa = (SAn_rz, san_rz, SAn_ss, san_ss)

    # loop to solve SAS functions numerically
    def body(carry, i):
        state, carry_substeps_tt, carry_substeps_TT, carry_substeps_sa = carry
        ttn_evap_soil, ttn_transp, ttn_cpr_rz, ttn_q_rz, ttn_q_ss = carry_substeps_tt
        TTn_evap_soil, TTn_transp, TTn_cpr_rz, TTn_q_rz, TTn_q_ss = carry_substeps_TT
        SAn_rz, san_rz, SAn_ss, san_ss = carry_substeps_sa
        vs = state.variables

        # step i
        TTn_evap_soil = update(
            TTn_evap_soil,
            at[2:-2, 2:-2, :, i], calc_omega_q(state, SAn_rz, vs.sas_params_evap_soil, vs.evap_soil * h)[2:-2, 2:-2, :],
        )
        ttn_evap_soil = update(
            ttn_evap_soil,
            at[2:-2, 2:-2, :, i], npx.diff(TTn_evap_soil[2:-2, 2:-2, :, i], axis=-1),
        )
        TTn_transp = update(
            TTn_transp,
            at[2:-2, 2:-2, :, i], calc_omega_q(state, SAn_rz, vs.sas_params_transp, vs.transp * h)[2:-2, 2:-2, :],
        )
        ttn_transp = update(
            ttn_transp,
            at[2:-2, 2:-2, :, i], npx.diff(TTn_transp[2:-2, 2:-2, :, i], axis=-1),
        )
        TTn_q_rz = update(
            TTn_q_rz,
            at[2:-2, 2:-2, :, i], calc_omega_q(state, SAn_rz, vs.sas_params_q_rz, vs.q_rz * h)[2:-2, 2:-2, :],
        )
        ttn_q_rz = update(
            ttn_q_rz,
            at[2:-2, 2:-2, :, i], npx.diff(TTn_q_rz[2:-2, 2:-2, :, i], axis=-1),
        )
        TTn_cpr_rz = update(
            TTn_cpr_rz,
            at[2:-2, 2:-2, :, i], calc_omega_q(state, SAn_ss, vs.sas_params_cpr_rz, vs.cpr_rz * h)[2:-2, 2:-2, :],
        )
        ttn_cpr_rz = update(
            ttn_cpr_rz,
            at[2:-2, 2:-2, :, i], npx.diff(TTn_cpr_rz[2:-2, 2:-2, :, i], axis=-1),
        )
        TTn_q_ss = update(
            TTn_q_ss,
            at[2:-2, 2:-2, :, i], calc_omega_q(state, SAn_ss, vs.sas_params_q_ss, vs.q_ss * h)[2:-2, 2:-2, :],
        )
        ttn_q_ss = update(
            ttn_q_ss,
            at[2:-2, 2:-2, :, i], npx.diff(TTn_q_ss[2:-2, 2:-2, :, i], axis=-1),
        )

        san_rz = update_add(
            san_rz,
            at[2:-2, 2:-2, 1, :], (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] + vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttn_cpr_rz[2:-2, 2:-2, :, i] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * ttn_evap_soil[2:-2, 2:-2, :, i] - vs.transp[2:-2, 2:-2, npx.newaxis] * ttn_transp[2:-2, 2:-2, :, i] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttn_q_rz[2:-2, 2:-2, :, i]) * h,
        )
        san_ss = update_add(
            san_ss,
            at[2:-2, 2:-2, 1, :], (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] + vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttn_q_rz[2:-2, 2:-2, :, i] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttn_cpr_rz[2:-2, 2:-2, :, i] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * ttn_q_ss[2:-2, 2:-2, :, i]) * h,
        )
        SAn_rz = update(
            SAn_rz,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(san_rz[2:-2, 2:-2, 1, :], axis=-1),
        )
        SAn_ss = update(
            SAn_ss,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(san_ss[2:-2, 2:-2, 1, :], axis=-1),
        )

        carry_substeps_tt = (ttn_evap_soil, ttn_transp, ttn_cpr_rz, ttn_q_rz, ttn_q_ss)
        carry_substeps_TT = (TTn_evap_soil, TTn_transp, TTn_cpr_rz, TTn_q_rz, TTn_q_ss)
        carry_substeps_sa = (SAn_rz, san_rz, SAn_ss, san_ss)

        carry = (state, carry_substeps_tt, carry_substeps_TT, carry_substeps_sa)

        return carry, None

    n_substeps = npx.arange(0, settings.sas_solver_substeps, dtype=int)

    carry = (state, carry_substeps_tt, carry_substeps_TT, carry_substeps_sa)
    res, _ = scan(body, carry, n_substeps)
    res_substeps = res[1]
    ttn_evap_soil = res_substeps[0]
    ttn_transp = res_substeps[1]
    ttn_cpr_rz = res_substeps[2]
    ttn_q_rz = res_substeps[3]
    ttn_q_ss = res_substeps[4]

    # backward travel time distributions
    vs.tt_evap_soil = update(
        vs.tt_evap_soil,
        at[2:-2, 2:-2, :], npx.sum(ttn_evap_soil / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.tt_transp = update(
        vs.tt_transp,
        at[2:-2, 2:-2, :], npx.sum(ttn_transp / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.tt_q_rz = update(
        vs.tt_q_rz,
        at[2:-2, 2:-2, :], npx.sum(ttn_q_rz / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.tt_cpr_rz = update(
        vs.tt_cpr_rz,
        at[2:-2, 2:-2, :], npx.sum(ttn_cpr_rz / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.tt_q_ss = update(
        vs.tt_q_ss,
        at[2:-2, 2:-2, :], npx.sum(ttn_q_ss / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    # cumulative backward travel time distributions
    vs.TT_evap_soil = update(
        vs.TT_evap_soil,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_evap_soil, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.TT_transp = update(
        vs.TT_transp,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_transp, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.TT_q_rz = update(
        vs.TT_q_rz,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_rz, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.TT_cpr_rz = update(
        vs.TT_cpr_rz,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_cpr_rz, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.TT_q_ss = update(
        vs.TT_q_ss,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_ss, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    # update StorAge
    vs.sa_rz = update_add(
        vs.sa_rz,
        at[2:-2, 2:-2, 1, :], (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] + vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :] - vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :]),
    )
    vs.sa_ss = update_add(
        vs.sa_ss,
        at[2:-2, 2:-2, 1, :], (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] + vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :]),
    )

    return KernelOutput(tt_evap_soil=vs.tt_evap_soil, tt_transp=vs.tt_transp, tt_q_rz=vs.tt_q_rz, tt_cpr_rz=vs.tt_cpr_rz, tt_q_ss=vs.tt_q_ss,
                        TT_evap_soil=vs.TT_evap_soil, TT_transp=vs.TT_transp, TT_q_rz=vs.TT_q_rz, TT_cpr_rz=vs.TT_cpr_rz, TT_q_ss=vs.TT_q_ss,
                        sa_rz=vs.sa_rz, sa_ss=vs.sa_ss)


@roger_kernel
def solute_transport_model_euler(state):
    """Calculates solute transport model with Euler method
    """
    vs = state.variables
    settings = state.settings
    h = 1 / settings.sas_solver_substeps

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :], calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    SAn_rz = allocate(state.dimensions, ("x", "y", "timesteps", "nages"))
    san_rz = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
    msan_rz = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
    SAn_rz = update(
        SAn_rz,
        at[2:-2, 2:-2, :, :], vs.SA_rz[2:-2, 2:-2, :, :],
    )
    san_rz = update(
        san_rz,
        at[2:-2, 2:-2, :, :], vs.sa_rz[2:-2, 2:-2, :, :],
    )
    msan_rz = update(
        msan_rz,
        at[2:-2, 2:-2, :, :], vs.msa_rz[2:-2, 2:-2, :, :],
    )

    SAn_ss = allocate(state.dimensions, ("x", "y", "timesteps", "nages"))
    san_ss = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
    msan_ss = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
    SAn_ss = update(
        SAn_ss,
        at[2:-2, 2:-2, :, :], vs.SA_ss[2:-2, 2:-2, :, :],
    )
    san_ss = update(
        san_ss,
        at[2:-2, 2:-2, :, :], vs.sa_ss[2:-2, 2:-2, :, :],
    )
    msan_ss = update(
        msan_ss,
        at[2:-2, 2:-2, :, :], vs.msa_ss[2:-2, 2:-2, :, :],
    )

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
            at[2:-2, 2:-2, 0], npx.where(vs.inf_mat_rz[2:-2, 2:-2] > 0, vs.inf_mat_rz[2:-2, 2:-2] * vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_mat_rz = update(
            vs.mtt_inf_mat_rz,
            at[2:-2, 2:-2, :], npx.where(vs.mtt_inf_mat_rz == 0, npx.nan, vs.mtt_inf_mat_rz)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
        )
        vs.mtt_inf_pf_rz = update(
            vs.mtt_inf_pf_rz,
            at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_rz[2:-2, 2:-2] > 0, vs.inf_pf_rz[2:-2, 2:-2] * vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_pf_rz = update(
            vs.mtt_inf_pf_rz,
            at[2:-2, 2:-2, :], npx.where(vs.mtt_inf_pf_rz == 0, npx.nan, vs.mtt_inf_pf_rz)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
        )
        vs.mtt_inf_pf_ss = update(
            vs.mtt_inf_pf_ss,
            at[2:-2, 2:-2, 0], npx.where(vs.inf_pf_ss[2:-2, 2:-2] > 0, vs.inf_pf_ss[2:-2, 2:-2] * vs.C_in[2:-2, 2:-2], npx.nan) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.mtt_inf_pf_ss = update(
            vs.mtt_inf_pf_ss,
            at[2:-2, 2:-2, :], npx.where(vs.mtt_inf_pf_ss == 0, npx.nan, vs.mtt_inf_pf_ss)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
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

    TTn_evap_soil = allocate(state.dimensions, ("x", "y", "nages", settings.sas_solver_substeps))
    TTn_transp = allocate(state.dimensions, ("x", "y", "nages", settings.sas_solver_substeps))
    TTn_cpr_rz = allocate(state.dimensions, ("x", "y", "nages", settings.sas_solver_substeps))
    TTn_q_rz = allocate(state.dimensions, ("x", "y", "nages", settings.sas_solver_substeps))
    TTn_q_ss = allocate(state.dimensions, ("x", "y", "nages", settings.sas_solver_substeps))
    ttn_evap_soil = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    ttn_transp = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    ttn_cpr_rz = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    ttn_q_rz = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    ttn_q_ss = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    mttn_evap_soil = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    mttn_transp = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    mttn_cpr_rz = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    mttn_q_rz = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))
    mttn_q_ss = allocate(state.dimensions, ("x", "y", "ages", settings.sas_solver_substeps))

    carry_substeps_tt = (ttn_evap_soil, ttn_transp, ttn_cpr_rz, ttn_q_rz, ttn_q_ss)
    carry_substeps_mtt = (mttn_evap_soil, mttn_transp, mttn_cpr_rz, mttn_q_rz, mttn_q_ss)
    carry_substeps_TT = (TTn_evap_soil, TTn_transp, TTn_cpr_rz, TTn_q_rz, TTn_q_ss)
    carry_substeps_sa = (SAn_rz, san_rz, SAn_ss, san_ss)
    carry_substeps_msa = (msan_rz, msan_ss)

    # loop to solve SAS functions numerically
    def body(carry, i):
        state, carry_substeps_tt, carry_substeps_mtt, carry_substeps_TT, carry_substeps_sa, carry_substeps_msa = carry
        ttn_evap_soil, ttn_transp, ttn_cpr_rz, ttn_q_rz, ttn_q_ss = carry_substeps_tt
        mttn_evap_soil, mttn_transp, mttn_cpr_rz, mttn_q_rz, mttn_q_ss = carry_substeps_mtt
        TTn_evap_soil, TTn_transp, TTn_cpr_rz, TTn_q_rz, TTn_q_ss = carry_substeps_TT
        SAn_rz, san_rz, SAn_ss, san_ss = carry_substeps_sa
        msan_rz, msan_ss = carry_substeps_msa
        vs = state.variables
        settings = state.settings

        # step i
        TTn_evap_soil = update(
            TTn_evap_soil,
            at[2:-2, 2:-2, :, i], calc_omega_q(state, SAn_rz, vs.sas_params_evap_soil, vs.evap_soil * h)[2:-2, 2:-2, :],
        )
        ttn_evap_soil = update(
            ttn_evap_soil,
            at[2:-2, 2:-2, :, i], npx.diff(TTn_evap_soil[2:-2, 2:-2, :, i], axis=-1),
        )
        if settings.enable_oxygen18 or settings.enable_deuterium:
            mttn_evap_soil = update(
                mttn_evap_soil,
                at[2:-2, 2:-2, :, i], calc_mtt(state, san_rz, ttn_evap_soil[:, :, :, i], vs.evap_soil * h, msan_rz, vs.alpha_q)[2:-2, 2:-2, :],
            )
        TTn_transp = update(
            TTn_transp,
            at[2:-2, 2:-2, :, i], calc_omega_q(state, SAn_rz, vs.sas_params_transp, vs.transp * h)[2:-2, 2:-2, :],
        )
        ttn_transp = update(
            ttn_transp,
            at[2:-2, 2:-2, :, i], npx.diff(TTn_transp[2:-2, 2:-2, :, i], axis=-1),
        )
        mttn_transp = update(
            mttn_transp,
            at[2:-2, 2:-2, :, i], calc_mtt(state, san_rz, ttn_transp[:, :, :, i], vs.transp * h, msan_rz, vs.alpha_transp)[2:-2, 2:-2, :],
        )
        TTn_q_rz = update(
            TTn_q_rz,
            at[2:-2, 2:-2, :, i], calc_omega_q(state, SAn_rz, vs.sas_params_q_rz, vs.q_rz * h)[2:-2, 2:-2, :],
        )
        ttn_q_rz = update(
            ttn_q_rz,
            at[2:-2, 2:-2, :, i], npx.diff(TTn_q_rz[2:-2, 2:-2, :, i], axis=-1),
        )
        mttn_q_rz = update(
            mttn_q_rz,
            at[2:-2, 2:-2, :, i], calc_mtt(state, san_rz, ttn_q_rz[:, :, :, i], vs.q_rz * h, msan_rz, vs.alpha_q)[2:-2, 2:-2, :],
        )
        TTn_cpr_rz = update(
            TTn_cpr_rz,
            at[2:-2, 2:-2, :, i], calc_omega_q(state, SAn_ss, vs.sas_params_cpr_rz, vs.cpr_rz * h)[2:-2, 2:-2, :],
        )
        ttn_cpr_rz = update(
            ttn_cpr_rz,
            at[2:-2, 2:-2, :, i], npx.diff(TTn_cpr_rz[2:-2, 2:-2, :, i], axis=-1),
        )
        mttn_cpr_rz = update(
            mttn_cpr_rz,
            at[2:-2, 2:-2, :, i], calc_mtt(state, san_ss, ttn_cpr_rz[:, :, :, i], vs.cpr_rz * h, msan_ss, vs.alpha_q)[2:-2, 2:-2, :],
        )
        TTn_q_ss = update(
            TTn_q_ss,
            at[2:-2, 2:-2, :, i], calc_omega_q(state, SAn_ss, vs.sas_params_q_ss, vs.q_ss * h)[2:-2, 2:-2, :],
        )
        ttn_q_ss = update(
            ttn_q_ss,
            at[2:-2, 2:-2, :, i], npx.diff(TTn_q_ss[2:-2, 2:-2, :, i], axis=-1),
        )
        mttn_q_ss = update(
            mttn_q_ss,
            at[2:-2, 2:-2, :, i], calc_mtt(state, san_ss, ttn_cpr_rz[:, :, :, i], vs.q_ss * h, msan_ss, vs.alpha_q)[2:-2, 2:-2, :],
        )

        san_rz = update_add(
            san_rz,
            at[2:-2, 2:-2, 1, :], (vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] + vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttn_cpr_rz[2:-2, 2:-2, :, i] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * ttn_evap_soil[2:-2, 2:-2, :, i] - vs.transp[2:-2, 2:-2, npx.newaxis] * ttn_transp[2:-2, 2:-2, :, i] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttn_q_rz[2:-2, 2:-2, :, i]) * h,
        )
        san_ss = update_add(
            san_ss,
            at[2:-2, 2:-2, 1, :], (vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] + vs.q_rz[2:-2, 2:-2, npx.newaxis] * ttn_q_rz[2:-2, 2:-2, :, i] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * ttn_cpr_rz[2:-2, 2:-2, :, i] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * ttn_q_ss[2:-2, 2:-2, :, i]) * h,
        )
        msan_rz = update_add(
            msan_rz,
            at[2:-2, 2:-2, 1, :], (npx.where(npx.isnan(vs.mtt_inf_mat_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_mat_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_inf_pf_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(mttn_cpr_rz[2:-2, 2:-2, :, i]), 0, mttn_cpr_rz[2:-2, 2:-2, :, i]) - npx.where(npx.isnan(mttn_evap_soil[2:-2, 2:-2, :, i]), 0, mttn_evap_soil[2:-2, 2:-2, :, i]) - npx.where(npx.isnan(mttn_transp[2:-2, 2:-2, :, i]), 0, mttn_transp[2:-2, 2:-2, :, i]) - npx.where(npx.isnan(mttn_q_rz[2:-2, 2:-2, :, i]), 0, mttn_q_rz[2:-2, 2:-2, :, i])) * h,
        )
        msan_ss = update_add(
            msan_ss,
            at[2:-2, 2:-2, 1, :], (npx.where(npx.isnan(vs.mtt_inf_pf_ss[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_ss[2:-2, 2:-2, :]) + npx.where(npx.isnan(mttn_q_rz[2:-2, 2:-2, :, i]), 0, mttn_q_rz[2:-2, 2:-2, :, i]) - npx.where(npx.isnan(mttn_cpr_rz[2:-2, 2:-2, :, i]), 0, mttn_cpr_rz[2:-2, 2:-2, :, i]) - npx.where(npx.isnan(mttn_q_ss[2:-2, 2:-2, :, i]), 0, mttn_q_ss[2:-2, 2:-2, :, i])) * h,
        )
        SAn_rz = update(
            SAn_rz,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(san_rz[2:-2, 2:-2, 1, :], axis=-1),
        )
        SAn_ss = update(
            SAn_ss,
            at[2:-2, 2:-2, 1, 1:], npx.cumsum(san_ss[2:-2, 2:-2, 1, :], axis=-1),
        )

        carry_substeps_tt = (ttn_evap_soil, ttn_transp, ttn_cpr_rz, ttn_q_rz, ttn_q_ss)
        carry_substeps_mtt = (mttn_evap_soil, mttn_transp, mttn_cpr_rz, mttn_q_rz, mttn_q_ss)
        carry_substeps_TT = (TTn_evap_soil, TTn_transp, TTn_cpr_rz, TTn_q_rz, TTn_q_ss)
        carry_substeps_sa = (SAn_rz, san_rz, SAn_ss, san_ss)
        carry_substeps_msa = (msan_rz, msan_ss)

        carry = (state, carry_substeps_tt, carry_substeps_mtt, carry_substeps_TT, carry_substeps_sa, carry_substeps_msa)

        return carry, None

    n_substeps = npx.arange(0, settings.sas_solver_substeps, dtype=int)

    carry = (state, carry_substeps_tt, carry_substeps_mtt, carry_substeps_TT, carry_substeps_sa, carry_substeps_msa)
    res, _ = scan(body, carry, n_substeps)
    res_tt = res[1]
    ttn_evap_soil = res_tt[0]
    ttn_transp = res_tt[1]
    ttn_cpr_rz = res_tt[2]
    ttn_q_rz = res_tt[3]
    ttn_q_ss = res_tt[4]
    res_mtt = res[2]
    mttn_evap_soil = res_mtt[0]
    mttn_transp = res_mtt[1]
    mttn_cpr_rz = res_mtt[2]
    mttn_q_rz = res_mtt[3]
    mttn_q_ss = res_mtt[4]

    # backward travel time distributions
    vs.tt_evap_soil = update(
        vs.tt_evap_soil,
        at[2:-2, 2:-2, :], npx.sum(ttn_evap_soil / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.tt_transp = update(
        vs.tt_transp,
        at[2:-2, 2:-2, :], npx.sum(ttn_transp / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.tt_q_rz = update(
        vs.tt_q_rz,
        at[2:-2, 2:-2, :], npx.sum(ttn_q_rz / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.tt_cpr_rz = update(
        vs.tt_cpr_rz,
        at[2:-2, 2:-2, :], npx.sum(ttn_cpr_rz / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.tt_q_ss = update(
        vs.tt_q_ss,
        at[2:-2, 2:-2, :], npx.sum(ttn_q_ss / settings.sas_solver_substeps, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.mtt_evap_soil = update(
        vs.mtt_evap_soil,
        at[2:-2, 2:-2, :], npx.sum(mttn_evap_soil, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.mtt_transp = update(
        vs.mtt_transp,
        at[2:-2, 2:-2, :], npx.sum(mttn_transp, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.mtt_q_rz = update(
        vs.mtt_q_rz,
        at[2:-2, 2:-2, :], npx.sum(mttn_q_rz, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.mtt_cpr_rz = update(
        vs.mtt_cpr_rz,
        at[2:-2, 2:-2, :], npx.sum(mttn_cpr_rz, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.mtt_q_ss = update(
        vs.mtt_q_ss,
        at[2:-2, 2:-2, :], npx.sum(mttn_q_ss, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    # cumulative backward travel time distributions
    vs.TT_evap_soil = update(
        vs.TT_evap_soil,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_evap_soil, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.TT_transp = update(
        vs.TT_transp,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_transp, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.TT_q_rz = update(
        vs.TT_q_rz,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_rz, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.TT_cpr_rz = update(
        vs.TT_cpr_rz,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_cpr_rz, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.TT_q_ss = update(
        vs.TT_q_ss,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_ss, axis=-1)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    # update StorAge
    vs.sa_rz = update_add(
        vs.sa_rz,
        at[2:-2, 2:-2, 1, :], vs.inf_mat_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] + vs.inf_pf_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] + vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :] - vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :] - vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :],
    )
    vs.sa_ss = update_add(
        vs.sa_ss,
        at[2:-2, 2:-2, 1, :], vs.inf_pf_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_inf_pf_ss[2:-2, 2:-2, :] + vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :] - vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :] - vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :],
    )
    vs.msa_rz = update_add(
        vs.msa_rz,
        at[2:-2, 2:-2, 1, :], npx.where(npx.isnan(vs.mtt_inf_mat_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_mat_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_inf_pf_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_cpr_rz[2:-2, 2:-2, :]), 0, vs.mtt_cpr_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_evap_soil[2:-2, 2:-2, :]), 0, vs.mtt_evap_soil[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_transp[2:-2, 2:-2, :]), 0, vs.mtt_transp[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_q_rz[2:-2, 2:-2, :]), 0, vs.mtt_q_rz[2:-2, 2:-2, :]),
    )
    vs.msa_ss = update_add(
        vs.msa_ss,
        at[2:-2, 2:-2, 1, :], npx.where(npx.isnan(vs.mtt_inf_pf_ss[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_ss[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_q_rz[2:-2, 2:-2, :]), 0, vs.mtt_q_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_cpr_rz[2:-2, 2:-2, :]), 0, vs.mtt_cpr_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_q_ss[2:-2, 2:-2, :]), 0, vs.mtt_q_ss[2:-2, 2:-2, :]),
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

    if vs.itt >= 19:
        print('')

    return KernelOutput(tt_evap_soil=vs.tt_evap_soil, tt_transp=vs.tt_transp, tt_q_rz=vs.tt_q_rz, tt_cpr_rz=vs.tt_cpr_rz, tt_q_ss=vs.tt_q_ss,
                        TT_evap_soil=vs.TT_evap_soil, TT_transp=vs.TT_transp, TT_q_rz=vs.TT_q_rz, TT_cpr_rz=vs.TT_cpr_rz, TT_q_ss=vs.TT_q_ss,
                        mtt_evap_soil=vs.mtt_evap_soil, mtt_transp=vs.mtt_transp, mtt_q_rz=vs.mtt_q_rz, mtt_cpr_rz=vs.mtt_cpr_rz, mtt_q_ss=vs.mtt_q_ss,
                        C_inf_mat_rz=vs.C_inf_mat_rz, C_inf_pf_rz=vs.C_inf_pf_rz, C_inf_pf_ss=vs.C_inf_pf_ss, C_evap_soil=vs.C_evap_soil, C_transp=vs.C_transp, C_q_rz=vs.C_q_rz, C_cpr_rz=vs.C_cpr_rz, C_q_ss=vs.C_q_ss,
                        sa_rz=vs.sa_rz, sa_ss=vs.sa_ss, msa_rz=vs.msa_rz, msa_ss=vs.msa_ss)


@roger_routine
def calculate_storage_selection(state):
    """Calculates transport model
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate):
        if settings.sas_solver == "Euler":
            vs.update(transport_model_euler(state))

        elif settings.sas_solver == "RK4":
            vs.update(transport_model_rk4(state))

        else:
            vs.update(transport_model_deterministic(state))

    elif settings.enable_offline_transport and (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate):
        if settings.sas_solver == "Euler":
            vs.update(solute_transport_model_euler(state))

        elif settings.sas_solver == "RK4":
            vs.update(solute_transport_model_rk4(state))

        else:
            vs.update(transport_model_deterministic(state))
