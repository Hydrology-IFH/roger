from roger import roger_kernel
from roger.variables import allocate
from roger.core.operators import numpy as npx, scipy_special as spsx, update, at


@roger_kernel
def uniform(state, SA, sas_params):
    """Uniform SAS function"""
    vs = state.variables

    mask = (sas_params[:, :, 0, npx.newaxis] == 1)

    S = allocate(state.dimensions, ("x", "y", 1))
    lam = allocate(state.dimensions, ("x", "y", 1))
    Omega = allocate(state.dimensions, ("x", "y", "nages"))
    S = update(
        S,
        at[:, :, :], npx.max(SA[:, :, vs.tau, :], axis=-1)[:, :, npx.newaxis] * mask * vs.maskCatch[:, :, npx.newaxis],
    )
    lam = update(
        lam,
        at[:, :, :], (1 / S) * mask * vs.maskCatch[:, :, npx.newaxis],
    )
    Omega = update(
        Omega,
        at[:, :, :], npx.where(SA[:, :, vs.tau, :] < S, npx.where(SA[:, :, vs.tau, :] > 0, lam * SA[:, :, vs.tau, :], 0.), 1.) * mask * vs.maskCatch[:, :, npx.newaxis],
    )
    Omega = update(
        Omega,
        at[:, :, -1], 1 * mask[:, :, 0] * vs.maskCatch,
    )

    return Omega


@roger_kernel
def kumaraswami(state, SA, sas_params):
    """Kumaraswami SAS function"""
    vs = state.variables

    mask3 = npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([3, 31, 32, 33, 34, 35]))
    mask31 = (sas_params[:, :, 0] == 31)
    mask32 = (sas_params[:, :, 0] == 32)
    mask33 = (sas_params[:, :, 0] == 33)
    mask34 = (sas_params[:, :, 0] == 34)
    mask35 = (sas_params[:, :, 0] == 35)

    S = allocate(state.dimensions, ("x", "y", 1))
    Omega = allocate(state.dimensions, ("x", "y", "nages"))
    S = update(
        S,
        at[:, :, :], npx.max(SA[:, :, vs.tau, :], axis=-1)[:, :, npx.newaxis] * vs.maskCatch[:, :, npx.newaxis],
    )

    # make parameters storage dependent
    S_rel = allocate(state.dimensions, ("x", "y"))
    S_rel = update(
        S_rel,
        at[:, :], (S[:, :, 0] - sas_params[:, :, 5]) / (sas_params[:, :, 6] - sas_params[:, :, 5]) * vs.maskCatch,
    )
    S_rel = update(
        S_rel,
        at[:, :], npx.where(S_rel < 0, 0, S_rel),
    )
    S_rel = update(
        S_rel,
        at[:, :], npx.where(S_rel > 1, 1, S_rel),
    )

    sas_params = update(
        sas_params,
        at[:, :, 1], npx.where(mask31, 1, sas_params[:, :, 1]),
    )
    sas_params = update(
        sas_params,
        at[:, :, 2], npx.where(mask31, sas_params[:, :, 3] + (S_rel * sas_params[:, :, 4]), sas_params[:, :, 2]),
    )
    sas_params = update(
        sas_params,
        at[:, :, 1], npx.where(mask32, sas_params[:, :, 3] + ((1 - S_rel) * sas_params[:, :, 4]), sas_params[:, :, 1]),
    )
    sas_params = update(
        sas_params,
        at[:, :, 2], npx.where(mask32, 1, sas_params[:, :, 2]),
    )
    sas_params = update(
        sas_params,
        at[:, :, 1], npx.where(mask33, 1, sas_params[:, :, 1]),
    )
    sas_params = update(
        sas_params,
        at[:, :, 2], npx.where(mask33, sas_params[:, :, 3] + ((1 - S_rel) * sas_params[:, :, 4]), sas_params[:, :, 2]),
    )
    sas_params = update(
        sas_params,
        at[:, :, 1], npx.where(mask34, sas_params[:, :, 3] + (S_rel * sas_params[:, :, 4]), sas_params[:, :, 1]),
    )
    sas_params = update(
        sas_params,
        at[:, :, 2], npx.where(mask34, 1, sas_params[:, :, 2]),
    )
    sas_params = update(
        sas_params,
        at[:, :, 1], npx.where(mask35, sas_params[:, :, 3] + ((1 - S_rel) * sas_params[:, :, 4]), sas_params[:, :, 1]),
    )
    sas_params = update(
        sas_params,
        at[:, :, 2], npx.where(mask35, sas_params[:, :, 3] + (S_rel * sas_params[:, :, 4]), sas_params[:, :, 2]),
    )

    Omega = update(
        Omega,
        at[:, :, :], npx.where(S >= 0, npx.where(SA[:, :, vs.tau, :] > 0, npx.where(SA[:, :, vs.tau, :] < S,
                               1 - (1 - (SA[:, :, vs.tau, :]/S)**sas_params[:, :, 1, npx.newaxis])**sas_params[:, :, 2, npx.newaxis], 1.), 0.),
                               npx.where(SA[:, :, vs.tau, :] > 0,
                               1 - (1 - (SA[:, :, vs.tau, :]/S)**sas_params[:, :, 1, npx.newaxis])**sas_params[:, :, 2, npx.newaxis], 0.)) * mask3 * vs.maskCatch[:, :, npx.newaxis],
    )

    return Omega


@roger_kernel
def gamma(state, SA, sas_params):
    """Gamma SAS function"""
    vs = state.variables

    mask = (sas_params[:, :, 0, npx.newaxis] == 4)

    S = allocate(state.dimensions, ("x", "y", 1))
    lam = allocate(state.dimensions, ("x", "y", 1))
    rescale = allocate(state.dimensions, ("x", "y", 1))
    Omega = allocate(state.dimensions, ("x", "y", "nages"))
    S = update(
        S,
        at[:, :, :], npx.max(SA[:, :, vs.tau, :], axis=-1)[:, :, npx.newaxis] * mask * vs.maskCatch[:, :, npx.newaxis],
    )
    lam = update(
        lam,
        at[:, :, :], (1 / sas_params[:, :, 1, npx.newaxis]) * mask * vs.maskCatch[:, :, npx.newaxis],
    )
    rescale = update(
        rescale,
        at[:, :, :], npx.where(npx.isfinite(S), 1/(spsx.gammainc(sas_params[:, :, 2, npx.newaxis], lam * S)), 1.) * mask * vs.maskCatch[:, :, npx.newaxis],
    )

    Omega = update(
        Omega,
        at[:, :, :], npx.where(SA[:, :, vs.tau, :] > 0, npx.where(SA[:, :, vs.tau, :] < S,
                               spsx.gammainc(sas_params[:, :, 2, npx.newaxis], lam * SA[:, :, vs.tau, :]) * rescale, 1.), 0.) * mask * vs.maskCatch[:, :, npx.newaxis],
    )

    return Omega


@roger_kernel
def exponential(state, SA, sas_params):
    """Exponential SAS function"""
    vs = state.variables

    mask51 = (sas_params[:, :, 0, npx.newaxis] == 51)
    mask52 = (sas_params[:, :, 0, npx.newaxis] == 52)

    S = allocate(state.dimensions, ("x", "y", 1))
    Omega = allocate(state.dimensions, ("x", "y", "nages"))
    S = update(
        S,
        at[:, :, :], npx.max(SA[:, :, vs.tau, :], axis=-1)[:, :, npx.newaxis] * vs.maskCatch[:, :, npx.newaxis],
    )
    Omega = update(
        Omega,
        at[:, :, :], npx.where(mask51, npx.where(SA[:, :, vs.tau, :] > 0, npx.where(SA[:, :, vs.tau, :] < S,
                               1 - npx.exp(sas_params[:, :, 1, npx.newaxis] * (-1) * SA[:, :, vs.tau, :]), 1.), 0.), Omega) * vs.maskCatch[:, :, npx.newaxis],
    )
    Omega = update(
        Omega,
        at[:, :, :], npx.where(mask52, npx.where(SA[:, :, vs.tau, :] > 0, npx.where(SA[:, :, vs.tau, :] < S,
                               1 - npx.exp(sas_params[:, :, 1, npx.newaxis] * (-1) * SA[:, :, vs.tau, :]), 1.), 0.)[..., ::-1], Omega) * vs.maskCatch[:, :, npx.newaxis],
    )

    return Omega


@roger_kernel
def power(state, SA, sas_params):
    """Power SAS function"""
    vs = state.variables

    mask61 = (sas_params[:, :, 0, npx.newaxis] == 61)
    mask62 = (sas_params[:, :, 0, npx.newaxis] == 62)

    S = allocate(state.dimensions, ("x", "y", 1))
    Omega = allocate(state.dimensions, ("x", "y", "nages"))
    S = update(
        S,
        at[:, :, :], npx.max(SA[:, :, vs.tau, :], axis=-1)[:, :, npx.newaxis] * vs.maskCatch[:, :, npx.newaxis],
    )
    Omega = update(
        Omega,
        at[:, :, :], npx.where(mask61, npx.where(SA[:, :, vs.tau, :] > 0, npx.where(SA[:, :, vs.tau, :] <= S * (1/sas_params[:, :, 1, npx.newaxis]),
                               sas_params[:, :, 1, npx.newaxis] * (SA[:, :, vs.tau, :] / S)**sas_params[:, :, 2, npx.newaxis], 1.), 0.), Omega) * vs.maskCatch[:, :, npx.newaxis],
    )
    Omega = update(
        Omega,
        at[:, :, :], npx.where(mask62, npx.where(SA[:, :, vs.tau, :] > 0, npx.where(SA[:, :, vs.tau, :] <= S * (1/sas_params[:, :, 1, npx.newaxis]),
                               sas_params[:, :, 1, npx.newaxis] * (SA[:, :, vs.tau, :] / S)**sas_params[:, :, 2, npx.newaxis], 1.), 0.)[..., ::-1], Omega) * vs.maskCatch[:, :, npx.newaxis],
    )

    return Omega


@roger_kernel
def dirac(state, sas_params):
    """Dirac SAS function"""
    vs = state.variables

    mask21 = (sas_params[:, :, 0] == 21)
    mask22 = (sas_params[:, :, 0] == 22)

    Omega = allocate(state.dimensions, ("x", "y", "nages"))
    Omega = update(
        Omega,
        at[:, :, 1], npx.where(mask21, 1, 0) * vs.maskCatch,
    )
    Omega = update(
        Omega,
        at[:, :, -1], npx.where(mask22, 1, 0) * vs.maskCatch,
    )

    return Omega
