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
        at[2:-2, 2:-2, :], npx.max(SA[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis] * mask[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    lam = update(
        lam,
        at[2:-2, 2:-2, :], 1 / S[2:-2, 2:-2, :] * mask[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    Omega = update(
        Omega,
        at[2:-2, 2:-2, :], npx.where(SA[2:-2, 2:-2, vs.tau, :] < S[2:-2, 2:-2, :], npx.where(SA[2:-2, 2:-2, vs.tau, :] > 0, lam[2:-2, 2:-2, :] * SA[2:-2, 2:-2, vs.tau, :], 0.), 1.) * mask[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    Omega = update(
        Omega,
        at[2:-2, 2:-2, -1], npx.where(mask[2:-2, 2:-2, 0], 1, Omega[2:-2, 2:-2, -1]) * vs.maskCatch[2:-2, 2:-2],
    )
    Omega = update(
        Omega,
        at[2:-2, 2:-2, :], npx.where(S[2:-2, 2:-2, :] <= 0, 0, Omega[2:-2, 2:-2, :]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
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
        at[2:-2, 2:-2, :], npx.max(SA[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    # make parameters storage dependent
    S_rel = allocate(state.dimensions, ("x", "y"))
    S_rel = update(
        S_rel,
        at[2:-2, 2:-2], (S[2:-2, 2:-2, 0] - sas_params[2:-2, 2:-2, 5]) / (sas_params[2:-2, 2:-2, 6] - sas_params[2:-2, 2:-2, 5]) * vs.maskCatch[2:-2, 2:-2],
    )
    S_rel = update(
        S_rel,
        at[2:-2, 2:-2], npx.where(S_rel[2:-2, 2:-2] < 0, 0, S_rel[2:-2, 2:-2]),
    )
    S_rel = update(
        S_rel,
        at[2:-2, 2:-2], npx.where(S_rel[2:-2, 2:-2] > 1, 1, S_rel[2:-2, 2:-2]),
    )

    sas_params = update(
        sas_params,
        at[2:-2, 2:-2, 1], npx.where(mask31[2:-2, 2:-2], 1, sas_params[2:-2, 2:-2, 1]),
    )
    sas_params = update(
        sas_params,
        at[2:-2, 2:-2, 2], npx.where(mask31[2:-2, 2:-2], sas_params[2:-2, 2:-2, 3] + (S_rel[2:-2, 2:-2] * sas_params[2:-2, 2:-2, 4]), sas_params[2:-2, 2:-2, 2]),
    )
    sas_params = update(
        sas_params,
        at[2:-2, 2:-2, 1], npx.where(mask32[2:-2, 2:-2], sas_params[2:-2, 2:-2, 3] + ((1 - S_rel[2:-2, 2:-2]) * sas_params[2:-2, 2:-2, 4]), sas_params[2:-2, 2:-2, 1]),
    )
    sas_params = update(
        sas_params,
        at[2:-2, 2:-2, 2], npx.where(mask32[2:-2, 2:-2], 1, sas_params[2:-2, 2:-2, 2]),
    )
    sas_params = update(
        sas_params,
        at[2:-2, 2:-2, 1], npx.where(mask33[2:-2, 2:-2], 1, sas_params[2:-2, 2:-2, 1]),
    )
    sas_params = update(
        sas_params,
        at[2:-2, 2:-2, 2], npx.where(mask33[2:-2, 2:-2], sas_params[2:-2, 2:-2, 3] + ((1 - S_rel[2:-2, 2:-2]) * sas_params[2:-2, 2:-2, 4]), sas_params[2:-2, 2:-2, 2]),
    )
    sas_params = update(
        sas_params,
        at[2:-2, 2:-2, 1], npx.where(mask34[2:-2, 2:-2], sas_params[2:-2, 2:-2, 3] + (S_rel[2:-2, 2:-2] * sas_params[2:-2, 2:-2, 4]), sas_params[2:-2, 2:-2, 1]),
    )
    sas_params = update(
        sas_params,
        at[2:-2, 2:-2, 2], npx.where(mask34[2:-2, 2:-2], 1, sas_params[2:-2, 2:-2, 2]),
    )
    sas_params = update(
        sas_params,
        at[2:-2, 2:-2, 1], npx.where(mask35[2:-2, 2:-2], sas_params[2:-2, 2:-2, 3] + ((1 - S_rel[2:-2, 2:-2]) * sas_params[2:-2, 2:-2, 4]), sas_params[2:-2, 2:-2, 1]),
    )
    sas_params = update(
        sas_params,
        at[2:-2, 2:-2, 2], npx.where(mask35[2:-2, 2:-2], sas_params[2:-2, 2:-2, 3] + (S_rel[2:-2, 2:-2] * sas_params[2:-2, 2:-2, 4]), sas_params[2:-2, 2:-2, 2]),
    )

    Omega = update(
        Omega,
        at[2:-2, 2:-2, :], npx.where(S[2:-2, 2:-2, :] >= 0, npx.where(SA[2:-2, 2:-2, vs.tau, :] > 0, npx.where(SA[2:-2, 2:-2, vs.tau, :] < S[2:-2, 2:-2, :],
                               1 - (1 - (SA[2:-2, 2:-2, vs.tau, :]/S[2:-2, 2:-2, :])**sas_params[2:-2, 2:-2, 1, npx.newaxis])**sas_params[2:-2, 2:-2, 2, npx.newaxis], 1.), 0.),
                               npx.where(SA[2:-2, 2:-2, vs.tau, :] > 0,
                               1 - (1 - (SA[2:-2, 2:-2, vs.tau, :]/S[2:-2, 2:-2, :])**sas_params[2:-2, 2:-2, 1, npx.newaxis])**sas_params[2:-2, 2:-2, 2, npx.newaxis], 0.)) * mask3[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    Omega = update(
        Omega,
        at[2:-2, 2:-2, :], npx.where(S[2:-2, 2:-2, :] <= 0, 0, Omega[2:-2, 2:-2, :]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return Omega, sas_params


@roger_kernel
def gamma(state, SA, sas_params):
    """Gamma SAS function"""
    vs = state.variables

    mask = (sas_params[:, :, 0, npx.newaxis] == 4)

    S = allocate(state.dimensions, ("x", "y", 1))
    Omega = allocate(state.dimensions, ("x", "y", "nages"))
    S = update(
        S,
        at[2:-2, 2:-2, :], npx.max(SA[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis] * mask[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    Omega = update(
        Omega,
        at[2:-2, 2:-2, :], npx.where(SA[2:-2, 2:-2, vs.tau, :] > 0, npx.where(SA[2:-2, 2:-2, vs.tau, :] < S[2:-2, 2:-2, :],
                                     spsx.gammainc(sas_params[2:-2, 2:-2, 1, npx.newaxis], sas_params[2:-2, 2:-2, 2, npx.newaxis] * SA[2:-2, 2:-2, vs.tau, :]/S[2:-2, 2:-2, :]) / npx.exp(spsx.gammaln(sas_params[2:-2, 2:-2, 1, npx.newaxis])), 0.), 0) * mask[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    Omega = update(
        Omega,
        at[2:-2, 2:-2, :], npx.where(S[2:-2, 2:-2, :] <= 0, 0, Omega[2:-2, 2:-2, :]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
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
        at[2:-2, 2:-2, :], npx.max(SA[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    Omega = update(
        Omega,
        at[2:-2, 2:-2, :], npx.where(mask51[2:-2, 2:-2, :], npx.where(SA[2:-2, 2:-2, vs.tau, :] > 0, npx.where(SA[2:-2, 2:-2, vs.tau, :] < S[2:-2, 2:-2, :],
                               1 - npx.exp(sas_params[2:-2, 2:-2, 1, npx.newaxis] * (-1) * (SA[2:-2, 2:-2, vs.tau, :] / S[2:-2, 2:-2, :])), 1.), 0.), Omega[2:-2, 2:-2, :]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    Omega = update(
        Omega,
        at[2:-2, 2:-2, :], npx.where(mask52[2:-2, 2:-2, :], npx.where(SA[2:-2, 2:-2, vs.tau, :] > 0, npx.where(SA[2:-2, 2:-2, vs.tau, :] < S[2:-2, 2:-2, :],
                               1 - npx.exp(sas_params[2:-2, 2:-2, 1, npx.newaxis] * (-1) * (SA[2:-2, 2:-2, vs.tau, :] / S[2:-2, 2:-2, :])), 1.), 0.)[..., ::-1], Omega[2:-2, 2:-2, :]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    Omega = update(
        Omega,
        at[2:-2, 2:-2, :], npx.where(S[2:-2, 2:-2, :] <= 0, 0, Omega[2:-2, 2:-2, :]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return Omega


@roger_kernel
def power(state, SA, sas_params):
    """Power SAS function"""
    vs = state.variables

    mask6 = npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([6, 61, 62]))
    mask61 = (sas_params[:, :, 0] == 61)
    mask62 = (sas_params[:, :, 0] == 62)

    S = allocate(state.dimensions, ("x", "y", 1))
    Omega = allocate(state.dimensions, ("x", "y", "nages"))
    S = update(
        S,
        at[2:-2, 2:-2, :], npx.max(SA[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    # make parameters storage dependent
    S_rel = allocate(state.dimensions, ("x", "y"))
    S_rel = update(
        S_rel,
        at[2:-2, 2:-2], (S[2:-2, 2:-2, 0] - sas_params[2:-2, 2:-2, 5]) / (sas_params[2:-2, 2:-2, 6] - sas_params[2:-2, 2:-2, 5]) * vs.maskCatch[2:-2, 2:-2],
    )
    S_rel = update(
        S_rel,
        at[2:-2, 2:-2], npx.where(S_rel[2:-2, 2:-2] < 0, 0, S_rel[2:-2, 2:-2]),
    )
    S_rel = update(
        S_rel,
        at[2:-2, 2:-2], npx.where(S_rel[2:-2, 2:-2] > 1, 1, S_rel[2:-2, 2:-2]),
    )
    sas_params = update(
        sas_params,
        at[2:-2, 2:-2, 1], npx.where(mask61[2:-2, 2:-2], sas_params[2:-2, 2:-2, 3] + ((1 - S_rel[2:-2, 2:-2]) * sas_params[2:-2, 2:-2, 4]), sas_params[2:-2, 2:-2, 1]),
    )
    sas_params = update(
        sas_params,
        at[2:-2, 2:-2, 1], npx.where(mask62[2:-2, 2:-2], sas_params[2:-2, 2:-2, 3] + (S_rel[2:-2, 2:-2] * sas_params[2:-2, 2:-2, 4]), sas_params[2:-2, 2:-2, 1]),
    )
    Omega = update(
        Omega,
        at[2:-2, 2:-2, :], npx.where(SA[2:-2, 2:-2, vs.tau, :] > 0, npx.where(SA[2:-2, 2:-2, vs.tau, :] <= S[2:-2, 2:-2, :],
                              (SA[2:-2, 2:-2, vs.tau, :] / S[2:-2, 2:-2, :])**sas_params[2:-2, 2:-2, 1, npx.newaxis], 1.), 0.) * mask6[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    Omega = update(
        Omega,
        at[2:-2, 2:-2, :], npx.where(S[2:-2, 2:-2, :] <= 0, 0, Omega[2:-2, 2:-2, :]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return Omega, sas_params


@roger_kernel
def dirac(state, SA, sas_params):
    """Dirac SAS function"""
    vs = state.variables

    mask2 = npx.isin(sas_params[:, :, 0, npx.newaxis], npx.array([2]))

    S = allocate(state.dimensions, ("x", "y", 1))
    Omega = allocate(state.dimensions, ("x", "y", "nages"))
    S = update(
        S,
        at[2:-2, 2:-2, :], npx.max(SA[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    Omega = update(
        Omega,
        at[2:-2, 2:-2, :], npx.where(mask2[2:-2, 2:-2, :] & (vs.nages[npx.newaxis, npx.newaxis, :] <= sas_params[2:-2, 2:-2, 1, npx.newaxis]), 0, 1) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    Omega = update(
        Omega,
        at[2:-2, 2:-2, :], npx.where(S[2:-2, 2:-2, :] <= 0, 0, Omega[2:-2, 2:-2, :]) * mask2[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return Omega
