from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at


@roger_kernel
def denit_soil(state, msa, km, Dmax, sa, S_sat, S_pwp):
    """Calculates soil dentirification rate.
    """
    vs = state.variables
    settings = state.settings

    S = allocate(state.dimensions, ("x", "y"))
    S = update(
        S,
        at[2:-2, 2:-2], npx.sum(sa[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
    )

    # air temperature coefficient
    ta_coeff = allocate(state.dimensions, ("x", "y"))
    ta_coeff = update(
        ta_coeff,
        at[2:-2, 2:-2], npx.where(((vs.ta[2:-2, 2:-2, vs.tau] >= 5) & (vs.ta[2:-2, 2:-2, vs.tau] <= 50)), vs.ta[2:-2, 2:-2, vs.tau] / (50 - 5), 0) * vs.maskCatch[2:-2, 2:-2],
    )
    # calculate denitrification rate
    mr = allocate(state.dimensions, ("x", "y", "ages"))
    mr = update(
        mr,
        at[2:-2, 2:-2, :], Dmax[2:-2, 2:-2, npx.newaxis] * (msa[2:-2, 2:-2, vs.tau, :] / (km[2:-2, 2:-2, npx.newaxis] + msa[2:-2, 2:-2, vs.tau, :])) * ta_coeff[2:-2, 2:-2, npx.newaxis] * (vs.dt / (365 * 24)) * settings.dx * settings.dy * 100 * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    # no denitrification if storage is lower than 70 % of pore volume
    mr = update(
        mr,
        at[2:-2, 2:-2, :], npx.where(S[2:-2, 2:-2, npx.newaxis] >= 0.7 * (S_sat[2:-2, 2:-2, npx.newaxis] - S_pwp[2:-2, 2:-2, npx.newaxis]), mr, 0) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    # limit denitrification to available solute mass
    mr = update(
        mr,
        at[2:-2, 2:-2, :], npx.where(mr > msa[2:-2, 2:-2, vs.tau, :], msa[2:-2, 2:-2, vs.tau, :], mr) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return mr


@roger_kernel
def nit_soil(state, Nmin, knit, Dnit, sa, S_sat, S_pwp):
    """Calculates soil nitrification rate.
    """
    vs = state.variables
    settings = state.settings

    S = allocate(state.dimensions, ("x", "y"))
    S = update(
        S,
        at[2:-2, 2:-2], npx.sum(sa[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
    )

    # air temperature coefficient
    ta_coeff = allocate(state.dimensions, ("x", "y"))
    ta_coeff = update(
        ta_coeff,
        at[2:-2, 2:-2], npx.where(((vs.ta[2:-2, 2:-2, vs.tau] >= 1) & (vs.ta[2:-2, 2:-2, vs.tau] <= 30)), vs.ta[2:-2, 2:-2, vs.tau] / (30 - 1), 0) * vs.maskCatch[2:-2, 2:-2],
    )
    # calculate nitrification rate
    ma = allocate(state.dimensions, ("x", "y", "ages"))
    ma = update(
        ma,
        at[2:-2, 2:-2, :], Dnit[2:-2, 2:-2, npx.newaxis] * (Nmin[2:-2, 2:-2, vs.tau, :] / (knit[2:-2, 2:-2, npx.newaxis] + Nmin[2:-2, 2:-2, vs.tau, :])) * ta_coeff[2:-2, 2:-2, npx.newaxis] * (vs.dt / (365 * 24)) * settings.dx * settings.dy * 100 * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    # print(npx.sum(Nmin[2:-2, 2:-2, vs.tau, :], axis=-1)[3,3])
    # no nitrification if storage is greater than 70 % of pore volume
    ma = update(
        ma,
        at[2:-2, 2:-2, :], npx.where(S[2:-2, 2:-2, npx.newaxis] < 0.7 * (S_sat[2:-2, 2:-2, npx.newaxis] - S_pwp[2:-2, 2:-2, npx.newaxis]), ma, 0) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    # limit denitrification to available solute mass
    ma = update(
        ma,
        at[2:-2, 2:-2, :], npx.where(ma[2:-2, 2:-2, :] > Nmin[2:-2, 2:-2, vs.tau, :], Nmin[2:-2, 2:-2, vs.tau, :], ma[2:-2, 2:-2, :]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return ma


@roger_kernel
def min_soil(state, kmin):
    """Calculates soil nitrogen mineralization rate.
    """
    vs = state.variables
    settings = state.settings

    # air temperature coefficient
    ta_coeff = allocate(state.dimensions, ("x", "y"))
    ta_coeff = update(
        ta_coeff,
        at[2:-2, 2:-2], npx.where(((vs.ta[2:-2, 2:-2, vs.tau] >= 0) & (vs.ta[2:-2, 2:-2, vs.tau] <= 50)), vs.ta[2:-2, 2:-2, vs.tau] / (50 - 0), 0) * vs.maskCatch[2:-2, 2:-2],
    )

    ma = allocate(state.dimensions, ("x", "y"))
    ma = update(
        ma,
        at[2:-2, 2:-2], kmin[2:-2, 2:-2] * (vs.dt / (365 * 24)) * settings.dx * settings.dy * 100 * ta_coeff[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return ma


@roger_kernel
def denit_gw(state, msa, k):
    """Calculates groundwater dentrification rate.
    """
    vs = state.variables

    # calculate denitrification rate
    age = allocate(state.dimensions, ("x", "y", "ages"))
    mr = allocate(state.dimensions, ("x", "y", "ages"))
    age = update(
        age,
        at[2:-2, 2:-2, :], vs.ages[npx.newaxis, npx.newaxis, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    mr = update(
        mr,
        at[2:-2, 2:-2, :], msa[2:-2, 2:-2, :] * k[2:-2, 2:-2, npx.newaxis] * npx.exp(-k[2:-2, 2:-2, npx.newaxis] * age[2:-2, 2:-2, :]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    # limit denitrification to available solute mass
    mr = update(
        mr,
        at[2:-2, 2:-2, :], npx.where(mr[2:-2, 2:-2, :] > msa[2:-2, 2:-2, vs.tau, :], msa[2:-2, 2:-2, vs.tau, :], mr[2:-2, 2:-2, :]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return mr


@roger_kernel
def calculate_nitrogen_cycle_kernel(state):
    vs = state.variables

    vs.Nmin_rz = update_add(
        vs.Nmin_rz,
        at[2:-2, 2:-2, vs.tau, 0], min_soil(state, vs.kmin_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.Nmin_ss = update_add(
        vs.Nmin_ss,
        at[2:-2, 2:-2, vs.tau, 0], min_soil(state, vs.kmin_ss)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.ma_rz = update(
        vs.ma_rz,
        at[2:-2, 2:-2, :], nit_soil(state, vs.Nmin_rz, vs.km_nit_rz, vs.dmax_nit_rz, vs.sa_rz, vs.S_sat_rz, vs.S_pwp_rz)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.Nmin_rz = update_add(
        vs.Nmin_rz,
        at[2:-2, 2:-2, vs.tau, :], - vs.ma_rz[2:-2, 2:-2, :],
    )

    vs.msa_rz = update_add(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, 0], npx.sum(vs.ma_rz[2:-2, 2:-2, :], axis=-1),
    )

    vs.ma_ss = update(
        vs.ma_ss,
        at[2:-2, 2:-2, :], nit_soil(state, vs.Nmin_ss, vs.km_nit_ss, vs.dmax_nit_ss, vs.sa_ss, vs.S_sat_ss, vs.S_pwp_ss)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.Nmin_ss = update_add(
        vs.Nmin_ss,
        at[2:-2, 2:-2, vs.tau, :], - vs.ma_ss[2:-2, 2:-2, :],
    )

    vs.msa_ss = update_add(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.tau, 0], npx.sum(vs.ma_ss[2:-2, 2:-2, :], axis=-1),
    )

    vs.mr_rz = update(
        vs.mr_rz,
        at[2:-2, 2:-2, :], denit_soil(state, vs.msa_rz, vs.km_denit_rz, vs.dmax_denit_rz, vs.sa_rz, vs.S_sat_rz, vs.S_pwp_rz)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.msa_rz = update_add(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, :], - vs.mr_rz[2:-2, 2:-2, :],
    )

    vs.mr_ss = update(
        vs.mr_ss,
        at[2:-2, 2:-2, :], denit_soil(state, vs.msa_ss, vs.km_denit_ss, vs.dmax_denit_ss, vs.sa_ss, vs.S_sat_ss, vs.S_pwp_ss)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.msa_ss = update_add(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.tau, :], - vs.mr_ss[2:-2, 2:-2, :],
    )

    vs.ma_s = update(
        vs.ma_s,
        at[2:-2, 2:-2, :], vs.ma_rz[2:-2, 2:-2, :] + vs.ma_ss[2:-2, 2:-2, :],
    )

    vs.mr_s = update(
        vs.mr_s,
        at[2:-2, 2:-2, :], vs.mr_rz[2:-2, 2:-2, :] + vs.mr_ss[2:-2, 2:-2, :],
    )

    vs.Nmin_s = update(
        vs.Nmin_s,
        at[2:-2, 2:-2, vs.tau, :], vs.Nmin_rz[2:-2, 2:-2, vs.tau, :] + vs.Nmin_ss[2:-2, 2:-2, vs.tau, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(msa_rz=vs.msa_rz, msa_ss=vs.msa_ss, ma_rz=vs.ma_rz, ma_ss=vs.ma_ss, ma_s=vs.ma_s, mr_rz=vs.mr_rz, mr_ss=vs.mr_ss, mr_s=vs.mr_s, Nmin_rz=vs.Nmin_rz, Nmin_ss=vs.Nmin_ss, Nmin_s=vs.Nmin_s)


@roger_kernel
def calculate_nitrogen_cycle_gw_kernel(state):
    vs = state.variables

    vs.mr_gw = update(
        vs.mr_gw,
        at[2:-2, 2:-2, :], denit_gw(state, vs.msa_gw, vs.k_denit_gw)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.msa_gw = update_add(
        vs.msa_gw,
        at[2:-2, 2:-2, vs.tau, :], - vs.mr_gw[2:-2, 2:-2, :],
    )

    return KernelOutput(msa_rz=vs.msa_rz, msa_ss=vs.msa_ss)


@roger_routine
def calculate_nitrogen_cycle(state):
    """
    Calculates nitrogen cycle
    """
    vs = state.variables
    settings = state.settings

    vs.update(calculate_nitrogen_cycle_kernel(state))
    if settings.enable_groundwater:
        vs.update(calculate_nitrogen_cycle_gw_kernel(state))
