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
        at[:, :], npx.sum(sa[:, :, vs.tau, :], axis=-1) * vs.maskCatch,
    )

    # air temperature coefficient
    ta_coeff = allocate(state.dimensions, ("x", "y"))
    ta_coeff = update(
        ta_coeff,
        at[:, :], npx.where(((vs.ta[:, :, vs.tau] >= 5) & (vs.ta[:, :, vs.tau] <= 50)), vs.ta[:, :, vs.tau] / (50 - 5), 0) * vs.maskCatch,
    )
    # calculate denitrification rate
    mr = allocate(state.dimensions, ("x", "y", "ages"))
    mr = update(
        mr,
        at[:, :, :], Dmax[:, :, npx.newaxis] * (msa[:, :, vs.tau, :] / (km[:, :, npx.newaxis] + msa[:, :, vs.tau, :])) * ta_coeff[:, :, npx.newaxis] * (vs.dt / (365 * 24)) * settings.dx * settings.dy * 100 * vs.maskCatch[:, :, npx.newaxis],
    )
    # no denitrification if storage is lower than 70 % of pore volume
    mr = update(
        mr,
        at[:, :, :], npx.where(S[:, :, npx.newaxis] >= 0.7 * (S_sat[:, :, npx.newaxis] - S_pwp[:, :, npx.newaxis]), mr, 0) * vs.maskCatch[:, :, npx.newaxis],
    )
    # limit denitrification to available solute mass
    mr = update(
        mr,
        at[:, :, :], npx.where(mr > msa[:, :, vs.tau, :], msa[:, :, vs.tau, :], mr) * vs.maskCatch[:, :, npx.newaxis],
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
        at[:, :], npx.sum(sa[:, :, vs.tau, :], axis=-1) * vs.maskCatch,
    )

    # air temperature coefficient
    ta_coeff = allocate(state.dimensions, ("x", "y"))
    ta_coeff = update(
        ta_coeff,
        at[:, :], npx.where(((vs.ta[:, :, vs.tau] >= 1) & (vs.ta[:, :, vs.tau] <= 30)), vs.ta[:, :, vs.tau] / (30 - 1), 0) * vs.maskCatch,
    )
    # calculate nitrification rate
    ma = allocate(state.dimensions, ("x", "y", "ages"))
    ma = update(
        ma,
        at[:, :, :], Dnit[:, :, npx.newaxis] * (Nmin[:, :, vs.tau, :] / (knit[:, :, npx.newaxis] + Nmin[:, :, vs.tau, :])) * ta_coeff[:, :, npx.newaxis] * (vs.dt / (365 * 24)) * settings.dx * settings.dy * 100 * vs.maskCatch[:, :, npx.newaxis],
    )
    # print(npx.sum(Nmin[:, :, vs.tau, :], axis=-1)[3,3])
    # no nitrification if storage is greater than 70 % of pore volume
    ma = update(
        ma,
        at[:, :, :], npx.where(S[:, :, npx.newaxis] < 0.7 * (S_sat[:, :, npx.newaxis] - S_pwp[:, :, npx.newaxis]), ma, 0) * vs.maskCatch[:, :, npx.newaxis],
    )
    # limit denitrification to available solute mass
    ma = update(
        ma,
        at[:, :, :], npx.where(ma > Nmin[:, :, vs.tau, :], Nmin[:, :, vs.tau, :], ma) * vs.maskCatch[:, :, npx.newaxis],
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
        at[:, :], npx.where(((vs.ta[:, :, vs.tau] >= 0) & (vs.ta[:, :, vs.tau] <= 50)), vs.ta[:, :, vs.tau] / (50 - 0), 0) * vs.maskCatch,
    )

    ma = allocate(state.dimensions, ("x", "y"))
    ma = update(
        ma,
        at[:, :], kmin * (vs.dt / (365 * 24)) * settings.dx * settings.dy * 100 * ta_coeff * vs.maskCatch,
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
        at[:, :, :], vs.ages[npx.newaxis, npx.newaxis, :] * vs.maskCatch[:, :, npx.newaxis],
    )
    mr = update(
        mr,
        at[:, :, :], msa * k[:, :, npx.newaxis] * npx.exp(-k[:, :, npx.newaxis] * age) * vs.maskCatch[:, :, npx.newaxis],
    )
    # limit denitrification to available solute mass
    mr = update(
        mr,
        at[:, :, :], npx.where(mr > msa[:, :, vs.tau, :], msa[:, :, vs.tau, :], mr) * vs.maskCatch[:, :, npx.newaxis],
    )

    return mr


@roger_kernel
def calculate_nitrogen_cycle_kernel(state):
    vs = state.variables

    vs.Nmin_rz = update_add(
        vs.Nmin_rz,
        at[:, :, vs.tau, 0], min_soil(state, vs.kmin_rz) * vs.maskCatch,
    )

    vs.Nmin_ss = update_add(
        vs.Nmin_ss,
        at[:, :, vs.tau, 0], min_soil(state, vs.kmin_ss) * vs.maskCatch,
    )

    vs.ma_rz = update(
        vs.ma_rz,
        at[:, :, :], nit_soil(state, vs.Nmin_rz, vs.km_nit_rz, vs.dmax_nit_rz, vs.sa_rz, vs.S_sat_rz, vs.S_pwp_rz) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.Nmin_rz = update_add(
        vs.Nmin_rz,
        at[:, :, vs.tau, :], - vs.ma_rz,
    )

    vs.msa_rz = update_add(
        vs.msa_rz,
        at[:, :, vs.tau, 0], npx.sum(vs.ma_rz, axis=-1),
    )

    vs.ma_ss = update(
        vs.ma_ss,
        at[:, :, :], nit_soil(state, vs.Nmin_ss, vs.km_nit_ss, vs.dmax_nit_ss, vs.sa_ss, vs.S_sat_ss, vs.S_pwp_ss) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.Nmin_ss = update_add(
        vs.Nmin_ss,
        at[:, :, vs.tau, :], - vs.ma_ss,
    )

    vs.msa_ss = update_add(
        vs.msa_ss,
        at[:, :, vs.tau, 0], npx.sum(vs.ma_ss, axis=-1),
    )

    vs.mr_rz = update(
        vs.mr_rz,
        at[:, :, :], denit_soil(state, vs.msa_rz, vs.km_denit_rz, vs.dmax_denit_rz, vs.sa_rz, vs.S_sat_rz, vs.S_pwp_rz) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.msa_rz = update_add(
        vs.msa_rz,
        at[:, :, vs.tau, :], - vs.mr_rz,
    )

    vs.mr_ss = update(
        vs.mr_ss,
        at[:, :, :], denit_soil(state, vs.msa_ss, vs.km_denit_ss, vs.dmax_denit_ss, vs.sa_ss, vs.S_sat_ss, vs.S_pwp_ss) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.msa_ss = update_add(
        vs.msa_ss,
        at[:, :, vs.tau, :], - vs.mr_ss,
    )

    vs.ma_s = update(
        vs.ma_s,
        at[:, :, :], vs.ma_rz + vs.ma_ss,
    )

    vs.mr_s = update(
        vs.mr_s,
        at[:, :, :], vs.mr_rz + vs.mr_ss,
    )

    vs.Nmin_s = update(
        vs.Nmin_s,
        at[:, :, vs.tau, :], vs.Nmin_rz[:, :, vs.tau, :] + vs.Nmin_ss[:, :, vs.tau, :] * vs.maskCatch[:, :, npx.newaxis],
    )

    return KernelOutput(msa_rz=vs.msa_rz, msa_ss=vs.msa_ss, ma_rz=vs.ma_rz, ma_ss=vs.ma_ss, ma_s=vs.ma_s, mr_rz=vs.mr_rz, mr_ss=vs.mr_ss, mr_s=vs.mr_s, Nmin_rz=vs.Nmin_rz, Nmin_ss=vs.Nmin_ss, Nmin_s=vs.Nmin_s)


@roger_kernel
def calculate_nitrogen_cycle_gw_kernel(state):
    vs = state.variables

    vs.mr_gw = update(
        vs.mr_gw,
        at[:, :, :], denit_gw(state, vs.msa_gw, vs.k_denit_gw) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.msa_gw = update_add(
        vs.msa_gw,
        at[:, :, vs.tau, :], - vs.mr_gw,
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
