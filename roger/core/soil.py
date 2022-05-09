from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, at
from roger.core import transport


@roger_kernel
def calc_k(state):
    """
    Calculates hydraulic conductivity of soil
    """
    vs = state.variables

    vs.k = update(
        vs.k,
        at[2:-2, 2:-2, vs.tau], (vs.ks[2:-2, 2:-2]/(1 + (vs.theta[2:-2, 2:-2, vs.tau]/vs.theta_sat[2:-2, 2:-2])**(-vs.m_bc[2:-2, 2:-2]))) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(k=vs.k)


@roger_kernel
def calc_h(state):
    """
    Calculates soil water potential
    """
    vs = state.variables

    vs.h = update(
        vs.h,
        at[2:-2, 2:-2, vs.tau], (vs.ha[2:-2, 2:-2]/((vs.theta[2:-2, 2:-2, vs.tau]/vs.theta_sat[2:-2, 2:-2])**(1/vs.lambda_bc[2:-2, 2:-2]))) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(h=vs.h)


@roger_kernel
def calc_theta(state):
    """
    Calculates soil water content
    """
    vs = state.variables

    vs.theta = update(
        vs.theta,
        at[2:-2, 2:-2, vs.tau], ((vs.S_fp_s[2:-2, 2:-2] + vs.S_lp_s[2:-2, 2:-2])/vs.z_soil[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(theta=vs.theta)


@roger_kernel
def calc_theta_ff(state):
    """
    Calculates soil water conent including film flow
    """
    vs = state.variables

    vs.theta_ff = update(
        vs.theta_ff,
        at[2:-2, 2:-2, vs.tau], npx.sum(vs.S_f[2:-2, 2:-2, :], axis=-1) / vs.z_soil[2:-2, 2:-2] + vs.theta[2:-2, 2:-2, vs.tau],
    )

    return KernelOutput(
        theta_ff=vs.theta_ff,
        )


@roger_kernel
def calc_S(state):
    """
    Calculates soil water content
    """
    vs = state.variables

    vs.S_fp_s = update(
        vs.S_fp_s,
        at[2:-2, 2:-2], (vs.S_fp_rz[2:-2, 2:-2] + vs.S_fp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_lp_s = update(
        vs.S_lp_s,
        at[2:-2, 2:-2], (vs.S_lp_rz[2:-2, 2:-2] + vs.S_lp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_s = update(
        vs.S_s,
        at[2:-2, 2:-2, vs.tau], (vs.S_pwp_s[2:-2, 2:-2] + vs.S_fp_s[2:-2, 2:-2] + vs.S_lp_s[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(S_fp_s=vs.S_fp_s, S_lp_s=vs.S_lp_s, S_s=vs.S_s)


@roger_kernel
def calc_dS(state):
    """
    Calculates storage change
    """
    vs = state.variables

    vs.dS_s = update(
        vs.dS_s,
        at[2:-2, 2:-2], (vs.S_s[2:-2, 2:-2, vs.tau] - vs.S_s[2:-2, 2:-2, vs.taum1]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(dS_s=vs.dS_s)


@roger_routine
def calculate_soil(state):
    """
    Calculates soil storage and storage dependent variables
    """
    vs = state.variables
    settings = state.settings

    vs.update(calc_S(state))
    vs.update(calc_dS(state))
    vs.update(calc_theta(state))
    vs.update(calc_k(state))
    vs.update(calc_h(state))

    if settings.enable_film_flow:
        vs.update(calc_theta_ff(state))


@roger_kernel
def calculate_soil_transport_kernel(state):
    """
    Calculates StorAge
    """
    vs = state.variables

    vs.sa_s = update(
        vs.sa_s,
        at[2:-2, 2:-2, vs.tau, :], vs.sa_rz[2:-2, 2:-2, vs.tau, :] + vs.sa_ss[2:-2, 2:-2, vs.tau, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.SA_s = update(
        vs.SA_s,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_s, vs.sa_s)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    return KernelOutput(sa_s=vs.sa_s, SA_s=vs.SA_s)


@roger_kernel
def calculate_soil_transport_iso_kernel(state):
    """
    Calculates StorAge and isotope ratio
    """
    vs = state.variables

    vs.sa_s = update(
        vs.sa_s,
        at[2:-2, 2:-2, :, :], vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.SA_s = update(
        vs.SA_s,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_s, vs.sa_s)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    iso_rz = allocate(state.dimensions, ("x", "y", "ages"))
    iso_ss = allocate(state.dimensions, ("x", "y", "ages"))
    iso_rz = update(
        iso_rz,
        at[2:-2, 2:-2, :], npx.where(npx.isnan(vs.msa_rz[2:-2, 2:-2, vs.tau, :]), 0, vs.msa_rz[2:-2, 2:-2, vs.tau, :]),
    )
    iso_ss = update(
        iso_ss,
        at[2:-2, 2:-2, :], npx.where(npx.isnan(vs.msa_ss[2:-2, 2:-2, vs.tau, :]), 0, vs.msa_ss[2:-2, 2:-2, vs.tau, :]),
    )
    vs.msa_s = update(
        vs.msa_s,
        at[2:-2, 2:-2, vs.tau, :], (vs.sa_rz[2:-2, 2:-2, vs.tau, :] / vs.sa_s[2:-2, 2:-2, vs.tau, :]) * iso_rz[2:-2, 2:-2, :] + (vs.sa_ss[2:-2, 2:-2, vs.tau, :] / vs.sa_s[2:-2, 2:-2, vs.tau, :]) * iso_ss,
    )
    vs.msa_s = update(
        vs.msa_s,
        at[2:-2, 2:-2, vs.tau, :], npx.where(vs.msa_s[2:-2, 2:-2, vs.tau, :] == 0, npx.NaN, vs.msa_s[2:-2, 2:-2, vs.tau, :]),
    )

    vs.C_s = update(
        vs.C_s,
        at[2:-2, 2:-2, vs.tau], transport.calc_conc_iso_storage(state, vs.sa_s, vs.msa_s)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(sa_s=vs.sa_s, SA_s=vs.SA_s, msa_s=vs.msa_s, C_s=vs.C_s)


@roger_kernel
def calculate_soil_transport_anion_kernel(state):
    """
    Calculates StorAge and solute concentration
    """
    vs = state.variables

    vs.sa_s = update(
        vs.sa_s,
        at[2:-2, 2:-2, :, :], vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.SA_s = update(
        vs.SA_s,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_s, vs.sa_s)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.msa_s = update(
        vs.msa_s,
        at[2:-2, 2:-2, :, :], vs.msa_rz[2:-2, 2:-2, :, :] + vs.msa_ss[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.MSA_s = update(
        vs.MSA_s,
        at[2:-2, 2:-2, :, :], transport.calc_MSA(state, vs.MSA_s, vs.msa_s)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.M_s = update(
        vs.M_s,
        at[2:-2, 2:-2, vs.tau], npx.sum(vs.msa_s[2:-2, 2:-2, vs.tau, :], axis=-1)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.C_s = update(
        vs.C_s,
        at[2:-2, 2:-2, vs.tau], vs.M_s[2:-2, 2:-2, vs.tau] / npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1),
    )

    return KernelOutput(sa_s=vs.sa_s, SA_s=vs.SA_s, msa_s=vs.msa_s, MSA_s=vs.MSA_s, C_s=vs.C_s, M_s=vs.M_s)


@roger_routine
def calculate_soil_transport(state):
    """
    Calculates StorAge (and solute concentration)
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate):
        vs.update(calculate_soil_transport_kernel(state))

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(calculate_soil_transport_iso_kernel(state))

    if settings.enable_offline_transport and (settings.enable_chloride | settings.enable_bromide | settings.enable_nitrate):
        vs.update(calculate_soil_transport_anion_kernel(state))
