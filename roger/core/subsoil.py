from roger import roger_kernel, roger_routine, KernelOutput
from roger.core.operators import numpy as npx, update, at
from roger.core import transport


@roger_kernel
def calc_k(state):
    """
    Calculates hydraulic conductivity of subsoil
    """
    vs = state.variables

    vs.k_ss = update(
        vs.k_ss,
        at[2:-2, 2:-2, vs.tau], (vs.ks[2:-2, 2:-2]/(1 + (vs.theta_ss[2:-2, 2:-2, vs.tau]/vs.theta_sat[2:-2, 2:-2])**(-vs.m_bc[2:-2, 2:-2]))) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(k_ss=vs.k_ss)


@roger_kernel
def calc_h(state):
    """
    Calculates soil water potential of subsoil
    """
    vs = state.variables

    vs.h_ss = update(
        vs.h_ss,
        at[2:-2, 2:-2, vs.tau], (vs.ha[2:-2, 2:-2]/((vs.theta_ss[2:-2, 2:-2, vs.tau]/vs.theta_sat[2:-2, 2:-2])**(1/vs.lambda_bc[2:-2, 2:-2]))) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(h_ss=vs.h_ss)


@roger_kernel
def calc_theta(state):
    """
    Calculates soil water content of subsoil
    """
    vs = state.variables

    vs.theta_ss = update(
        vs.theta_ss,
        at[2:-2, 2:-2, vs.tau], ((vs.S_fp_ss[2:-2, 2:-2] + vs.S_lp_ss[2:-2, 2:-2])/(vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau]) + vs.theta_pwp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(theta_ss=vs.theta_ss)


@roger_kernel
def calc_theta_ff(state):
    """
    Calculates soil water conent including film flow
    """
    vs = state.variables

    vs.theta_ss_ff = update(
        vs.theta_ss_ff,
        at[2:-2, 2:-2, vs.tau], npx.sum(vs.S_f_ss[2:-2, 2:-2, :], axis=-1) / (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau]) + vs.theta_ss[2:-2, 2:-2, vs.tau],
    )

    return KernelOutput(
        theta_ss_ff=vs.theta_ss_ff,
        )


@roger_kernel
def calc_S(state):
    """
    Calculates soil water content of subsoil
    """
    vs = state.variables

    vs.S_ss = update(
        vs.S_ss,
        at[2:-2, 2:-2, vs.tau], (vs.S_pwp_ss[2:-2, 2:-2] + vs.S_fp_ss[2:-2, 2:-2] + vs.S_lp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(S_ss=vs.S_ss)


@roger_kernel
def calc_dS(state):
    """
    Calculates storage change of subsoil
    """
    vs = state.variables

    vs.dS_ss = update(
        vs.dS_ss,
        at[2:-2, 2:-2], (vs.S_ss[2:-2, 2:-2, vs.tau] - vs.S_ss[2:-2, 2:-2, vs.taum1]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(dS_ss=vs.dS_ss)


@roger_routine
def calculate_subsoil(state):
    """
    Calculates subsoil storage and storage dependent variables
    """
    vs = state.variables
    settings = state.settings

    vs.update(calc_theta(state))
    vs.update(calc_k(state))
    vs.update(calc_h(state))
    vs.update(calc_S(state))
    vs.update(calc_dS(state))

    if settings.enable_film_flow:
        vs.update(calc_theta_ff(state))


@roger_kernel
def calculate_subsoil_transport_kernel(state):
    """
    Calculates StorAge
    """
    vs = state.variables

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    return KernelOutput(SA_ss=vs.SA_ss)


@roger_kernel
def calculate_subsoil_transport_iso_kernel(state):
    """
    Calculates StorAge and isotope ratio
    """
    vs = state.variables

    vs.C_ss = update(
        vs.C_ss,
        at[2:-2, 2:-2, vs.tau], transport.calc_conc_iso_storage(state, vs.sa_ss, vs.msa_ss)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.M_ss = update(
        vs.M_ss,
        at[2:-2, 2:-2, vs.tau], npx.nansum(vs.msa_ss[2:-2, 2:-2, vs.tau, :] * vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    return KernelOutput(SA_ss=vs.SA_ss, C_ss=vs.C_ss, M_ss=vs.M_ss)


@roger_kernel
def calculate_subsoil_transport_anion_kernel(state):
    """
    Calculates StorAge and solute concentration
    """
    vs = state.variables

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.MSA_ss = update(
        vs.MSA_ss,
        at[2:-2, 2:-2, :, :], transport.calc_MSA(state, vs.MSA_ss, vs.msa_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.M_ss = update(
        vs.M_ss,
        at[2:-2, 2:-2, vs.tau], npx.sum(vs.msa_ss[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.C_ss = update(
        vs.C_ss,
        at[2:-2, 2:-2, vs.tau], vs.M_ss[2:-2, 2:-2, vs.tau] / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1),
    )

    return KernelOutput(SA_ss=vs.SA_ss, MSA_ss=vs.MSA_ss, C_ss=vs.C_ss, M_ss=vs.M_ss)


@roger_routine
def calculate_subsoil_transport(state):
    """
    Calculates StorAge (and solute concentration)
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_chloride & settings.enable_bromide & settings.enable_oxygen18 & settings.enable_deuterium & settings.enable_nitrate):
        vs.update(calculate_subsoil_transport_kernel(state))

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(calculate_subsoil_transport_iso_kernel(state))

    if settings.enable_offline_transport and (settings.enable_chloride | settings.enable_bromide | settings.enable_nitrate):
        vs.update(calculate_subsoil_transport_anion_kernel(state))
