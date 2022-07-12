from roger import roger_kernel, roger_routine, KernelOutput
from roger.core.operators import numpy as npx, update, at
from roger.core import transport


@roger_kernel
def calc_k(state):
    """
    Calculates hydraulic conductivity of root zone
    """
    vs = state.variables

    vs.k_rz = update(
        vs.k_rz,
        at[2:-2, 2:-2, vs.tau], (vs.ks[2:-2, 2:-2]/(1 + (vs.theta_rz[2:-2, 2:-2, vs.tau]/vs.theta_sat[2:-2, 2:-2])**(-vs.m_bc[2:-2, 2:-2]))) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(k_rz=vs.k_rz)


@roger_kernel
def calc_h(state):
    """
    Calculates soil water potential of root zone
    """
    vs = state.variables

    vs.h_rz = update(
        vs.h_rz,
        at[2:-2, 2:-2, vs.tau], (vs.ha[2:-2, 2:-2]/((vs.theta_rz[2:-2, 2:-2, vs.tau]/vs.theta_sat[2:-2, 2:-2])**(1/vs.lambda_bc[2:-2, 2:-2]))) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(h_rz=vs.h_rz)


@roger_kernel
def calc_theta(state):
    """
    Calculates soil water content of root zone
    """
    vs = state.variables

    vs.theta_rz = update(
        vs.theta_rz,
        at[2:-2, 2:-2, vs.tau], ((vs.S_fp_rz[2:-2, 2:-2] + vs.S_lp_rz[2:-2, 2:-2])/vs.z_root[2:-2, 2:-2, vs.tau] + vs.theta_pwp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(theta_rz=vs.theta_rz)


@roger_kernel
def calc_theta_ff(state):
    """
    Calculates soil water conent including film flow
    """
    vs = state.variables

    vs.theta_rz_ff = update(
        vs.theta_rz_ff,
        at[2:-2, 2:-2, vs.tau], npx.sum(vs.S_f_rz[2:-2, 2:-2, :], axis=-1) / vs.z_root[2:-2, 2:-2, vs.tau] + vs.theta_rz[2:-2, 2:-2, vs.tau],
    )

    return KernelOutput(
        theta_rz_ff=vs.theta_rz_ff,
        )


@roger_kernel
def calc_S(state):
    """
    Calculates soil water content of root zone
    """
    vs = state.variables

    vs.S_rz = update(
        vs.S_rz,
        at[2:-2, 2:-2, vs.tau], (vs.S_pwp_rz[2:-2, 2:-2] + vs.S_fp_rz[2:-2, 2:-2] + vs.S_lp_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(S_rz=vs.S_rz)


@roger_kernel
def calc_dS(state):
    """
    Calculates storage change of root zone
    """
    vs = state.variables

    vs.dS_rz = update(
        vs.dS_rz,
        at[2:-2, 2:-2], (vs.S_rz[2:-2, 2:-2, vs.tau] - vs.S_rz[2:-2, 2:-2, vs.taum1]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(dS_rz=vs.dS_rz)


@roger_routine
def calculate_root_zone(state):
    """
    Calculates root zone storage and storage dependent variables
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
def calculate_root_zone_transport_kernel(state):
    """
    Calculates StorAge
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    return KernelOutput(SA_rz=vs.SA_rz)


@roger_kernel
def calculate_root_zone_transport_iso_kernel(state):
    """
    Calculates StorAge and isotope ratio
    """
    vs = state.variables

    vs.C_rz = update(
        vs.C_rz,
        at[2:-2, 2:-2, vs.tau], transport.calc_conc_iso_storage(state, vs.sa_rz, vs.msa_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.M_rz = update(
        vs.M_rz,
        at[2:-2, 2:-2, vs.tau], npx.nansum(vs.msa_rz[2:-2, 2:-2, vs.tau, :] * vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    return KernelOutput(SA_rz=vs.SA_rz, C_rz=vs.C_rz, M_rz=vs.M_rz)


@roger_kernel
def calculate_root_zone_transport_anion_kernel(state):
    """
    Calculates StorAge and solute concentration
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.MSA_rz = update(
        vs.MSA_rz,
        at[2:-2, 2:-2, :, :], transport.calc_MSA(state, vs.MSA_rz, vs.msa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.M_rz = update(
        vs.M_rz,
        at[2:-2, 2:-2, vs.tau], npx.sum(vs.msa_rz[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.C_rz = update(
        vs.C_rz,
        at[2:-2, 2:-2, vs.tau], vs.M_rz[2:-2, 2:-2, vs.tau] / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1),
    )

    return KernelOutput(SA_rz=vs.SA_rz, MSA_rz=vs.MSA_rz, C_rz=vs.C_rz, M_rz=vs.M_rz)


@roger_routine
def calculate_root_zone_transport(state):
    """
    Calculates StorAge (and solute concentration)
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate):
        vs.update(calculate_root_zone_transport_kernel(state))

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(calculate_root_zone_transport_iso_kernel(state))

    if settings.enable_offline_transport and (settings.enable_chloride | settings.enable_bromide | settings.enable_nitrate):
        vs.update(calculate_root_zone_transport_anion_kernel(state))
