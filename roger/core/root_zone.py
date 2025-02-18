from roger import roger_kernel, roger_routine, KernelOutput
from roger.core.operators import numpy as npx, update, at
from roger.variables import allocate
from roger.core import transport


@roger_kernel
def calc_irrigation_demand(state):
    """
    Calculates irrigation demand
    """
    vs = state.variables

    # calculate the fine pore deficit
    fine_pore_deficit = allocate(state.dimensions, ("x", "y"))
    fine_pore_deficit = update(
        fine_pore_deficit,
        at[2:-2, 2:-2],
        vs.theta_fc[2:-2, 2:-2] - vs.theta_rz[2:-2, 2:-2, vs.tau],
    )
    fine_pore_deficit = update(
        fine_pore_deficit,
        at[2:-2, 2:-2],
        npx.where(fine_pore_deficit[2:-2, 2:-2] < 0, 0, fine_pore_deficit[2:-2, 2:-2]),
    )

    vs.irr_demand = update(
        vs.irr_demand,
        at[2:-2, 2:-2],
        fine_pore_deficit[2:-2, 2:-2] * vs.z_root[2:-2, 2:-2, vs.tau],
    )

    return KernelOutput(irr_demand=vs.irr_demand)


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

    # # horizontal redistribution
    # mask = (vs.S_fp_rz < vs.S_ufc_rz) & (vs.S_fp_rz >= 0) & (vs.S_lp_rz < vs.S_ac_rz) & (vs.S_lp_rz > 0)
    # S_fp_rz = allocate(state.dimensions, ("x", "y"))
    # S_fp_rz = update(
    #     S_fp_rz,
    #     at[2:-2, 2:-2], vs.S_fp_rz[2:-2, 2:-2],
    # )
    # vs.S_fp_rz = update(
    #     vs.S_fp_rz,
    #     at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], npx.where((vs.S_ufc_rz[2:-2, 2:-2] - vs.S_fp_rz[2:-2, 2:-2] > vs.S_lp_rz[2:-2, 2:-2]), vs.S_fp_rz[2:-2, 2:-2] + vs.S_lp_rz[2:-2, 2:-2], vs.S_ufc_rz[2:-2, 2:-2]), vs.S_fp_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    # )
    # vs.S_lp_rz = update(
    #     vs.S_lp_rz,
    #     at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], npx.where((vs.S_ufc_rz[2:-2, 2:-2] - S_fp_rz[2:-2, 2:-2] > vs.S_lp_rz[2:-2, 2:-2]), 0, vs.S_lp_rz[2:-2, 2:-2] - (vs.S_ufc_rz[2:-2, 2:-2] - S_fp_rz[2:-2, 2:-2])), vs.S_lp_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    # )

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

    vs.update(calc_S(state))
    vs.update(calc_dS(state))
    vs.update(calc_theta(state))
    vs.update(calc_irrigation_demand(state))
    vs.update(calc_k(state))
    vs.update(calc_h(state))

    if settings.enable_film_flow:
        vs.update(calc_theta_ff(state))


@roger_kernel
def calc_root_zone_transport_kernel(state):
    """
    Calculates StorAge
    """
    vs = state.variables

    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, vs.tau, :], npx.where(vs.sa_rz[2:-2, 2:-2, vs.tau, :] < 1e-8, 0, vs.sa_rz[2:-2, 2:-2, vs.tau, :]),
    )

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    return KernelOutput(SA_rz=vs.SA_rz)


@roger_kernel
def calc_root_zone_transport_iso_kernel(state):
    """
    Calculates StorAge and isotope ratio
    """
    vs = state.variables

    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, vs.tau, :], npx.where(vs.sa_rz[2:-2, 2:-2, vs.tau, :] < 1e-8, 0, vs.sa_rz[2:-2, 2:-2, vs.tau, :]),
    )
    vs.csa_rz = update(
        vs.csa_rz,
        at[2:-2, 2:-2, vs.tau, :], transport.conc_to_delta(state, vs.msa_rz[2:-2, 2:-2, vs.tau, :]),
    )
    vs.C_rz = update(
        vs.C_rz,
        at[2:-2, 2:-2, vs.tau], transport.calc_conc_iso_storage(state, vs.sa_rz, vs.msa_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_rz = update(
        vs.C_iso_rz,
        at[2:-2, 2:-2, vs.tau], transport.conc_to_delta(state, vs.C_rz[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    return KernelOutput(SA_rz=vs.SA_rz, C_rz=vs.C_rz, C_iso_rz=vs.C_iso_rz, csa_rz=vs.csa_rz)


@roger_kernel
def calc_root_zone_transport_anion_kernel(state):
    """
    Calculates StorAge and solute concentration
    """
    vs = state.variables

    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, vs.tau, :], npx.where(vs.sa_rz[2:-2, 2:-2, vs.tau, :] < 1e-8, 0, vs.sa_rz[2:-2, 2:-2, vs.tau, :]),
    )

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.msa_rz = update(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, :], npx.where(vs.sa_rz[2:-2, 2:-2, vs.tau, :] <= 0, 0, vs.msa_rz[2:-2, 2:-2, vs.tau, :]),
    )

    vs.csa_rz = update(
        vs.csa_rz,
        at[2:-2, 2:-2, :, :], npx.where(vs.sa_rz[2:-2, 2:-2, :, :] > 0, vs.msa_rz[2:-2, 2:-2, :, :] / vs.sa_rz[2:-2, 2:-2, :, :], 0) * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.M_rz = update(
        vs.M_rz,
        at[2:-2, 2:-2, vs.tau], npx.nansum(vs.msa_rz[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.C_rz = update(
        vs.C_rz,
        at[2:-2, 2:-2, vs.tau], npx.where(npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1) > 0, vs.M_rz[2:-2, 2:-2, vs.tau] / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1), 0),
    )

    return KernelOutput(SA_rz=vs.SA_rz, C_rz=vs.C_rz, M_rz=vs.M_rz, csa_rz=vs.csa_rz, msa_rz=vs.msa_rz)


@roger_routine
def calculate_root_zone_transport(state):
    """
    Calculates StorAge (and solute concentration)
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate | settings.enable_virtualtracer):
        vs.update(calc_root_zone_transport_kernel(state))

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(calc_root_zone_transport_iso_kernel(state))

    if settings.enable_offline_transport and (settings.enable_chloride | settings.enable_bromide | settings.enable_nitrate | settings.enable_virtualtracer):
        vs.update(calc_root_zone_transport_anion_kernel(state))
