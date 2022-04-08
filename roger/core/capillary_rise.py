from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at
from roger.core import transport


@roger_kernel
def calc_cpr_rz(state):
    """
    Calculates capillary rise from subsoil into root zone
    """
    vs = state.variables
    settings = state.settings

    dh = allocate(state.dimensions, ("x", "y"))
    dz = allocate(state.dimensions, ("x", "y"))

    # pressure head gradient
    dh = update(
        dh,
        at[:, :], (vs.h_rz[:, :, vs.tau] - vs.h_ss[:, :, vs.tau]) * 10.2 * vs.maskCatch,
    )
    # gravitational head gradient
    dz = update(
        dz,
        at[:, :], ((vs.z_root[:, :, vs.tau] + (vs.z_soil - vs.z_root[:, :, vs.tau])/2) - vs.z_root[:, :, vs.tau]/2) * vs.maskCatch,
    )

    # vertical uplift using the Richards equation
    vs.cpr_rz = update(
        vs.cpr_rz,
        at[:, :], (-1) * vs.k_rz[:, :, vs.tau] * (dh/dz + 1) * vs.dt * vs.maskCatch,
    )
    vs.cpr_rz = update(
        vs.cpr_rz,
        at[:, :], npx.where(vs.cpr_rz < 0, 0, vs.cpr_rz) * vs.maskCatch,
    )

    if settings.enable_film_flow:
        vs.cpr_rz = update(
            vs.cpr_rz,
            at[:, :], npx.where(npx.sum(vs.S_f, axis=-1) > 0, 0, vs.cpr_rz) * vs.maskCatch,
        )

    # limit uplift to available fine pore storage in subsoil
    vs.cpr_rz = update(
        vs.cpr_rz,
        at[:, :], npx.where(vs.cpr_rz > vs.S_fp_ss, vs.S_fp_ss, vs.cpr_rz) * vs.maskCatch,
    )

    # update root zone storage and subsoil storage
    mask1 = (vs.cpr_rz > 0) & (vs.S_lp_ss <= 0) & (vs.event_id <= 0)
    mask2 = (vs.cpr_rz > 0) & (vs.S_lp_ss > 0) & (vs.cpr_rz <= vs.S_lp_ss) & (vs.event_id <= 0)
    mask3 = (vs.cpr_rz > 0) & (vs.S_lp_ss > 0) & (vs.cpr_rz > vs.S_lp_ss) & (vs.event_id <= 0)
    vs.S_fp_rz = update_add(
        vs.S_fp_rz,
        at[:, :], mask1 * vs.cpr_rz * vs.maskCatch,
    )
    vs.S_fp_ss = update_add(
        vs.S_fp_ss,
        at[:, :], mask1 * -vs.cpr_rz * vs.maskCatch,
    )
    vs.S_fp_rz = update_add(
        vs.S_fp_rz,
        at[:, :], mask2 * vs.cpr_rz * vs.maskCatch,
    )
    vs.S_lp_ss = update_add(
        vs.S_lp_ss,
        at[:, :], mask2 * -vs.cpr_rz * vs.maskCatch,
    )
    vs.S_fp_rz = update_add(
        vs.S_fp_rz,
        at[:, :], mask3 * vs.cpr_rz * vs.maskCatch,
    )
    vs.S_fp_ss = update_add(
        vs.S_fp_ss,
        at[:, :], mask3 * -(vs.cpr_rz - vs.S_lp_ss) * vs.maskCatch,
    )
    vs.S_lp_ss = update(
        vs.S_lp_ss,
        at[:, :], npx.where(mask3, 0, vs.S_lp_ss) * vs.maskCatch,
    )

    # fine pore excess fills large pores
    mask4 = (vs.S_fp_rz > vs.S_ufc_rz)
    vs.S_lp_rz = update_add(
        vs.S_lp_rz,
        at[:, :], mask4 * (vs.S_fp_rz - vs.S_ufc_rz) * vs.maskCatch,
    )
    vs.S_fp_rz = update(
        vs.S_fp_rz,
        at[:, :], npx.where(mask4, vs.S_ufc_rz, vs.S_fp_rz) * vs.maskCatch,
    )

    return KernelOutput(cpr_rz=vs.cpr_rz, S_fp_rz=vs.S_fp_rz, S_lp_rz=vs.S_lp_rz,
                        S_fp_ss=vs.S_fp_ss, S_lp_ss=vs.S_lp_ss)


@roger_kernel
def calc_cpr_ss(state):
    """
    Calculates capillary rise from groundwater into subsoil
    """
    vs = state.variables
    settings = state.settings

    z = allocate(state.dimensions, ("x", "y"))

    # subsoil is saturated
    mask1 = (vs.z_sat[:, :, vs.tau] > 0)
    # capillary rise from groundwater table towards subsoil
    if settings.enable_groundwater_boundary | settings.enable_groundwater:
        z = allocate(state.dimensions, ("x", "y"))
        z = update(
            z,
            at[:, :], (vs.z_gw[:, :, vs.tau] * 1000 - vs.z_root[:, :, vs.tau]) * vs.maskCatch,
        )

        vs.cpr_ss = update(
            vs.cpr_ss,
            at[:, :], (vs.ks * vs.dt * (npx.power((-z)/(vs.ha*10), -vs.n_salv) - npx.power(vs.h_ss[:, :, vs.tau]/vs.ha, -vs.n_salv))/(1 + npx.power(vs.h_ss[:, :, vs.tau]/vs.ha, -vs.n_salv) + (vs.n_salv - 1) * npx.power((z*-1)/(vs.ha*10), -vs.n_salv))) * vs.maskCatch,
        )
        vs.cpr_ss = update(
            vs.cpr_ss,
            at[:, :], npx.where(npx.isnan(vs.cpr_ss), npx.power((-z)/(vs.ha*10), -vs.n_salv)/(1 + (vs.n_salv - 1) * npx.power((z*-1)/(vs.ha*10), -vs.n_salv)), vs.cpr_ss) * vs.maskCatch,
        )
        vs.cpr_ss = update(
            vs.cpr_ss,
            at[:, :], npx.where(vs.cpr_ss < 0, 0, vs.cpr_ss) * vs.maskCatch,
        )
        if settings.enable_film_flow:
            vs.cpr_ss = update(
                vs.cpr_ss,
                at[:, :], npx.where(npx.sum(vs.S_f, axis=-1) > 0, 0, vs.cpr_rz) * vs.maskCatch,
            )
        # no capillary rise if subsoil is saturated
        vs.cpr_ss = update(
            vs.cpr_ss,
            at[:, :], npx.where(mask1, 0, vs.cpr_ss) * vs.maskCatch,
        )
        # reducing capillary rise if large pores of subsoil contain water
        mask2 = (vs.z_sat[:, :, vs.tau] > 0) & (vs.cpr_ss > vs.S_lp_ss)
        vs.cpr_ss = update(
            vs.cpr_ss,
            at[:, :], npx.where(mask2, vs.cpr_ss - vs.S_lp_ss, vs.cpr_ss) * vs.maskCatch,
        )

        # update subsoil storage
        vs.S_fp_ss = update_add(
            vs.S_fp_ss,
            at[:, :], vs.cpr_ss * vs.maskCatch,
        )

        # fine pore excess fills large pores
        mask3 = (vs.S_fp_ss > vs.S_ufc_ss)
        vs.S_lp_ss = update_add(
            vs.S_lp_ss,
            at[:, :], mask3 * (vs.S_fp_ss - vs.S_ufc_ss) * vs.maskCatch,
        )
        vs.S_fp_ss = update(
            vs.S_fp_ss,
            at[:, :], npx.where(mask3, vs.S_ufc_ss, vs.S_fp_ss) * vs.maskCatch,
        )

    return KernelOutput(cpr_ss=vs.cpr_ss, S_fp_ss=vs.S_fp_ss, S_lp_ss=vs.S_lp_ss)


@roger_kernel
def update_groundwater(state):
    """
    Update groundwater storage
    """
    vs = state.variables
    settings = state.settings

    # capillary rise from groundwater table towards subsoil
    if settings.enable_groundwater:
        vs.S_gw = update_add(
            vs.S_gw,
            at[:, :, vs.tau], -vs.cpr_ss * vs.maskCatch,
        )

    return KernelOutput(S_gw=vs.S_gw)


@roger_routine
def calculate_capillary_rise(state):
    """
    Calculates capillary rise
    """
    vs = state.variables
    settings = state.settings

    vs.update(calc_cpr_rz(state))
    if settings.enable_groundwater_boundary | settings.enable_groundwater:
        vs.update(calc_cpr_ss(state))
    if settings.enable_groundwater:
        vs.update(update_groundwater(state))


@roger_kernel
def calculate_capillary_rise_rz_transport_kernel(state):
    """
    Calculates travel time of capillary rise
    """
    vs = state.variables

    vs.SA_ss = update(
        vs.SA_ss,
        at[:, :, :, :], transport.calc_SA(state, vs.SA_ss, vs.sa_ss) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.tt_cpr_rz = update(
        vs.tt_cpr_rz,
        at[:, :, :], transport.calc_tt(state, vs.SA_ss, vs.sa_ss, vs.cpr_rz, vs.sas_params_cpr_rz) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.TT_cpr_rz = update(
        vs.TT_cpr_rz,
        at[:, :, 1:], npx.cumsum(vs.tt_cpr_rz, axis=2),
    )

    vs.sa_ss = update(
        vs.sa_ss,
        at[:, :, :, :], transport.update_sa(state, vs.sa_ss, vs.tt_cpr_rz, vs.cpr_rz) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.sa_rz = update_add(
        vs.sa_rz,
        at[:, :, vs.tau, :], vs.tt_cpr_rz * vs.cpr_rz[:, :, npx.newaxis] * vs.maskCatch[:, :, npx.newaxis],
    )

    return KernelOutput(sa_ss=vs.sa_ss, tt_cpr_rz=vs.tt_cpr_rz, TT_cpr_rz=vs.TT_cpr_rz, sa_rz=vs.sa_rz)


@roger_kernel
def calculate_capillary_rise_rz_transport_iso_kernel(state):
    """
    Calculates isotope transport of capillary rise
    """
    vs = state.variables

    vs.SA_ss = update(
        vs.SA_ss,
        at[:, :, :, :], transport.calc_SA(state, vs.SA_ss, vs.sa_ss) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.tt_cpr_rz = update(
        vs.tt_cpr_rz,
        at[:, :, :], transport.calc_tt(state, vs.SA_ss, vs.sa_ss, vs.cpr_rz, vs.sas_params_cpr_rz) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.TT_cpr_rz = update(
        vs.TT_cpr_rz,
        at[:, :, 1:], npx.cumsum(vs.tt_cpr_rz, axis=2),
    )

    # calculate solute travel time distribution
    alpha = allocate(state.dimensions, ("x", "y"), fill=1)
    vs.mtt_cpr_rz = update(
        vs.mtt_cpr_rz,
        at[:, :, :], transport.calc_mtt(state, vs.sa_ss, vs.tt_cpr_rz, vs.cpr_rz, vs.msa_ss, alpha) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.C_cpr_rz = update(
        vs.C_cpr_rz,
        at[:, :], transport.calc_conc_iso_flux(state, vs.mtt_cpr_rz, vs.tt_cpr_rz) * vs.maskCatch,
    )

    # update StorAge with flux
    vs.sa_ss = update(
        vs.sa_ss,
        at[:, :, :, :], transport.update_sa(state, vs.sa_ss, vs.tt_cpr_rz, vs.cpr_rz) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )
    # update solute StorAge
    vs.msa_ss = update(
        vs.msa_ss,
        at[:, :, :, :], npx.where((vs.sa_ss > 0), vs.msa_ss, npx.NaN) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.msa_rz = update(
        vs.msa_rz,
        at[:, :, :, :], transport.calc_msa_iso(state, vs.sa_rz, vs.msa_rz, vs.cpr_rz, vs.tt_cpr_rz, vs.mtt_cpr_rz) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.sa_rz = update_add(
        vs.sa_rz,
        at[:, :, vs.tau, :], vs.tt_cpr_rz * vs.cpr_rz[:, :, npx.newaxis] * vs.maskCatch[:, :, npx.newaxis],
    )

    return KernelOutput(sa_ss=vs.sa_ss, msa_ss=vs.msa_ss, tt_cpr_rz=vs.tt_cpr_rz, TT_cpr_rz=vs.TT_cpr_rz, mtt_cpr_rz=vs.mtt_cpr_rz, C_cpr_rz=vs.C_cpr_rz, sa_rz=vs.sa_rz, msa_rz=vs.msa_rz)


@roger_kernel
def calculate_capillary_rise_rz_transport_anion_kernel(state):
    """
    Calculates chloride/bromide/nitrate transport of capillary rise
    """
    vs = state.variables

    vs.SA_ss = update(
        vs.SA_ss,
        at[:, :, :, :], transport.calc_SA(state, vs.SA_ss, vs.sa_ss) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.tt_cpr_rz = update(
        vs.tt_cpr_rz,
        at[:, :, :], transport.calc_tt(state, vs.SA_ss, vs.sa_ss, vs.cpr_rz, vs.sas_params_cpr_rz) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.TT_cpr_rz = update(
        vs.TT_cpr_rz,
        at[:, :, 1:], npx.cumsum(vs.tt_cpr_rz, axis=2),
    )

    # calculate isotope travel time distribution
    vs.mtt_cpr_rz = update(
        vs.mtt_cpr_rz,
        at[:, :, :], transport.calc_mtt(state, vs.sa_ss, vs.tt_cpr_rz, vs.cpr_rz, vs.msa_ss, vs.alpha_q) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.C_cpr_rz = update(
        vs.C_cpr_rz,
        at[:, :], npx.where(vs.cpr_rz > 0, npx.sum(vs.mtt_cpr_rz, axis=2) / vs.cpr_rz, 0) * vs.maskCatch,
    )

    # update StorAge with flux
    vs.sa_ss = update(
        vs.sa_ss,
        at[:, :, :, :], transport.update_sa(state, vs.sa_ss, vs.tt_cpr_rz, vs.cpr_rz) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )
    # update solute StorAge
    vs.msa_ss = update_add(
        vs.msa_ss,
        at[:, :, vs.tau, :], - vs.mtt_cpr_rz * vs.maskCatch[:, :, npx.newaxis],
    )

    # update StorAge with flux
    vs.sa_rz = update_add(
        vs.sa_rz,
        at[:, :, vs.tau, :], vs.tt_cpr_rz * vs.cpr_rz[:, :, npx.newaxis] * vs.maskCatch[:, :, npx.newaxis],
    )
    # update solute StorAge
    vs.msa_rz = update_add(
        vs.msa_rz,
        at[:, :, vs.tau, :], vs.mtt_cpr_rz * vs.maskCatch[:, :, npx.newaxis],
    )

    return KernelOutput(sa_ss=vs.sa_ss, msa_ss=vs.msa_ss, tt_cpr_rz=vs.tt_cpr_rz, TT_cpr_rz=vs.TT_cpr_rz, mtt_cpr_rz=vs.mtt_cpr_rz, C_cpr_rz=vs.C_cpr_rz, sa_rz=vs.sa_rz, msa_rz=vs.msa_rz)


@roger_kernel
def calculate_capillary_rise_ss_transport_kernel(state):
    """
    Calculates travel time of capillary rise
    """
    pass


@roger_kernel
def calculate_capillary_rise_ss_transport_iso_kernel(state):
    """
    Calculates isotope transport of capillary rise
    """
    pass


@roger_kernel
def calculate_capillary_rise_ss_transport_anion_kernel(state):
    """
    Calculates chloride/bromide/nitrate transport of capillary rise
    """
    pass


@roger_routine
def calculate_capillary_rise_rz_transport(state):
    """
    Calculates capillary rise transport
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate):
        vs.update(calculate_capillary_rise_rz_transport_kernel(state))

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(calculate_capillary_rise_rz_transport_iso_kernel(state))

    if settings.enable_offline_transport and (settings.enable_chloride | settings.enable_bromide | settings.enable_nitrate):
        vs.update(calculate_capillary_rise_rz_transport_anion_kernel(state))


@roger_routine
def calculate_capillary_rise_ss_transport(state):
    """
    Calculates capillary rise transport
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate):
        vs.update(calculate_capillary_rise_ss_transport_kernel(state))

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(calculate_capillary_rise_ss_transport_iso_kernel(state))

    if settings.enable_offline_transport and (settings.enable_chloride | settings.enable_bromide | settings.enable_nitrate):
        vs.update(calculate_capillary_rise_ss_transport_anion_kernel(state))
