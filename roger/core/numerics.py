from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.distributed import global_and
from roger.core.operators import numpy as npx, update, update_multiply, for_loop, at
from roger.core import transport
from roger.core.groundwater import _ss_z
from roger.core.utilities import _get_row_no
from roger import runtime_settings as rs, logger


@roger_kernel
def rescale_SA_soil_kernel(state):
    """
    Rescale StorAge.
    """
    vs = state.variables

    vs.sa_rz = update_multiply(
        vs.sa_rz,
        at[2:-2, 2:-2, 0, :], vs.S_rz_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
    )
    vs.sa_ss = update_multiply(
        vs.sa_ss,
        at[2:-2, 2:-2, 0, :], vs.S_ss_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
    )

    vs.sa_rz = update_multiply(
        vs.sa_rz,
        at[2:-2, 2:-2, 1, :], vs.S_rz_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
    )
    vs.sa_ss = update_multiply(
        vs.sa_ss,
        at[2:-2, 2:-2, 1, :], vs.S_ss_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
    )

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :2, 1:], npx.cumsum(vs.sa_rz[2:-2, 2:-2, :2, :], axis=-1),
    )
    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :2, 0], 0,
    )

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :2, 1:], npx.cumsum(vs.sa_ss[2:-2, 2:-2, :2, :], axis=-1),
    )
    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :2, 0], 0,
    )

    vs.sa_s = update(
        vs.sa_s,
        at[2:-2, 2:-2, :2, :], vs.sa_rz[2:-2, 2:-2, :2, :] + vs.sa_ss[2:-2, 2:-2, :2, :],
    )

    vs.SA_s = update(
        vs.SA_s,
        at[2:-2, 2:-2, :2, 1:], npx.cumsum(vs.sa_s[2:-2, 2:-2, :2, :], axis=-1),
    )
    vs.SA_s = update(
        vs.SA_s,
        at[2:-2, 2:-2, :2, 0], 0,
    )

    return KernelOutput(
        sa_rz=vs.sa_rz, sa_ss=vs.sa_ss, sa_s=vs.sa_s, SA_rz=vs.SA_rz, SA_ss=vs.SA_ss, SA_s=vs.SA_s,
    )


@roger_kernel
def rescale_SA_MSA_soil_kernel(state):
    """
    Rescale solute concentration.
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_bromide:
        vs.C_rz = update(
            vs.C_rz,
            at[2:-2, 2:-2, :], 0,
        )

        vs.C_ss = update(
            vs.C_ss,
            at[2:-2, 2:-2, :], 0,
        )

        vs.C_s = update(
            vs.C_s,
            at[2:-2, 2:-2, :], 0,
        )

    elif (settings.enable_chloride | settings.enable_nitrate):
        vs.msa_rz = update_multiply(
            vs.msa_rz,
            at[2:-2, 2:-2, 0, :], vs.S_rz_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.msa_ss = update_multiply(
            vs.msa_ss,
            at[2:-2, 2:-2, 0, :], vs.S_ss_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.msa_rz = update_multiply(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :], vs.S_rz_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.msa_ss = update_multiply(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :], vs.S_ss_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.sa_rz = update_multiply(
            vs.sa_rz,
            at[2:-2, 2:-2, 0, :], vs.S_rz_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.sa_ss = update_multiply(
            vs.sa_ss,
            at[2:-2, 2:-2, 0, :], vs.S_ss_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.sa_rz = update_multiply(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :], vs.S_rz_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.sa_ss = update_multiply(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :], vs.S_ss_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.SA_rz = update(
            vs.SA_rz,
            at[2:-2, 2:-2, :2, 1:], npx.cumsum(vs.sa_rz[2:-2, 2:-2, :2, :], axis=-1),
        )
        vs.SA_rz = update(
            vs.SA_rz,
            at[2:-2, 2:-2, :2, 0], 0,
        )
        vs.SA_ss = update(
            vs.SA_ss,
            at[2:-2, 2:-2, :2, 1:], npx.cumsum(vs.sa_ss[2:-2, 2:-2, :2, :], axis=-1),
        )
        vs.SA_ss = update(
            vs.SA_ss,
            at[2:-2, 2:-2, :2, 0], 0,
        )
        vs.sa_s = update(
            vs.sa_s,
            at[2:-2, 2:-2, :2, :], vs.sa_rz[2:-2, 2:-2, :2, :] + vs.sa_ss[2:-2, 2:-2, :2, :],
        )
        vs.SA_s = update(
            vs.SA_s,
            at[2:-2, 2:-2, :2, 1:], npx.cumsum(vs.sa_s[2:-2, 2:-2, :2, :], axis=-1),
        )
        vs.SA_s = update(
            vs.SA_s,
            at[2:-2, 2:-2, :2, 0], 0,
        )
        vs.C_rz = update(
            vs.C_rz,
            at[2:-2, 2:-2, :2], npx.sum(vs.msa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis] / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.C_ss = update(
            vs.C_ss,
            at[2:-2, 2:-2, :2], npx.sum(vs.msa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis] / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.msa_s = update(
            vs.msa_s,
            at[2:-2, 2:-2, :, :], vs.msa_rz[2:-2, 2:-2, :, :] + vs.msa_ss[2:-2, 2:-2, :, :],
        )
        vs.C_s = update(
            vs.C_s,
            at[2:-2, 2:-2, :2], npx.sum(vs.msa_s[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis] / npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )

    elif (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.sa_rz = update_multiply(
            vs.sa_rz,
            at[2:-2, 2:-2, 0, :], vs.S_rz_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.sa_ss = update_multiply(
            vs.sa_ss,
            at[2:-2, 2:-2, 0, :], vs.S_ss_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.sa_rz = update_multiply(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :], vs.S_rz_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.sa_ss = update_multiply(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :], vs.S_ss_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.SA_rz = update(
            vs.SA_rz,
            at[2:-2, 2:-2, :2, 1:], npx.cumsum(vs.sa_rz[2:-2, 2:-2, :2, :], axis=-1),
        )
        vs.SA_rz = update(
            vs.SA_rz,
            at[2:-2, 2:-2, :2, 0], 0,
        )
        vs.SA_ss = update(
            vs.SA_ss,
            at[2:-2, 2:-2, :2, 1:], npx.cumsum(vs.sa_ss[2:-2, 2:-2, :2, :], axis=-1),
        )
        vs.SA_ss = update(
            vs.SA_ss,
            at[2:-2, 2:-2, :2, 0], 0,
        )
        vs.sa_s = update(
            vs.sa_s,
            at[2:-2, 2:-2, :, :], vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :],
        )
        vs.SA_s = update(
            vs.SA_s,
            at[2:-2, 2:-2, :2, 1:], npx.cumsum(vs.sa_s[2:-2, 2:-2, :2, :], axis=-1),
        )
        vs.SA_s = update(
            vs.SA_s,
            at[2:-2, 2:-2, :2, 0], 0,
        )
        vs.C_rz = update(
            vs.C_rz,
            at[2:-2, 2:-2, :2], transport.calc_conc_iso_storage(state, vs.sa_rz, vs.msa_rz)[2:-2, 2:-2, npx.newaxis],
        )
        vs.C_iso_rz = update(
            vs.C_iso_rz,
            at[2:-2, 2:-2, vs.taum1], transport.conc_to_delta(state, vs.C_rz[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_iso_rz = update(
            vs.C_iso_rz,
            at[2:-2, 2:-2, vs.tau], transport.conc_to_delta(state, vs.C_rz[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_ss = update(
            vs.C_ss,
            at[2:-2, 2:-2, :2], transport.calc_conc_iso_storage(state, vs.sa_ss, vs.msa_ss)[2:-2, 2:-2, npx.newaxis],
        )
        vs.C_iso_ss = update(
            vs.C_iso_ss,
            at[2:-2, 2:-2, vs.taum1], transport.conc_to_delta(state, vs.C_ss[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_iso_ss = update(
            vs.C_iso_ss,
            at[2:-2, 2:-2, vs.tau], transport.conc_to_delta(state, vs.C_ss[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.msa_s = update(
            vs.msa_s,
            at[2:-2, 2:-2, :, :], npx.where(vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :] > 0, vs.msa_rz[2:-2, 2:-2, :, :] * (vs.sa_rz[2:-2, 2:-2, :, :] / (vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :])) + vs.msa_ss[2:-2, 2:-2, :, :] * (vs.sa_ss[2:-2, 2:-2, :, :] / (vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :])), 0),
        )
        vs.msa_s = update(
            vs.msa_s,
            at[2:-2, 2:-2, :vs.taup1, :], npx.where(npx.isnan(vs.msa_s[2:-2, 2:-2, :vs.taup1, :]), 0, vs.msa_s[2:-2, 2:-2, :vs.taup1, :]),
        )
        vs.msa_s = update(
            vs.msa_s,
            at[2:-2, 2:-2, :vs.taup1, 0], 0,
        )
        vs.C_s = update(
            vs.C_s,
            at[2:-2, 2:-2, :2], transport.calc_conc_iso_storage(state, vs.sa_s, vs.msa_s)[2:-2, 2:-2, npx.newaxis],
        )
        vs.C_iso_s = update(
            vs.C_iso_s,
            at[2:-2, 2:-2, vs.taum1], transport.conc_to_delta(state, vs.C_s[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.C_iso_s = update(
            vs.C_iso_s,
            at[2:-2, 2:-2, vs.tau], transport.conc_to_delta(state, vs.C_s[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
        )

    return KernelOutput(
        C_rz=vs.C_rz, C_ss=vs.C_ss, C_s=vs.C_s, msa_rz=vs.msa_rz, msa_ss=vs.msa_ss, msa_s=vs.msa_s,
        sa_rz=vs.sa_rz, sa_ss=vs.sa_ss, sa_s=vs.sa_s, SA_rz=vs.SA_rz, SA_ss=vs.SA_ss, SA_s=vs.SA_s,
        C_iso_rz=vs.C_iso_rz, C_iso_ss=vs.C_iso_ss, C_iso_s=vs.C_iso_s,
    )


@roger_kernel
def rescale_SA_gw_kernel(state):
    pass


@roger_kernel
def rescale_conc_gw_kernel(state):
    pass


@roger_routine
def rescale_SA(state):
    """
    Rescale StorAge after warmup.
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_bromide | settings.enable_chloride | settings.enable_nitrate):
        vs.update(rescale_SA_MSA_soil_kernel(state))
    elif settings.enable_offline_transport and not (settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_bromide | settings.enable_chloride | settings.enable_nitrate):
        vs.update(rescale_SA_soil_kernel(state))

    if settings.enable_bromide:
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, :2, :], 0,
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, :2, :], 0,
        )
        vs.msa_s = update(
            vs.msa_s,
            at[2:-2, 2:-2, :2, :], 0,
        )

    if settings.enable_offline_transport & settings.enable_groundwater:
        vs.update(rescale_SA_gw_kernel(state))
        if (settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_bromide | settings.enable_chloride | settings.enable_nitrate):
            vs.update(rescale_conc_gw_kernel(state))


@roger_routine
def calc_grid(state):
    """
    setup grid based on dxt,dyt,dzt and x_origin, y_origin
    """
    pass


@roger_kernel
def calc_topo_kernel(state):
    vs = state.variables
    """
    River mask
    """
    river_mask = vs.lu_id == 20
    vs.maskRiver = update(vs.maskRiver, at[...], river_mask)

    """
    lake mask
    """
    lake_mask = vs.lu_id == 14
    vs.maskLake = update(vs.maskLake, at[...], lake_mask)

    """
    Catchment mask
    """
    catch_mask = (vs.lu_id != 14) & (vs.lu_id != 20) & (vs.lu_id != 999)
    vs.maskCatch = update(vs.maskCatch, at[...], catch_mask)

    return KernelOutput(
        maskLake=vs.maskLake,
        maskRiver=vs.maskRiver,
        maskCatch=vs.maskCatch,
    )


@roger_kernel
def calc_topo_gw_kernel(state):
    vs = state.variables
    """
    Boundary mask
    """
    grid = allocate(state.dimensions, ("x", "y"))
    catch = allocate(state.dimensions, ("x", "y"))
    catch = update(catch, at[...], npx.where(vs.maskCatch, 1, 0))
    vs.maskBoundGw = update(vs.maskBoundGw, at[...], npx.where(grid - catch < 0, True, False))

    return KernelOutput(
        maskBoundGw=vs.maskBoundGw,
    )


@roger_kernel
def calc_topo_urban_kernel(state):
    vs = state.variables
    """
    Urban mask
    """
    urban_mask = (vs.lu_id == 0) & (vs.lu_id == 31) & (vs.lu_id == 32) & (vs.lu_id == 33)
    vs.maskUrban = update(vs.maskUrban, at[...], urban_mask)

    return KernelOutput(
        maskUrban=vs.maskUrban,
    )


@roger_routine
def calc_topo(state):
    """
    Calulates masks
    """
    vs = state.variables
    settings = state.settings

    vs.update(calc_topo_kernel(state))

    if settings.enable_urban:
        vs.update(calc_topo_urban_kernel(state))

    if settings.enable_groundwater:
        vs.update(calc_topo_gw_kernel(state))


@roger_kernel
def calc_initial_conditions_surface_kernel(state):
    vs = state.variables

    vs.S_sur = update(
        vs.S_sur,
        at[2:-2, 2:-2, :2], (vs.S_int_top[2:-2, 2:-2, :2] + vs.S_int_ground[2:-2, 2:-2, :2] + vs.S_dep[2:-2, 2:-2, :2] + vs.S_snow[2:-2, 2:-2, :2]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(
        S_sur=vs.S_sur
    )


@roger_kernel
def calc_initial_conditions_soil_kernel(state):
    vs = state.variables

    vs.S_fp_s = update(
        vs.S_fp_s,
        at[2:-2, 2:-2], (vs.S_fp_rz[2:-2, 2:-2] + vs.S_fp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_lp_s = update(
        vs.S_lp_s,
        at[2:-2, 2:-2], (vs.S_lp_rz[2:-2, 2:-2] + vs.S_lp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_s = update(
        vs.S_s,
        at[2:-2, 2:-2, :2], (vs.S_rz[2:-2, 2:-2, :2] + vs.S_ss[2:-2, 2:-2, :2]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.theta = update(
        vs.theta,
        at[2:-2, 2:-2, :2], (vs.S_s[2:-2, 2:-2, :2] / vs.z_soil[2:-2, 2:-2, npx.newaxis]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(
        S_fp_s=vs.S_fp_s,
        S_lp_s=vs.S_lp_s,
        S_s=vs.S_s,
        theta=vs.theta,
    )


@roger_kernel
def calc_initial_conditions_root_zone_kernel(state):
    vs = state.variables

    mask1 = (vs.theta_rz[:, :, vs.tau] > vs.theta_pwp)
    mask2 = (vs.theta_rz[:, :, vs.tau] <= vs.theta_pwp)
    vs.theta_fp_rz = update(
        vs.theta_fp_rz,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], vs.theta_rz[2:-2, 2:-2, vs.tau] - vs.theta_pwp[2:-2, 2:-2], vs.theta_fp_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.theta_fp_rz = update(
        vs.theta_fp_rz,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], 0, vs.theta_fp_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    mask3 = (vs.theta_fp_rz >= vs.theta_ufc)
    vs.theta_fp_rz = update(
        vs.theta_fp_rz,
        at[2:-2, 2:-2], npx.where(mask3[2:-2, 2:-2], vs.theta_ufc[2:-2, 2:-2], vs.theta_fp_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    mask4 = (vs.theta_rz[:, :, vs.tau] > vs.theta_fc)
    mask5 = (vs.theta_rz[:, :, vs.tau] <= vs.theta_fc)
    vs.theta_lp_rz = update(
        vs.theta_lp_rz,
        at[2:-2, 2:-2], npx.where(mask4[2:-2, 2:-2], vs.theta_rz[2:-2, 2:-2, vs.tau] - vs.theta_fc[2:-2, 2:-2], vs.theta_lp_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.theta_lp_rz = update(
        vs.theta_lp_rz,
        at[2:-2, 2:-2], npx.where(mask5[2:-2, 2:-2], 0, vs.theta_lp_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_fp_rz = update(
        vs.S_fp_rz,
        at[2:-2, 2:-2], (vs.theta_fp_rz[2:-2, 2:-2] * vs.z_root[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_lp_rz = update(
        vs.S_lp_rz,
        at[2:-2, 2:-2], (vs.theta_lp_rz[2:-2, 2:-2] * vs.z_root[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_rz = update(
        vs.S_rz,
        at[2:-2, 2:-2, :2], (vs.S_pwp_rz[2:-2, 2:-2, npx.newaxis] + vs.S_fp_rz[2:-2, 2:-2, npx.newaxis] + vs.S_lp_rz[2:-2, 2:-2, npx.newaxis]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(
        theta_fp_rz=vs.theta_fp_rz,
        theta_lp_rz=vs.theta_lp_rz,
        S_fp_rz=vs.S_fp_rz,
        S_lp_rz=vs.S_lp_rz,
        S_rz=vs.S_rz,
    )


@roger_kernel
def calc_initial_conditions_subsoil_kernel(state):
    vs = state.variables

    mask6 = (vs.theta_ss[:, :, vs.tau] > vs.theta_pwp)
    mask7 = (vs.theta_ss[:, :, vs.tau] <= vs.theta_pwp)
    vs.theta_fp_ss = update(
        vs.theta_fp_ss,
        at[2:-2, 2:-2], npx.where(mask6[2:-2, 2:-2], vs.theta_ss[2:-2, 2:-2, vs.tau] - vs.theta_pwp[2:-2, 2:-2], vs.theta_fp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.theta_fp_ss = update(
        vs.theta_fp_ss,
        at[2:-2, 2:-2], npx.where(mask7[2:-2, 2:-2], 0, vs.theta_fp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    mask8 = (vs.theta_fp_ss >= vs.theta_ufc)
    vs.theta_fp_ss = update(
        vs.theta_fp_ss,
        at[2:-2, 2:-2], npx.where(mask8[2:-2, 2:-2], vs.theta_ufc[2:-2, 2:-2], vs.theta_fp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    mask9 = (vs.theta_ss[:, :, vs.tau] > vs.theta_fc)
    mask10 = (vs.theta_ss[:, :, vs.tau] <= vs.theta_fc)
    vs.theta_lp_ss = update(
        vs.theta_lp_ss,
        at[2:-2, 2:-2], npx.where(mask9[2:-2, 2:-2], vs.theta_ss[2:-2, 2:-2, vs.tau] - vs.theta_fc[2:-2, 2:-2], vs.theta_lp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )
    vs.theta_lp_ss = update(
        vs.theta_lp_ss,
        at[2:-2, 2:-2], npx.where(mask10[2:-2, 2:-2], 0, vs.theta_lp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_fp_ss = update(
        vs.S_fp_ss,
        at[2:-2, 2:-2], (vs.theta_fp_ss[2:-2, 2:-2] * (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau])) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_lp_ss = update(
        vs.S_lp_ss,
        at[2:-2, 2:-2], (vs.theta_lp_ss[2:-2, 2:-2] * (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau])) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_ss = update(
        vs.S_ss,
        at[2:-2, 2:-2, :2], (vs.S_pwp_ss[2:-2, 2:-2, npx.newaxis] + vs.S_fp_ss[2:-2, 2:-2, npx.newaxis] + vs.S_lp_ss[2:-2, 2:-2, npx.newaxis]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(
        theta_fp_ss=vs.theta_fp_ss,
        theta_lp_ss=vs.theta_lp_ss,
        S_fp_ss=vs.S_fp_ss,
        S_lp_ss=vs.S_lp_ss,
        S_ss=vs.S_ss,
    )


@roger_kernel
def calc_initial_conditions_kernel(state):
    vs = state.variables

    vs.S = update(
        vs.S,
        at[2:-2, 2:-2, :2], vs.S_sur[2:-2, 2:-2, :2] + vs.S_s[2:-2, 2:-2, :2] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(
        S=vs.S,
        )


@roger_kernel
def calc_initial_conditions_with_gw_kernel(state):
    vs = state.variables

    vs.S = update(
        vs.S,
        at[2:-2, 2:-2, :2], vs.S_sur[2:-2, 2:-2, :2] + vs.S_s[2:-2, 2:-2, :2] + vs.S_vad[2:-2, 2:-2, :2] + vs.S_gw[2:-2, 2:-2, :2] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(
        S=vs.S,
        )


@roger_kernel
def calc_initial_conditions_crops_kernel(state):
    pass


@roger_kernel
def calc_initial_conditions_transport_kernel(state):
    pass


@roger_kernel
def calc_initial_conditions_groundwater_kernel(state):
    vs = state.variables

    # Calculates storativity
    dz = allocate(state.dimensions, ("x", "y"))
    z = allocate(state.dimensions, ("x", "y", 1001))
    z = update(
        z,
        at[2:-2, 2:-2, :], npx.linspace(vs.z_gw[2:-2, 2:-2, vs.tau], vs.z_gw_tot, num=1001, axis=-1) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    dz = update(
        dz,
        at[2:-2, 2:-2], (z[2:-2, 2:-2, 1] - z[2:-2, 2:-2, 0]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_gw = update(
        vs.S_gw,
        at[2:-2, 2:-2, vs.taum1], (npx.sum(_ss_z(z, vs.n0[2:-2, 2:-2, npx.newaxis], vs.bdec[2:-2, 2:-2, npx.newaxis]), axis=-1) * dz) * 1000 * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_gw = update(
        vs.S_gw,
        at[2:-2, 2:-2, vs.tau], vs.S_gw[2:-2, 2:-2, vs.taum1],
    )

    return KernelOutput(S_gw=vs.S_gw)


@roger_routine
def calc_initial_conditions(state):
    """
    calculate storage, concentrations, etc
    """
    vs = state.variables
    settings = state.settings

    if not settings.enable_offline_transport:
        vs.update(calc_initial_conditions_surface_kernel(state))
        vs.update(calc_initial_conditions_root_zone_kernel(state))
        vs.update(calc_initial_conditions_subsoil_kernel(state))
        vs.update(calc_initial_conditions_soil_kernel(state))
        if not settings.enable_groundwater:
            vs.update(calc_initial_conditions_kernel(state))
        elif settings.enable_groundwater:
            vs.update(calc_initial_conditions_groundwater_kernel(state))
            vs.update(calc_initial_conditions_with_gw_kernel(state))
        if settings.enable_crop_phenology:
            vs.update(calc_initial_conditions_crops_kernel(state))
    elif settings.enable_offline_transport:
        vs.update(calc_initial_conditions_transport_kernel(state))


@roger_kernel
def calc_parameters_surface_kernel(state):
    vs = state.variables

    # land use dependent upper interception storage
    S_int_top_tot = allocate(state.dimensions, ("x", "y"))
    trees_cond = allocate(state.dimensions, ("x", "y"), dtype=bool, fill=False)
    trees_cond = update(
        trees_cond,
        at[:, :], npx.isin(vs.lu_id, npx.array([10, 11, 12, 15])),
    )

    def loop_body_S_int_top_tot(i, S_int_top_tot):
        mask = (vs.lu_id == i) & trees_cond
        row_no = _get_row_no(vs.lut_ilu[:, 0], i)
        S_int_top_tot = update(
            S_int_top_tot,
            at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.lut_ilu[row_no, vs.month[vs.tau]], S_int_top_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
        )

        return S_int_top_tot

    S_int_top_tot = for_loop(10, 16, loop_body_S_int_top_tot, S_int_top_tot)

    vs.S_int_top_tot = update(
        vs.S_int_top_tot,
        at[2:-2, 2:-2], S_int_top_tot[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2]
    )

    # land use dependent lower interception storage
    S_int_ground_tot = allocate(state.dimensions, ("x", "y"))

    ground_cond = allocate(state.dimensions, ("x", "y"), dtype=bool, fill=False)
    ground_cond = update(
        ground_cond,
        at[:, :], npx.isin(vs.lu_id, npx.array([0, 5, 6, 7, 8, 9, 13, 98, 31, 32, 33, 40, 41, 50, 98]))
    )

    def loop_body_S_int_ground_tot(i, S_int_ground_tot):
        mask = (vs.lu_id == i) & ground_cond
        row_no = _get_row_no(vs.lut_ilu[:, 0], i)
        S_int_ground_tot = update(
            S_int_ground_tot,
            at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.lut_ilu[row_no, vs.month[vs.tau]], S_int_ground_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
        )

        return S_int_ground_tot

    trees_ground_cond = allocate(state.dimensions, ("x", "y"), dtype=bool, fill=False)
    trees_ground_cond = update(
        trees_ground_cond,
        at[:, :], npx.isin(vs.lu_id, npx.array([10, 11, 12, 15]))
    )

    def loop_body_S_int_ground_tot_trees(i, S_int_ground_tot):
        mask = (vs.lu_id == i) & trees_ground_cond
        S_int_ground_tot = update(
            S_int_ground_tot,
            at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], 1, S_int_ground_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
        )

        return S_int_ground_tot

    S_int_ground_tot = for_loop(0, 51, loop_body_S_int_ground_tot, S_int_ground_tot)
    S_int_ground_tot = for_loop(10, 16, loop_body_S_int_ground_tot_trees, S_int_ground_tot)

    vs.S_int_ground_tot = update(
        vs.S_int_ground_tot,
        at[2:-2, 2:-2], S_int_ground_tot[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2]
    )

    # land use dependent ground cover (canopy cover)
    ground_cover = allocate(state.dimensions, ("x", "y"))

    cc_cond = allocate(state.dimensions, ("x", "y"), dtype=bool, fill=False)
    cc_cond = update(
        cc_cond,
        at[:, :], npx.isin(vs.lu_id, npx.array([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 98, 31, 32, 33, 40, 41, 50, 98]))
    )

    def loop_body_ground_cover(i, ground_cover):
        mask = (vs.lu_id == i) & cc_cond
        row_no = _get_row_no(vs.lut_gc[:, 0], i)
        ground_cover = update(
            ground_cover,
            at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.lut_gc[row_no, vs.month[vs.tau]], ground_cover[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
        )

        return ground_cover

    ground_cover = for_loop(0, 51, loop_body_ground_cover, ground_cover)

    vs.ground_cover = update(
        vs.ground_cover,
        at[2:-2, 2:-2, vs.tau], ground_cover[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2]
    )

    # land use dependent transpiration coeffcient
    basal_transp_coeff = allocate(state.dimensions, ("x", "y"))

    def loop_body_basal_transp_coeff(i, basal_transp_coeff):
        mask = (vs.lu_id == i) & cc_cond
        row_no = _get_row_no(vs.lut_gc[:, 0], i)
        basal_transp_coeff = update(
            basal_transp_coeff,
            at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.lut_gc[row_no, vs.month[vs.tau]] / vs.lut_gcm[row_no, 1], basal_transp_coeff[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
        )

        return basal_transp_coeff

    basal_transp_coeff = for_loop(0, 51, loop_body_basal_transp_coeff, basal_transp_coeff)

    basal_transp_coeff = update(
        basal_transp_coeff,
        at[2:-2, 2:-2], npx.where(vs.maskRiver[2:-2, 2:-2] | vs.maskLake[2:-2, 2:-2], 0, basal_transp_coeff[2:-2, 2:-2]),
    )

    vs.basal_transp_coeff = update(
        vs.basal_transp_coeff,
        at[2:-2, 2:-2], basal_transp_coeff[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2]
    )

    # land use dependent evaporation coeffcient
    basal_evap_coeff = allocate(state.dimensions, ("x", "y"))

    def loop_body_basal_evap_coeff(i, basal_evap_coeff):
        mask = (vs.lu_id == i) & cc_cond
        row_no = _get_row_no(vs.lut_gc[:, 0], i)
        basal_evap_coeff = update(
            basal_evap_coeff,
            at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], 1 - ((vs.lut_gc[row_no, vs.month[vs.tau]] / vs.lut_gcm[row_no, 1]) * vs.lut_gcm[row_no, 1]), basal_evap_coeff[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
        )

        return basal_evap_coeff

    basal_evap_coeff = for_loop(0, 51, loop_body_basal_evap_coeff, basal_evap_coeff)

    basal_evap_coeff = update(
        basal_evap_coeff,
        at[2:-2, 2:-2], npx.where(vs.maskRiver[2:-2, 2:-2] | vs.maskLake[2:-2, 2:-2], 1, basal_evap_coeff[2:-2, 2:-2]),
    )

    vs.basal_evap_coeff = update(
        vs.basal_evap_coeff,
        at[2:-2, 2:-2], basal_evap_coeff[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2]
    )

    # maximum snow interception storage
    vs.swe_top_tot = update(
        vs.swe_top_tot,
        at[2:-2, 2:-2], npx.where((vs.ta[2:-2, 2:-2, vs.tau] > -1) & (vs.lu_id[2:-2, 2:-2] == 10), 9, vs.swe_top_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_top_tot = update(
        vs.swe_top_tot,
        at[2:-2, 2:-2], npx.where((vs.ta[2:-2, 2:-2, vs.tau] > -1) & (vs.lu_id[2:-2, 2:-2] == 11), 15, vs.swe_top_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_top_tot = update(
        vs.swe_top_tot,
        at[2:-2, 2:-2], npx.where((vs.ta[2:-2, 2:-2, vs.tau] > -1) & (vs.lu_id[2:-2, 2:-2] == 12), 25, vs.swe_top_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_top_tot = update(
        vs.swe_top_tot,
        at[2:-2, 2:-2], npx.where((vs.ta[2:-2, 2:-2, vs.tau] >= -3) & (vs.ta[2:-2, 2:-2, vs.tau] <= -1) & (vs.lu_id[2:-2, 2:-2] == 10), 2.5 + 0.5 * vs.ta[2:-2, 2:-2, vs.tau] * 9, vs.swe_top_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_top_tot = update(
        vs.swe_top_tot,
        at[2:-2, 2:-2], npx.where((vs.ta[2:-2, 2:-2, vs.tau] >= -3) & (vs.ta[2:-2, 2:-2, vs.tau] <= -1) & (vs.lu_id[2:-2, 2:-2] == 11), 2.5 + 0.5 * vs.ta[2:-2, 2:-2, vs.tau] * 15, vs.swe_top_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_top_tot = update(
        vs.swe_top_tot,
        at[2:-2, 2:-2], npx.where((vs.ta[2:-2, 2:-2, vs.tau] >= -3) & (vs.ta[2:-2, 2:-2, vs.tau] <= -1) & (vs.lu_id[2:-2, 2:-2] == 12), 2.5 + 0.5 * vs.ta[2:-2, 2:-2, vs.tau] * 25, vs.swe_top_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_top_tot = update(
        vs.swe_top_tot,
        at[2:-2, 2:-2], npx.where((vs.ta[2:-2, 2:-2, vs.tau] < -3) & (vs.lu_id[2:-2, 2:-2] == 10), 18, vs.swe_top_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_top_tot = update(
        vs.swe_top_tot,
        at[2:-2, 2:-2], npx.where((vs.ta[2:-2, 2:-2, vs.tau] < -3) & (vs.lu_id[2:-2, 2:-2] == 11), 30, vs.swe_top_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_top_tot = update(
        vs.swe_top_tot,
        at[2:-2, 2:-2], npx.where((vs.ta[2:-2, 2:-2, vs.tau] < -3) & (vs.lu_id[2:-2, 2:-2] == 12), 50, vs.swe_top_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.lai = update(
        vs.lai,
        at[2:-2, 2:-2], npx.log(1 / (1 - vs.ground_cover[2:-2, 2:-2, vs.tau])) / npx.log(1 / 0.7) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.throughfall_coeff_top = update(
        vs.throughfall_coeff_top,
        at[2:-2, 2:-2], npx.where(npx.isin(vs.lu_id[2:-2, 2:-2], npx.array([10, 11, 12])), npx.where(vs.lai[2:-2, 2:-2] > 1, 0.1, 1.1 - vs.lai[2:-2, 2:-2]), 0) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.throughfall_coeff_ground = update(
        vs.throughfall_coeff_ground,
        at[2:-2, 2:-2], npx.where(npx.isin(vs.lu_id[2:-2, 2:-2], npx.arange(500, 598)), npx.where(vs.lai[2:-2, 2:-2] > 1, 0.1, 1.1 - vs.lai[2:-2, 2:-2]), 0) * vs.maskCatch[2:-2, 2:-2]
    )

    return KernelOutput(
        S_int_top_tot=vs.S_int_top_tot,
        S_int_ground_tot=vs.S_int_ground_tot,
        ground_cover=vs.ground_cover,
        basal_transp_coeff=vs.basal_transp_coeff,
        basal_evap_coeff=vs.basal_evap_coeff,
        swe_top_tot=vs.swe_top_tot,
        lai=vs.lai,
        throughfall_coeff_top=vs.throughfall_coeff_top,
        throughfall_coeff_ground=vs.throughfall_coeff_ground,
    )


def validate_parameters_surface(state):
    vs = state.variables

    mask1 = ((vs.slope > 1) | (vs.slope < 0))
    if global_and(npx.any(mask1)):
        raise ValueError('slope-parameter is out of range.')

    mask2 = ((vs.sealing > 1) | (vs.sealing < 0))
    if global_and(npx.any(mask2)):
        raise ValueError('sealing-parameter is out of range.')

    mask3 = ((vs.lu_id > 1000) | (vs.lu_id < 0))
    if global_and(npx.any(mask3)):
        raise ValueError('lu_id-parameter is out of range.')

    if global_and(npx.any(npx.isnan(vs.slope))):
        raise ValueError('slope-parameter contains non-numeric values.')

    if global_and(npx.any(npx.isnan(vs.sealing))):
        raise ValueError('sealing-parameter contains non-numeric values.')

    if global_and(npx.any(npx.isnan(vs.lu_id))):
        raise ValueError('lu_id-parameter contains non-numeric values.')


@roger_kernel
def calc_parameters_soil_kernel(state):
    vs = state.variables
    settings = state.settings

    vs.S_ac_s = update(
        vs.S_ac_s,
        at[2:-2, 2:-2], (vs.z_soil[2:-2, 2:-2] * vs.theta_ac[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_ufc_s = update(
        vs.S_ufc_s,
        at[2:-2, 2:-2], (vs.z_soil[2:-2, 2:-2] * vs.theta_ufc[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_pwp_s = update(
        vs.S_pwp_s,
        at[2:-2, 2:-2], (vs.z_soil[2:-2, 2:-2] * vs.theta_pwp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_fc_s = update(
        vs.S_fc_s,
        at[2:-2, 2:-2], (vs.z_soil[2:-2, 2:-2] * (vs.theta_ufc[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2])) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_sat_s = update(
        vs.S_sat_s,
        at[2:-2, 2:-2], (vs.z_soil[2:-2, 2:-2] * (vs.theta_ac[2:-2, 2:-2] + vs.theta_ufc[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2])) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.theta_sat = update(
        vs.theta_sat,
        at[2:-2, 2:-2], (vs.theta_ac[2:-2, 2:-2] + vs.theta_ufc[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.theta_fc = update(
        vs.theta_fc,
        at[2:-2, 2:-2], (vs.theta_ufc[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    # pore-size distribution index
    vs.lambda_bc = update(
        vs.lambda_bc,
        at[2:-2, 2:-2], ((npx.log(vs.theta_fc[2:-2, 2:-2]/vs.theta_sat[2:-2, 2:-2]) - npx.log(vs.theta_pwp[2:-2, 2:-2]/vs.theta_sat[2:-2, 2:-2])) / (npx.log(15850) - npx.log(63))) * vs.maskCatch[2:-2, 2:-2]
    )
    # air entry value (or bubbling pressure)
    vs.ha = update(
        vs.ha,
        at[2:-2, 2:-2], ((vs.theta_pwp[2:-2, 2:-2]/vs.theta_sat[2:-2, 2:-2])**(1./vs.lambda_bc[2:-2, 2:-2])*(-15850)) * vs.maskCatch[2:-2, 2:-2]
    )
    # pore-connectivity parameter
    vs.m_bc = update(
        vs.m_bc,
        at[2:-2, 2:-2], ((settings.a_bc + settings.b_bc * vs.lambda_bc[2:-2, 2:-2]) / vs.lambda_bc[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )
    # salvucci exponent
    vs.n_salv = update(
        vs.n_salv,
        at[2:-2, 2:-2], (settings.a_bc + settings.b_bc * vs.lambda_bc[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )
    # wetting front suction (in mm)
    # *(-10) for conversion to mm
    vs.wfs = update(
        vs.wfs,
        at[2:-2, 2:-2], (((2+3*vs.lambda_bc[2:-2, 2:-2])/(1+3*vs.lambda_bc[2:-2, 2:-2])*vs.ha[2:-2, 2:-2]/2) * (-10)) * vs.maskCatch[2:-2, 2:-2]
    )

    # soil water content at different soil water potentials
    vs.theta_27 = update(
        vs.theta_27,
        at[2:-2, 2:-2], ((vs.ha[2:-2, 2:-2]/(-(10**2.7)))**vs.lambda_bc[2:-2, 2:-2]*vs.theta_sat[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )
    vs.theta_4 = update(
        vs.theta_4,
        at[2:-2, 2:-2], ((vs.ha[2:-2, 2:-2]/(-(10**4)))**vs.lambda_bc[2:-2, 2:-2]*vs.theta_sat[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )
    vs.theta_6 = update(
        vs.theta_6,
        at[2:-2, 2:-2], ((vs.ha[2:-2, 2:-2]/(-(10**6)))**vs.lambda_bc[2:-2, 2:-2]*vs.theta_sat[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    # calculate sand content
    vs.sand = update(
        vs.sand,
        at[2:-2, 2:-2], (1 * (vs.theta_ac[2:-2, 2:-2] / 0.24)) * vs.maskCatch[2:-2, 2:-2]
    )
    vs.sand = update(
        vs.sand,
        at[2:-2, 2:-2], npx.where(vs.sand[2:-2, 2:-2] < 0, 0, vs.sand[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )
    vs.sand = update(
        vs.sand,
        at[2:-2, 2:-2], npx.where(vs.sand[2:-2, 2:-2] > 1, 1, vs.sand[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    # calculate clay content
    vs.clay = update(
        vs.clay,
        at[2:-2, 2:-2], (settings.clay_max * (vs.theta_6[2:-2, 2:-2] - settings.clay_min) / 0.3) * vs.maskCatch[2:-2, 2:-2]
    )
    vs.clay = update(
        vs.clay,
        at[2:-2, 2:-2], npx.where(vs.clay[2:-2, 2:-2] < settings.clay_min, settings.clay_min, vs.clay[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    # calculate maximum depth of shrinkage cracks (in mm) which depends on
    # soil clay content
    vs.z_sc_max = update(
        vs.z_sc_max,
        at[2:-2, 2:-2], (vs.clay[2:-2, 2:-2] * 700) * vs.maskCatch[2:-2, 2:-2]
    )

    # calculate drainage area of vertical macropores
    vs.mp_drain_area = update(
        vs.mp_drain_area,
        at[2:-2, 2:-2], 1 - npx.exp((-1) * (vs.dmpv[2:-2, 2:-2]/82)**0.887) * vs.maskCatch[2:-2, 2:-2]
    )

    return KernelOutput(
        S_ac_s=vs.S_ac_s,
        S_ufc_s=vs.S_ufc_s,
        S_pwp_s=vs.S_pwp_s,
        S_fc_s=vs.S_fc_s,
        S_sat_s=vs.S_sat_s,
        theta_sat=vs.theta_sat,
        theta_fc=vs.theta_fc,
        lambda_bc=vs.lambda_bc,
        ha=vs.ha,
        m_bc=vs.m_bc,
        n_salv=vs.n_salv,
        wfs=vs.wfs,
        theta_27=vs.theta_27,
        theta_4=vs.theta_4,
        theta_6=vs.theta_6,
        sand=vs.sand,
        clay=vs.clay,
        z_sc_max=vs.z_sc_max,
        mp_drain_area=vs.mp_drain_area,
        )


@roger_kernel
def calc_parameters_root_zone_kernel(state):
    vs = state.variables
    settings = state.settings

    # calculate readily evaporable water
    mask1 = (vs.theta_pwp < settings.theta_rew_min)
    mask2 = (vs.theta_pwp >= settings.theta_rew_min) & (vs.theta_pwp <= settings.theta_rew_max)
    mask3 = (vs.theta_pwp > settings.theta_rew_max)
    vs.rew = update(
        vs.rew,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], settings.rew_min, vs.rew[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )
    vs.rew = update(
        vs.rew,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], vs.theta_pwp[2:-2, 2:-2] / settings.theta_rew_max, vs.rew[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )
    vs.rew = update(
        vs.rew,
        at[2:-2, 2:-2], npx.where(mask3[2:-2, 2:-2], settings.rew_max, vs.rew[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    # evaporation depth
    vs.z_evap = update(
        vs.z_evap,
        at[2:-2, 2:-2], ((vs.rew[2:-2, 2:-2] / settings.rew_max) * settings.z_evap_max) * vs.maskCatch[2:-2, 2:-2]
    )

    # total evaporable water
    vs.tew = update(
        vs.tew,
        at[2:-2, 2:-2], ((vs.theta_fc[2:-2, 2:-2] - 0.5 * vs.theta_pwp[2:-2, 2:-2]) * vs.z_evap[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    # root depth
    z_root = allocate(state.dimensions, ("x", "y"))

    cc_cond = allocate(state.dimensions, ("x", "y"), dtype=bool, fill=False)
    cc_cond = update(
        cc_cond,
        at[:, :], npx.isin(vs.lu_id, npx.array([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 98, 31, 32, 33, 40, 41, 50, 98]))
    )

    # assign land use specific root depth
    def loop_body(i, z_root):
        mask = (vs.lu_id == i) & cc_cond
        row_no = _get_row_no(vs.lut_rdlu[:, 0], i)
        z_root = update(
            z_root,
            at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.lut_rdlu[row_no, 1], z_root[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
        )

        return z_root

    z_root = for_loop(0, 51, loop_body, z_root)

    z_root = update(
        z_root,
        at[2:-2, 2:-2], npx.where(vs.maskRiver[2:-2, 2:-2] | vs.maskLake[2:-2, 2:-2], 0, z_root[2:-2, 2:-2]),
    )

    z_root = update(
        z_root,
        at[2:-2, 2:-2], npx.where(((vs.lu_id[2:-2, 2:-2] == 10) | (vs.lu_id[2:-2, 2:-2] == 11) | (vs.lu_id[2:-2, 2:-2] == 12) | (vs.lu_id[2:-2, 2:-2] == 15) | (vs.lu_id[2:-2, 2:-2] == 16)), 1500, z_root[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    z_root = update(
        z_root,
        at[2:-2, 2:-2], npx.where((vs.lu_id[2:-2, 2:-2] == 100), 300, z_root[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    z_root = update(
        z_root,
        at[2:-2, 2:-2], npx.where((z_root[2:-2, 2:-2] >= vs.z_soil[2:-2, 2:-2]), 0.9 * vs.z_soil[2:-2, 2:-2], z_root[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.z_root = update(
        vs.z_root,
        at[2:-2, 2:-2, 0], z_root[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2]
    )

    vs.z_root = update(
        vs.z_root,
        at[2:-2, 2:-2, 1], z_root[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2]
    )

    mask_crops = npx.isin(vs.lu_id, npx.arange(500, 600, 1, dtype=int))
    vs.z_root = update(
        vs.z_root,
        at[2:-2, 2:-2, 0], npx.where(mask_crops[2:-2, 2:-2], vs.z_evap[2:-2, 2:-2], vs.z_root[2:-2, 2:-2, 0]) * vs.maskCatch[2:-2, 2:-2]
    )
    vs.z_root = update(
        vs.z_root,
        at[2:-2, 2:-2, 1], npx.where(mask_crops[2:-2, 2:-2], vs.z_evap[2:-2, 2:-2], vs.z_root[2:-2, 2:-2, 1]) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_ac_rz = update(
        vs.S_ac_rz,
        at[2:-2, 2:-2], (vs.theta_ac[2:-2, 2:-2] * vs.z_root[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_ufc_rz = update(
        vs.S_ufc_rz,
        at[2:-2, 2:-2], (vs.theta_ufc[2:-2, 2:-2] * vs.z_root[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_pwp_rz = update(
        vs.S_pwp_rz,
        at[2:-2, 2:-2], (vs.theta_pwp[2:-2, 2:-2] * vs.z_root[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_sat_rz = update(
        vs.S_sat_rz,
        at[2:-2, 2:-2], ((vs.theta_ac[2:-2, 2:-2] + vs.theta_ufc[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2]) * vs.z_root[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_fc_rz = update(
        vs.S_fc_rz,
        at[2:-2, 2:-2], ((vs.theta_ufc[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2]) * vs.z_root[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2]
    )

    return KernelOutput(
        z_root=vs.z_root,
        z_evap=vs.z_evap,
        rew=vs.rew,
        tew=vs.tew,
        S_ac_rz=vs.S_ac_rz,
        S_ufc_rz=vs.S_ufc_rz,
        S_pwp_rz=vs.S_pwp_rz,
        S_fc_rz=vs.S_fc_rz,
        S_sat_rz=vs.S_sat_rz,
        )


@roger_kernel
def calc_parameters_subsoil_kernel(state):
    vs = state.variables

    vs.S_ac_ss = update(
        vs.S_ac_ss,
        at[2:-2, 2:-2], (vs.theta_ac[2:-2, 2:-2] * (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau])) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_ufc_ss = update(
        vs.S_ufc_ss,
        at[2:-2, 2:-2], (vs.theta_ufc[2:-2, 2:-2] * (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau])) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_pwp_ss = update(
        vs.S_pwp_ss,
        at[2:-2, 2:-2], (vs.theta_pwp[2:-2, 2:-2] * (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau])) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_sat_ss = update(
        vs.S_sat_ss,
        at[2:-2, 2:-2], ((vs.theta_ac[2:-2, 2:-2] + vs.theta_ufc[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2]) * (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau])) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_fc_ss = update(
        vs.S_fc_ss,
        at[2:-2, 2:-2], ((vs.theta_ufc[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2]) * (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau])) * vs.maskCatch[2:-2, 2:-2]
    )

    return KernelOutput(
        S_ac_ss=vs.S_ac_ss,
        S_ufc_ss=vs.S_ufc_ss,
        S_pwp_ss=vs.S_pwp_ss,
        S_fc_ss=vs.S_fc_ss,
        S_sat_ss=vs.S_sat_ss,
        )


@roger_kernel
def calc_parameters_lateral_flow_kernel(state):
    vs = state.variables

    v_mp_layer = allocate(state.dimensions, ("x", "y", 8))

    # assign macropore flow velocity from look up table (in mm/h)
    def loop_body(i, v_mp_layer):
        nrow = _get_row_no(vs.lut_mlms[:, 0], i)
        # convert m/h to mm/h
        v_mp_layer = update(
            v_mp_layer,
            at[2:-2, 2:-2, 7], npx.where(vs.slope_per[2:-2, 2:-2] == i, vs.lut_mlms[nrow, 1]*1000, v_mp_layer[2:-2, 2:-2, 7]) * vs.maskCatch[2:-2, 2:-2]
        )
        v_mp_layer = update(
            v_mp_layer,
            at[2:-2, 2:-2, 6], npx.where(vs.slope_per[2:-2, 2:-2] == i, vs.lut_mlms[nrow, 2]*1000, v_mp_layer[2:-2, 2:-2, 6]) * vs.maskCatch[2:-2, 2:-2]
        )
        v_mp_layer = update(
            v_mp_layer,
            at[2:-2, 2:-2, 5], npx.where(vs.slope_per[2:-2, 2:-2] == i, vs.lut_mlms[nrow, 3]*1000, v_mp_layer[2:-2, 2:-2, 5]) * vs.maskCatch[2:-2, 2:-2]
        )
        v_mp_layer = update(
            v_mp_layer,
            at[2:-2, 2:-2, 4], npx.where(vs.slope_per[2:-2, 2:-2] == i, vs.lut_mlms[nrow, 4]*1000, v_mp_layer[2:-2, 2:-2, 4]) * vs.maskCatch[2:-2, 2:-2]
        )
        v_mp_layer = update(
            v_mp_layer,
            at[2:-2, 2:-2, 3], npx.where(vs.slope_per[2:-2, 2:-2] == i, vs.lut_mlms[nrow, 5]*1000, v_mp_layer[2:-2, 2:-2, 3]) * vs.maskCatch[2:-2, 2:-2]
        )
        v_mp_layer = update(
            v_mp_layer,
            at[2:-2, 2:-2, 2], npx.where(vs.slope_per[2:-2, 2:-2] == i, vs.lut_mlms[nrow, 6]*1000, v_mp_layer[2:-2, 2:-2, 2]) * vs.maskCatch[2:-2, 2:-2]
        )
        v_mp_layer = update(
            v_mp_layer,
            at[2:-2, 2:-2, 1], npx.where(vs.slope_per[2:-2, 2:-2] == i, vs.lut_mlms[nrow, 7]*1000, v_mp_layer[2:-2, 2:-2, 1]) * vs.maskCatch[2:-2, 2:-2]
        )
        v_mp_layer = update(
            v_mp_layer,
            at[2:-2, 2:-2, 0], npx.where(vs.slope_per[2:-2, 2:-2] == i, vs.lut_mlms[nrow, 8]*1000, v_mp_layer[2:-2, 2:-2, 0]) * vs.maskCatch[2:-2, 2:-2]
        )

        return v_mp_layer

    v_mp_layer = for_loop(1, npx.max(vs.slope_per)+1, loop_body, v_mp_layer)

    vs.v_mp_layer_8 = update(
        vs.v_mp_layer_8,
        at[2:-2, 2:-2], v_mp_layer[2:-2, 2:-2, 7] * vs.maskCatch[2:-2, 2:-2]
    )
    vs.v_mp_layer_7 = update(
        vs.v_mp_layer_7,
        at[2:-2, 2:-2], v_mp_layer[2:-2, 2:-2, 6] * vs.maskCatch[2:-2, 2:-2]
    )
    vs.v_mp_layer_6 = update(
        vs.v_mp_layer_6,
        at[2:-2, 2:-2], v_mp_layer[2:-2, 2:-2, 5] * vs.maskCatch[2:-2, 2:-2]
    )
    vs.v_mp_layer_5 = update(
        vs.v_mp_layer_5,
        at[2:-2, 2:-2], v_mp_layer[2:-2, 2:-2, 4] * vs.maskCatch[2:-2, 2:-2]
    )
    vs.v_mp_layer_4 = update(
        vs.v_mp_layer_4,
        at[2:-2, 2:-2], v_mp_layer[2:-2, 2:-2, 3] * vs.maskCatch[2:-2, 2:-2]
    )
    vs.v_mp_layer_3 = update(
        vs.v_mp_layer_3,
        at[2:-2, 2:-2], v_mp_layer[2:-2, 2:-2, 2] * vs.maskCatch[2:-2, 2:-2]
    )
    vs.v_mp_layer_2 = update(
        vs.v_mp_layer_2,
        at[2:-2, 2:-2], v_mp_layer[2:-2, 2:-2, 1] * vs.maskCatch[2:-2, 2:-2]
    )
    vs.v_mp_layer_1 = update(
        vs.v_mp_layer_1,
        at[2:-2, 2:-2], v_mp_layer[2:-2, 2:-2, 0] * vs.maskCatch[2:-2, 2:-2]
    )

    return KernelOutput(
        v_mp_layer_8=vs.v_mp_layer_8,
        v_mp_layer_7=vs.v_mp_layer_7,
        v_mp_layer_6=vs.v_mp_layer_6,
        v_mp_layer_5=vs.v_mp_layer_5,
        v_mp_layer_4=vs.v_mp_layer_4,
        v_mp_layer_3=vs.v_mp_layer_3,
        v_mp_layer_2=vs.v_mp_layer_2,
        v_mp_layer_1=vs.v_mp_layer_1,
    )


@roger_kernel
def calc_parameters_crops_kernel(state):
    vs = state.variables

    for i in range(500, 600):
        mask = (vs.crop_type == i)
        row_no = _get_row_no(vs.lut_crops[:, 0], i)
        vs.doy_start = update(
            vs.doy_start,
            at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 1], vs.doy_start[2:-2, 2:-2, :]),
        )
        vs.doy_mid = update(
            vs.doy_mid,
            at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 2], vs.doy_mid[2:-2, 2:-2, :]),
        )
        vs.doy_dec = update(
            vs.doy_dec,
            at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 3], vs.doy_dec[2:-2, 2:-2, :]),
        )
        vs.doy_end = update(
            vs.doy_end,
            at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 4], vs.doy_end[2:-2, 2:-2, :]),
        )
        vs.ta_base = update(
            vs.ta_base,
            at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 11], vs.ta_base[2:-2, 2:-2, :]),
        )
        vs.ta_ceil = update(
            vs.ta_ceil,
            at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 12], vs.ta_ceil[2:-2, 2:-2, :]),
        )
        vs.ccc_min = update(
            vs.ccc_min,
            at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 13], vs.ccc_min[2:-2, 2:-2, :]),
        )
        vs.ccc_max = update(
            vs.ccc_max,
            at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 14], vs.ccc_max[2:-2, 2:-2, :]),
        )
        vs.crop_height_max = update(
            vs.crop_height_max,
            at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 16] * 1000, vs.crop_height_max[2:-2, 2:-2, :]),
        )
        vs.ccc_growth_rate = update(
            vs.ccc_growth_rate,
            at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 18], vs.ccc_growth_rate[2:-2, 2:-2, :]),
        )
        vs.basal_crop_coeff_mid = update(
            vs.basal_crop_coeff_mid,
            at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 21] * vs.lut_crop_scale[2:-2, 2:-2, row_no], vs.basal_crop_coeff_mid[2:-2, 2:-2, :]),
        )
        vs.z_root_crop_max = update(
            vs.z_root_crop_max,
            at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 15] * 1000, vs.z_root_crop_max[2:-2, 2:-2, :]),
        )
        vs.root_growth_rate = update(
            vs.root_growth_rate,
            at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 19], vs.root_growth_rate[2:-2, 2:-2, :]),
        )
        vs.water_stress_coeff_crop = update(
            vs.water_stress_coeff_crop,
            at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 20], vs.water_stress_coeff_crop[2:-2, 2:-2, :]),
        )

    vs.theta_water_stress_crop = update(
        vs.theta_water_stress_crop,
        at[2:-2, 2:-2, :], (1 - vs.water_stress_coeff_crop[2:-2, 2:-2]) * (vs.theta_fc[2:-2, 2:-2, npx.newaxis] - vs.theta_pwp[2:-2, 2:-2, npx.newaxis]) + vs.theta_pwp[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(
        doy_start=vs.doy_start,
        doy_mid=vs.doy_mid,
        doy_dec=vs.doy_dec,
        doy_end=vs.doy_end,
        ta_base=vs.ta_base,
        ta_ceil=vs.ta_ceil,
        ccc_min=vs.ccc_min,
        ccc_max=vs.ccc_max,
        crop_height_max=vs.crop_height_max,
        ccc_growth_rate=vs.ccc_growth_rate,
        basal_crop_coeff_mid=vs.basal_crop_coeff_mid,
        z_root_crop_max=vs.z_root_crop_max,
        root_growth_rate=vs.root_growth_rate,
        water_stress_coeff_crop=vs.water_stress_coeff_crop,
        theta_water_stress_crop=vs.theta_water_stress_crop,
        )


@roger_kernel
def calc_parameters_transport_kernel(state):
    pass


def validate_parameters_transport(state):
    pass


def calc_parameters_groundwater_kernel(state):
    pass


def validate_parameters_groundwater(state):
    pass


def validate_parameters_soil(state):
    vs = state.variables
    settings = state.settings

    mask1 = (vs.z_soil > 0) & ((vs.theta_pwp + vs.theta_ufc + vs.theta_ac > 0.99) | (vs.theta_pwp + vs.theta_ufc + vs.theta_ac < 0.01))
    if global_and(npx.any(mask1[2:-2, 2:-2])):
        raise ValueError('theta-parameters are out of range.')

    mask2 = (vs.z_soil > 0) & ((vs.ks > 10000) | (vs.ks < 0))
    if global_and(npx.any(mask2[2:-2, 2:-2])):
        raise ValueError('ks-parameter is out of range.')

    mask3 = (vs.z_soil > 0) & ((vs.lmpv > vs.z_soil) | (vs.lmpv < 0))
    if global_and(npx.any(mask3[2:-2, 2:-2])):
        raise ValueError('lmpv-parameter is out of range.')

    if global_and(npx.any(npx.isnan(vs.z_soil[2:-2, 2:-2]))):
        raise ValueError('z_soil-parameter contains non-numeric values.')

    if global_and(npx.any(npx.isnan(vs.dmpv[2:-2, 2:-2]))):
        raise ValueError('dmpv-parameter contains non-numeric values.')

    if global_and(npx.any(npx.isnan(vs.lmpv[2:-2, 2:-2]))):
        raise ValueError('lmpv-parameter contains non-numeric values.')

    if global_and(npx.any(npx.isnan(vs.theta_pwp[2:-2, 2:-2]))):
        raise ValueError('theta_pwp-parameter contains non-numeric values.')

    if global_and(npx.any(npx.isnan(vs.theta_ufc[2:-2, 2:-2]))):
        raise ValueError('theta_ufc-parameter contains non-numeric values.')

    if global_and(npx.any(npx.isnan(vs.theta_ac[2:-2, 2:-2]))):
        raise ValueError('theta_ac-parameter contains non-numeric values.')

    if global_and(npx.any(npx.isnan(vs.ks[2:-2, 2:-2]))):
        raise ValueError('ks-parameter contains non-numeric values.')

    if global_and(npx.any(npx.isnan(vs.kf[2:-2, 2:-2]))):
        raise ValueError('kf-parameter contains non-numeric values.')

    if settings.enable_lateral_flow:
        if global_and(npx.any(npx.isnan(vs.dmph[2:-2, 2:-2]))):
            raise ValueError('dmph-parameter contains non-numeric values.')


@roger_routine
def calc_parameters(state):
    vs = state.variables
    settings = state.settings

    if not settings.enable_offline_transport:
        validate_parameters_surface(state)
        validate_parameters_soil(state)
        vs.update(calc_parameters_surface_kernel(state))
        vs.update(calc_parameters_soil_kernel(state))
        vs.update(calc_parameters_root_zone_kernel(state))
        vs.update(calc_parameters_subsoil_kernel(state))
        if settings.enable_lateral_flow:
            vs.update(calc_parameters_lateral_flow_kernel(state))
        if settings.enable_groundwater:
            vs.update(calc_parameters_groundwater_kernel(state))
        if settings.enable_crop_phenology:
            vs.update(calc_parameters_crops_kernel(state))
    elif settings.enable_offline_transport:
        vs.update(calc_parameters_transport_kernel(state))


@roger_kernel
def calc_storage_kernel(state):
    vs = state.variables
    settings = state.settings

    if not settings.enable_film_flow:
        vs.S = update(
            vs.S,
            at[2:-2, 2:-2, vs.tau], vs.S_sur[2:-2, 2:-2, vs.tau] + vs.S_s[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2]
        )

    elif settings.enable_film_flow:
        vs.S = update(
            vs.S,
            at[2:-2, 2:-2, vs.tau], vs.S_sur[2:-2, 2:-2, vs.tau] + vs.S_s[2:-2, 2:-2, vs.tau] + npx.sum(vs.S_f[2:-2, 2:-2, :], axis=-1) * vs.maskCatch[2:-2, 2:-2]
        )

    vs.dS = update(
        vs.dS,
        at[2:-2, 2:-2], vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1] * vs.maskCatch[2:-2, 2:-2]
    )

    return KernelOutput(
        S=vs.S,
        dS=vs.dS,
        )


@roger_kernel
def calc_storage_with_gw_kernel(state):
    vs = state.variables

    vs.S = update(
        vs.S,
        at[2:-2, 2:-2, vs.tau], vs.S_sur[2:-2, 2:-2, vs.tau] + vs.S_s[2:-2, 2:-2, vs.tau] + vs.S_vad[2:-2, 2:-2, vs.tau] + vs.S_gw[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2]
    )

    vs.dS = update(
        vs.dS,
        at[2:-2, 2:-2], vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1] * vs.maskCatch[2:-2, 2:-2]
    )

    return KernelOutput(
        S=vs.S,
        dS=vs.dS,
        )


@roger_routine
def calc_storage(state):
    vs = state.variables
    settings = state.settings

    if not settings.enable_groundwater:
        vs.update(calc_storage_kernel(state))
    elif settings.enable_groundwater:
        vs.update(calc_storage_with_gw_kernel(state))


@roger_kernel
def calc_dS_num_error(state):
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_groundwater_boundary | settings.enable_crop_phenology):
        vs.dS_num_error = update(
            vs.dS_num_error,
            at[2:-2, 2:-2], npx.abs((npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1)) - (vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2] -
            npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)) * settings.h)
        )
    return KernelOutput(
        dS_num_error=vs.dS_num_error,
        )


@roger_kernel
def calc_dC_num_error(state):
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and (settings.enable_deuterium or settings.enable_oxygen18):
        vs.dC_num_error = update(
            vs.dC_num_error,
            at[2:-2, 2:-2], npx.abs((npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.tau] - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.taum1]) - (vs.inf_mat_rz[2:-2, 2:-2] * npx.where(npx.isnan(vs.C_inf_mat_rz[2:-2, 2:-2]), 0, vs.C_inf_mat_rz[2:-2, 2:-2]) + vs.inf_pf_rz[2:-2, 2:-2] * npx.where(npx.isnan(vs.C_inf_pf_rz[2:-2, 2:-2]), 0, vs.C_inf_pf_rz[2:-2, 2:-2]) + vs.inf_pf_ss[2:-2, 2:-2] * npx.where(npx.isnan(vs.C_inf_pf_ss[2:-2, 2:-2]), 0, vs.C_inf_pf_ss[2:-2, 2:-2]) - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_evap_soil[2:-2, 2:-2]), 0, vs.C_evap_soil[2:-2, 2:-2]) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_transp[2:-2, 2:-2]), 0, vs.C_transp[2:-2, 2:-2]) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_q_ss[2:-2, 2:-2]), 0, vs.C_q_ss[2:-2, 2:-2])) * settings.h)
        )

    return KernelOutput(
        dC_num_error=vs.dC_num_error,
        )


@roger_routine
def calculate_num_error(state):
    vs = state.variables
    settings = state.settings

    if not settings.enable_offline_transport:
        vs.update(calc_dS_num_error(state))
    elif settings.enable_offline_transport:
        vs.update(calc_dS_num_error(state))
        vs.update(calc_dC_num_error(state))


@roger_kernel
def sanity_check(state):
    """
    Checks for mass conservation
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_lateral_flow and not settings.enable_groundwater_boundary and not settings.enable_groundwater and not settings.enable_offline_transport:
        check1 = global_and(npx.all(npx.isclose(vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1], vs.prec[2:-2, 2:-2, vs.tau] - vs.q_sur[2:-2, 2:-2] - vs.aet[2:-2, 2:-2] - vs.q_ss[2:-2, 2:-2] - vs.q_sub[2:-2, 2:-2], atol=settings.atol, rtol=settings.rtol)))
        check2 = global_and(npx.all((vs.S_fp_rz[2:-2, 2:-2] > -settings.atol) & (vs.S_lp_rz[2:-2, 2:-2] > -settings.atol) & (vs.S_fp_ss[2:-2, 2:-2] > -settings.atol) & (vs.S_lp_ss[2:-2, 2:-2] > -settings.atol)))
        check3 = global_and(npx.all((vs.S_fp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ufc_rz[2:-2, 2:-2]) & (vs.S_lp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ac_rz[2:-2, 2:-2]) & (vs.S_fp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ufc_ss[2:-2, 2:-2]) & (vs.S_lp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ac_ss[2:-2, 2:-2])))
        check = check1 & check2 & check3

    elif settings.enable_lateral_flow and settings.enable_groundwater_boundary and not settings.enable_offline_transport:
        check1 = global_and(npx.all(npx.isclose(vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1], vs.prec[2:-2, 2:-2, vs.tau] - vs.q_sur[2:-2, 2:-2] - vs.aet[2:-2, 2:-2] - vs.q_ss[2:-2, 2:-2] - vs.q_sub[2:-2, 2:-2] + vs.cpr_ss[2:-2, 2:-2], atol=settings.atol, rtol=settings.rtol)))
        check2 = global_and(npx.all((vs.S_fp_rz[2:-2, 2:-2] > -settings.atol) & (vs.S_lp_rz[2:-2, 2:-2] > -settings.atol) & (vs.S_fp_ss[2:-2, 2:-2] > -settings.atol) & (vs.S_lp_ss[2:-2, 2:-2] > -settings.atol)))
        check3 = global_and(npx.all((vs.S_fp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ufc_rz[2:-2, 2:-2]) & (vs.S_lp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ac_rz[2:-2, 2:-2]) & (vs.S_fp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ufc_ss[2:-2, 2:-2]) & (vs.S_lp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ac_ss[2:-2, 2:-2])))
        check = check1 & check2 & check3

    elif settings.enable_lateral_flow and settings.enable_groundwater and not settings.enable_offline_transport:
        check1 = global_and(npx.all(npx.isclose(vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1], vs.prec[2:-2, 2:-2, vs.tau] - vs.q_sur[2:-2, 2:-2] - vs.aet[2:-2, 2:-2] - vs.q_sub[2:-2, 2:-2] - vs.q_gw[2:-2, 2:-2] - vs.q_leak[2:-2, 2:-2], atol=settings.atol, rtol=settings.rtol)))
        check2 = global_and(npx.all((vs.S_fp_rz[2:-2, 2:-2] > -settings.atol) & (vs.S_lp_rz[2:-2, 2:-2] > -settings.atol) & (vs.S_fp_ss[2:-2, 2:-2] > -settings.atol) & (vs.S_lp_ss[2:-2, 2:-2] > -settings.atol)))
        check3 = global_and(npx.all((vs.S_fp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ufc_rz[2:-2, 2:-2]) & (vs.S_lp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ac_rz[2:-2, 2:-2]) & (vs.S_fp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ufc_ss[2:-2, 2:-2]) & (vs.S_lp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ac_ss[2:-2, 2:-2])))
        check = check1 & check2 & check3

    elif settings.enable_groundwater_boundary and not settings.enable_offline_transport:
        check1 = global_and(npx.all(npx.isclose(vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1], vs.prec[2:-2, 2:-2, vs.tau] - vs.q_sur[2:-2, 2:-2] - vs.aet[2:-2, 2:-2] - vs.q_ss[2:-2, 2:-2] + vs.cpr_ss[2:-2, 2:-2], atol=settings.atol, rtol=settings.rtol)))
        check2 = global_and(npx.all((vs.S_fp_rz[2:-2, 2:-2] > -settings.atol) & (vs.S_lp_rz[2:-2, 2:-2] > -settings.atol) & (vs.S_fp_ss[2:-2, 2:-2] > -settings.atol) & (vs.S_lp_ss[2:-2, 2:-2] > -settings.atol)))
        check3 = global_and(npx.all((vs.S_fp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ufc_rz[2:-2, 2:-2]) & (vs.S_lp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ac_rz[2:-2, 2:-2]) & (vs.S_fp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ufc_ss[2:-2, 2:-2]) & (vs.S_lp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ac_ss[2:-2, 2:-2])))
        check = check1 & check2 & check3

    elif settings.enable_film_flow and not settings.enable_offline_transport:
        check1 = global_and(npx.all(npx.isclose(vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1], vs.prec[2:-2, 2:-2, vs.tau] - vs.q_sur[2:-2, 2:-2] - vs.aet[2:-2, 2:-2] - vs.q_ss[2:-2, 2:-2] - vs.ff_drain[2:-2, 2:-2], atol=settings.atol, rtol=settings.rtol)))
        check2 = global_and(npx.all((vs.S_fp_rz[2:-2, 2:-2] > -settings.atol) & (vs.S_lp_rz[2:-2, 2:-2] > -settings.atol) & (vs.S_fp_ss[2:-2, 2:-2] > -settings.atol) & (vs.S_lp_ss[2:-2, 2:-2] > -settings.atol)))
        check3 = global_and(npx.all((vs.S_fp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ufc_rz[2:-2, 2:-2]) & (vs.S_lp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ac_rz[2:-2, 2:-2]) & (vs.S_fp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ufc_ss[2:-2, 2:-2]) & (vs.S_lp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ac_ss[2:-2, 2:-2])))
        check = check1 & check2 & check3

    elif not settings.enable_lateral_flow and not settings.enable_groundwater_boundary and not settings.enable_groundwater and not settings.enable_offline_transport:
        check1 = global_and(npx.all(npx.isclose(vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1], vs.prec[2:-2, 2:-2, vs.tau] - vs.q_sur[2:-2, 2:-2] - vs.aet[2:-2, 2:-2] - vs.q_ss[2:-2, 2:-2], atol=settings.atol, rtol=settings.rtol)))
        check2 = global_and(npx.all((vs.S_fp_rz[2:-2, 2:-2] > -settings.atol) & (vs.S_lp_rz[2:-2, 2:-2] > -settings.atol) & (vs.S_fp_ss[2:-2, 2:-2] > -settings.atol) & (vs.S_lp_ss[2:-2, 2:-2] > -settings.atol)))
        check3 = global_and(npx.all((vs.S_fp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ufc_rz[2:-2, 2:-2]) & (vs.S_lp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ac_rz[2:-2, 2:-2]) & (vs.S_fp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ufc_ss[2:-2, 2:-2]) & (vs.S_lp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ac_ss[2:-2, 2:-2])))
        check = check1 & check2 & check3

    elif settings.enable_offline_transport and not (settings.enable_deuterium or settings.enable_oxygen18 or settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate):
        check = global_and(npx.all(npx.isclose(npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1),
                                               (vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2] -
                                               npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)) * settings.h, atol=settings.atol, rtol=settings.rtol)))

    elif settings.enable_offline_transport and (settings.enable_deuterium or settings.enable_oxygen18):
        check1 = global_and(npx.all(npx.isclose(npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1),
                                                (vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2] -
                                                npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)) * settings.h, atol=settings.atol, rtol=settings.rtol)))
        check2 = global_and(npx.all(npx.isclose(npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.tau] - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.taum1],
                                                (vs.inf_mat_rz[2:-2, 2:-2] * npx.where(npx.isnan(vs.C_inf_mat_rz[2:-2, 2:-2]), 0, vs.C_inf_mat_rz[2:-2, 2:-2]) + vs.inf_pf_rz[2:-2, 2:-2] * npx.where(npx.isnan(vs.C_inf_pf_rz[2:-2, 2:-2]), 0, vs.C_inf_pf_rz[2:-2, 2:-2]) + vs.inf_pf_ss[2:-2, 2:-2] * npx.where(npx.isnan(vs.C_inf_pf_ss[2:-2, 2:-2]), 0, vs.C_inf_pf_ss[2:-2, 2:-2]) -
                                                npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_evap_soil[2:-2, 2:-2]), 0, vs.C_evap_soil[2:-2, 2:-2]) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_transp[2:-2, 2:-2]), 0, vs.C_transp[2:-2, 2:-2]) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_q_ss[2:-2, 2:-2]), 0, vs.C_q_ss[2:-2, 2:-2])) * settings.h, atol=settings.atol, rtol=settings.rtol)))
        check3 = global_and(npx.all((npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) <= (vs.S_sat_rz[2:-2, 2:-2] + vs.S_sat_ss[2:-2, 2:-2]) - (vs.S_pwp_rz[2:-2, 2:-2] + vs.S_pwp_ss[2:-2, 2:-2])) & (npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) >= 0)))
        check4 = global_and(npx.all((npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1) <= vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2]) & (npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1) >= 0)))
        check5 = global_and(npx.all((npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1) <= vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2]) & (npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1) >= 0)))

        check = check1 & check2 & check3 & check4 & check5

        if rs.loglevel == 'warning' and rs.backend == 'numpy' and not check:
            check11 = npx.isclose(npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1),
                                                    vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2] -
                                                    npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2), atol=settings.atol, rtol=settings.rtol)
            check22 = npx.isclose(npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.tau] - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.taum1],
                                  vs.inf_mat_rz[2:-2, 2:-2] * npx.where(npx.isnan(vs.C_inf_mat_rz[2:-2, 2:-2]), 0, vs.C_inf_mat_rz[2:-2, 2:-2]) + vs.inf_pf_rz[2:-2, 2:-2] * npx.where(npx.isnan(vs.C_inf_pf_rz[2:-2, 2:-2]), 0, vs.C_inf_pf_rz[2:-2, 2:-2]) + vs.inf_pf_ss[2:-2, 2:-2] * npx.where(npx.isnan(vs.C_inf_pf_ss[2:-2, 2:-2]), 0, vs.C_inf_pf_ss[2:-2, 2:-2]) -
                                  npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_evap_soil[2:-2, 2:-2]), 0, vs.C_evap_soil[2:-2, 2:-2]) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_transp[2:-2, 2:-2]), 0, vs.C_transp[2:-2, 2:-2]) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_q_ss[2:-2, 2:-2]), 0, vs.C_q_ss[2:-2, 2:-2]), atol=settings.atol, rtol=settings.rtol)
            check33 = (npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) <= (vs.S_sat_rz[2:-2, 2:-2] + vs.S_sat_ss[2:-2, 2:-2]) - (vs.S_pwp_rz[2:-2, 2:-2] + vs.S_pwp_ss[2:-2, 2:-2])) & (npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) >= 0)
            check44 = (vs.sa_rz[2:-2, 2:-2, vs.tau, :] >= 0)
            check55 = (vs.sa_ss[2:-2, 2:-2, vs.tau, :] >= 0)

            if not check11.all():
                logger.warning(f"Water balance diverged at iteration {vs.itt}")
                rows11 = npx.where(check11 == False)[0].tolist()
                cols11 = npx.where(check11 == False)[1].tolist()
                rowscols11 = tuple(zip(rows11, cols11))
                if rowscols11:
                    logger.warning(f"Water balance diverged at {rowscols11}")
            if not check22.all():
                logger.warning(f"Solute balance diverged at iteration {vs.itt}")
                rows22 = npx.where(check22 == False)[0].tolist()
                cols22 = npx.where(check22 == False)[1].tolist()
                rowscols22 = tuple(zip(rows22, cols22))
                if rowscols22:
                    logger.debug(f"Solute balance diverged at {rowscols22}")
            if not check33.all():
                logger.warning(f"StorAge is out of bounds at iteration {vs.itt}")
                rows33 = npx.where(check33 == False)[0].tolist()
                cols33 = npx.where(check33 == False)[1].tolist()
                rowscols33 = tuple(zip(rows33, cols33))
                if rowscols33:
                    logger.warning(f"Solute balance diverged at {rowscols33}")
            if not check44.all():
                logger.warning(f"Root zone StorAge is out of bounds at iteration {vs.itt}")
                rows44 = npx.where(npx.any(check44 == False, axis=-1))[0].tolist()
                cols44 = npx.where(npx.any(check44 == False, axis=-1))[1].tolist()
                rowscols44 = tuple(zip(rows44, cols44))
                if rowscols44:
                    logger.warning(f"Root zone StorAge is out of bounds at at {rowscols44}")
            if not check55.all():
                logger.warning(f"Root zone StorAge is out of bounds at iteration {vs.itt}")
                rows55 = npx.where(npx.any(check55 == False, axis=-1))[0].tolist()
                cols55 = npx.where(npx.any(check55 == False, axis=-1))[1].tolist()
                rowscols55 = tuple(zip(rows55, cols55))
                if rowscols55:
                    logger.warning(f"Root zone StorAge is out of bounds at at {rowscols55}")

    elif settings.enable_offline_transport and (settings.enable_bromide or settings.enable_chloride):
        check1 = global_and(npx.all(npx.isclose(npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1),
                                                vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2] -
                                                npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2), atol=settings.atol, rtol=settings.rtol)))
        check2 = global_and(npx.all(npx.isclose(npx.sum(vs.msa_s[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.msa_s[2:-2, 2:-2, vs.taum1, :], axis=-1),
                                                vs.inf_mat_rz[2:-2, 2:-2] * vs.C_inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] * vs.C_inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2] * vs.C_inf_pf_ss[2:-2, 2:-2] -
                                                npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) * vs.C_transp[2:-2, 2:-2] - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2) * vs.C_q_ss[2:-2, 2:-2], atol=settings.atol, rtol=settings.rtol)))
        check3 = global_and(npx.all((npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) <= (vs.S_sat_rz[2:-2, 2:-2] + vs.S_sat_ss[2:-2, 2:-2]) - (vs.S_pwp_rz[2:-2, 2:-2] + vs.S_pwp_ss[2:-2, 2:-2])) & (npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) >= 0)))

        check = check1 & check2 & check3

    elif settings.enable_offline_transport and settings.enable_nitrate:
        check1 = global_and(npx.all(npx.isclose(npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1),
                                                vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2] -
                                                npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2), atol=settings.atol, rtol=settings.rtol)))
        check2 = global_and(npx.all(npx.isclose(npx.sum(vs.msa_s[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.msa_s[2:-2, 2:-2, vs.taum1, :], axis=-1),
                                                vs.inf_mat_rz[2:-2, 2:-2] * vs.C_inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] * vs.C_inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2] * vs.C_inf_pf_ss[2:-2, 2:-2] + npx.sum(vs.ma_s[2:-2, 2:-2, :], axis=2) -
                                                npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) * vs.C_transp[2:-2, 2:-2] - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2) * vs.C_q_ss[2:-2, 2:-2] - npx.sum(vs.mr_s[2:-2, 2:-2, :], axis=2), atol=settings.atol, rtol=settings.rtol)))
        check3 = global_and(npx.all((npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) <= (vs.S_sat_rz[2:-2, 2:-2] + vs.S_sat_ss[2:-2, 2:-2]) - (vs.S_pwp_rz[2:-2, 2:-2] + vs.S_pwp_ss[2:-2, 2:-2])) & (npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) >= 0)))

        check = check1 & check2 & check3

    elif settings.enable_offline_transport and settings.enable_groundwater_boundary:
        check = global_and(npx.all(npx.isclose(npx.sum(vs.SA_s[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.SA_s[2:-2, 2:-2, vs.tau, 1, :], axis=-1), vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2] - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2) + npx.sum(vs.cpr_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_ss[2:-2, 2:-2, :], axis=2), atol=settings.atol, rtol=settings.rtol)))

    return check
