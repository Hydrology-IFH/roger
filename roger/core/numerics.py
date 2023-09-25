from roger import roger_kernel, roger_routine, KernelOutput
from roger.distributed import global_and
from roger.core.operators import numpy as npx, update, at
from roger import runtime_settings as rs, logger


def validate_parameters_surface(state):
    vs = state.variables

    mask1 = (vs.slope > 1) | (vs.slope < 0)
    if global_and(npx.any(mask1)):
        raise ValueError("slope-parameter is out of range.")

    mask2 = (vs.sealing > 1) | (vs.sealing < 0)
    if global_and(npx.any(mask2)):
        raise ValueError("sealing-parameter is out of range.")

    mask3 = (vs.lu_id > 1000) | (vs.lu_id < 0)
    if global_and(npx.any(mask3)):
        raise ValueError("lu_id-parameter is out of range.")

    if global_and(npx.any(npx.isnan(vs.slope))):
        raise ValueError("slope-parameter contains non-numeric values.")

    if global_and(npx.any(npx.isnan(vs.sealing))):
        raise ValueError("sealing-parameter contains non-numeric values.")

    if global_and(npx.any(npx.isnan(vs.lu_id))):
        raise ValueError("lu_id-parameter contains non-numeric values.")


def validate_parameters_soil(state):
    vs = state.variables
    settings = state.settings

    mask1 = (vs.z_soil > 0) & (
        (vs.theta_pwp + vs.theta_ufc + vs.theta_ac > 0.99) | (vs.theta_pwp + vs.theta_ufc + vs.theta_ac < 0.01)
    )
    if global_and(npx.any(mask1[2:-2, 2:-2])):
        raise ValueError("theta-parameters are out of range.")

    mask2 = (vs.z_soil > 0) & ((vs.ks > 10000) | (vs.ks < 0))
    if global_and(npx.any(mask2[2:-2, 2:-2])):
        raise ValueError("ks-parameter is out of range.")

    mask3 = (vs.z_soil > 0) & ((vs.lmpv > vs.z_soil) | (vs.lmpv < 0))
    if global_and(npx.any(mask3[2:-2, 2:-2])):
        raise ValueError("lmpv-parameter is out of range.")

    if global_and(npx.any(npx.isnan(vs.z_soil[2:-2, 2:-2]))):
        raise ValueError("z_soil-parameter contains non-numeric values.")

    if global_and(npx.any(npx.isnan(vs.dmpv[2:-2, 2:-2]))):
        raise ValueError("dmpv-parameter contains non-numeric values.")

    if global_and(npx.any(npx.isnan(vs.lmpv[2:-2, 2:-2]))):
        raise ValueError("lmpv-parameter contains non-numeric values.")

    if global_and(npx.any(npx.isnan(vs.theta_pwp[2:-2, 2:-2]))):
        raise ValueError("theta_pwp-parameter contains non-numeric values.")

    if global_and(npx.any(npx.isnan(vs.theta_ufc[2:-2, 2:-2]))):
        raise ValueError("theta_ufc-parameter contains non-numeric values.")

    if global_and(npx.any(npx.isnan(vs.theta_ac[2:-2, 2:-2]))):
        raise ValueError("theta_ac-parameter contains non-numeric values.")

    if global_and(npx.any(npx.isnan(vs.ks[2:-2, 2:-2]))):
        raise ValueError("ks-parameter contains non-numeric values.")

    if global_and(npx.any(npx.isnan(vs.kf[2:-2, 2:-2]))):
        raise ValueError("kf-parameter contains non-numeric values.")

    if settings.enable_lateral_flow:
        if global_and(npx.any(npx.isnan(vs.dmph[2:-2, 2:-2]))):
            raise ValueError("dmph-parameter contains non-numeric values.")


def validate_initial_conditions_surface(state):
    pass


def validate_initial_conditions_soil(state):
    vs = state.variables
    mask1 = vs.theta_rz[:, :, vs.taum1] > vs.theta_sat
    if global_and(npx.any(mask1[2:-2, 2:-2])):
        raise ValueError("theta_rz is too high.")

    mask2 = vs.theta_ss[:, :, vs.taum1] > vs.theta_sat
    if global_and(npx.any(mask2[2:-2, 2:-2])):
        raise ValueError("theta_ss is too high.")


def validate_initial_conditions_groundwater(state):
    pass


@roger_kernel
def calc_storage_kernel(state):
    vs = state.variables
    settings = state.settings

    if not settings.enable_film_flow:
        vs.S = update(
            vs.S,
            at[2:-2, 2:-2, vs.tau],
            vs.S_sur[2:-2, 2:-2, vs.tau] + vs.S_s[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2],
        )

    elif settings.enable_film_flow:
        vs.S = update(
            vs.S,
            at[2:-2, 2:-2, vs.tau],
            vs.S_sur[2:-2, 2:-2, vs.tau]
            + vs.S_s[2:-2, 2:-2, vs.tau]
            + npx.sum(vs.S_f[2:-2, 2:-2, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
        )

    vs.dS = update(
        vs.dS, at[2:-2, 2:-2], vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1] * vs.maskCatch[2:-2, 2:-2]
    )

    return KernelOutput(
        S=vs.S,
        dS=vs.dS,
    )


@roger_kernel
def calc_storage_with_gwr_kernel(state):
    vs = state.variables

    vs.S = update(
        vs.S,
        at[2:-2, 2:-2, vs.tau],
        vs.S_sur[2:-2, 2:-2, vs.tau]
        + vs.S_s[2:-2, 2:-2, vs.tau]
        + vs.S_vad[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.dS = update(
        vs.dS, at[2:-2, 2:-2], vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1] * vs.maskCatch[2:-2, 2:-2]
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
        at[2:-2, 2:-2, vs.tau],
        vs.S_sur[2:-2, 2:-2, vs.tau]
        + vs.S_s[2:-2, 2:-2, vs.tau]
        + vs.S_vad[2:-2, 2:-2, vs.tau]
        + vs.S_gw[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.dS = update(
        vs.dS, at[2:-2, 2:-2], vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1] * vs.maskCatch[2:-2, 2:-2]
    )

    return KernelOutput(
        S=vs.S,
        dS=vs.dS,
    )


@roger_routine
def calc_storage(state):
    """
    Calculates storage volume and storage change from all storage layers
    """
    vs = state.variables
    settings = state.settings

    if not settings.enable_groundwater:
        vs.update(calc_storage_kernel(state))
    elif settings.enable_groundwater_boundary:
        vs.update(calc_storage_with_gwr_kernel(state))
    elif settings.enable_groundwater:
        vs.update(calc_storage_with_gw_kernel(state))


@roger_kernel
def calc_dS_num_error(state):
    """
    Calculates numerical error of water balance
    """
    vs = state.variables
    settings = state.settings

    if (
        settings.enable_lateral_flow
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_groundwater_boundary
        and not settings.enable_groundwater
        and not settings.enable_offline_transport
    ):
        vs.dS_num_error = update(
            vs.dS_num_error,
            at[2:-2, 2:-2],
            npx.abs(
                (vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1])
                - (
                    vs.prec[2:-2, 2:-2, vs.tau]
                    - vs.q_sur[2:-2, 2:-2]
                    - vs.aet[2:-2, 2:-2]
                    - vs.q_ss[2:-2, 2:-2]
                    - vs.q_sub[2:-2, 2:-2]
                )
            ),
        )
    elif (
        settings.enable_lateral_flow
        and settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_groundwater_boundary
        and not settings.enable_groundwater
        and not settings.enable_offline_transport
    ):
        vs.dS_num_error = update(
            vs.dS_num_error,
            at[2:-2, 2:-2],
            npx.abs(
                (vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1])
                - (
                    vs.prec[2:-2, 2:-2, vs.tau]
                    - vs.q_sur_out[2:-2, 2:-2]
                    + vs.q_sur_in[2:-2, 2:-2]
                    - vs.aet[2:-2, 2:-2]
                    - vs.q_ss[2:-2, 2:-2]
                    - vs.q_sub_out[2:-2, 2:-2]
                    + vs.q_sub_in[2:-2, 2:-2]
                )
            ),
        )
    elif (
        settings.enable_lateral_flow
        and settings.enable_groundwater_boundary
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_offline_transport
    ):
        vs.dS_num_error = update(
            vs.dS_num_error,
            at[2:-2, 2:-2],
            npx.abs(
                (vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1])
                - (
                    vs.prec[2:-2, 2:-2, vs.tau]
                    - vs.q_sur[2:-2, 2:-2]
                    - vs.aet[2:-2, 2:-2]
                    - vs.q_re[2:-2, 2:-2]
                    - vs.q_sub[2:-2, 2:-2]
                    + vs.cpr_ss[2:-2, 2:-2]
                )
            ),
        )

    elif (
        settings.enable_lateral_flow
        and settings.enable_groundwater
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_offline_transport
    ):
        pass

    elif (
        not settings.enable_crop_phenology
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_lateral_flow
        and not settings.enable_groundwater_boundary
        and not settings.enable_groundwater
        and not settings.enable_offline_transport
    ):
        vs.dS_num_error = update(
            vs.dS_num_error,
            at[2:-2, 2:-2],
            npx.abs(
                (vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1])
                - (vs.prec[2:-2, 2:-2, vs.tau] - vs.q_sur[2:-2, 2:-2] - vs.aet[2:-2, 2:-2] - vs.q_ss[2:-2, 2:-2])
            ),
        )

        vs.dS_rz_num_error = update(
            vs.dS_rz_num_error,
            at[2:-2, 2:-2],
            npx.abs(
                (vs.S_rz[2:-2, 2:-2, vs.tau] - vs.S_rz[2:-2, 2:-2, vs.taum1])
                - (
                    vs.inf_mat_rz[2:-2, 2:-2]
                    + vs.inf_mp_rz[2:-2, 2:-2]
                    + vs.inf_sc_rz[2:-2, 2:-2]
                    + vs.cpr_rz[2:-2, 2:-2]
                    - vs.transp[2:-2, 2:-2]
                    - vs.evap_soil[2:-2, 2:-2]
                    - vs.q_rz[2:-2, 2:-2]
                )
            ),
        )

        vs.dS_ss_num_error = update(
            vs.dS_ss_num_error,
            at[2:-2, 2:-2],
            npx.abs(
                (vs.S_ss[2:-2, 2:-2, vs.tau] - vs.S_ss[2:-2, 2:-2, vs.taum1])
                - (vs.inf_mp_ss[2:-2, 2:-2] + vs.q_rz[2:-2, 2:-2] - vs.q_ss[2:-2, 2:-2] - vs.cpr_rz[2:-2, 2:-2])
            ),
        )

    elif (
        settings.enable_crop_phenology
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_lateral_flow
        and not settings.enable_groundwater_boundary
        and not settings.enable_groundwater
        and not settings.enable_offline_transport
    ):
        vs.dS_num_error = update(
            vs.dS_num_error,
            at[2:-2, 2:-2],
            npx.abs(
                (vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1])
                - (vs.prec[2:-2, 2:-2, vs.tau] - vs.q_sur[2:-2, 2:-2] - vs.aet[2:-2, 2:-2] - vs.q_ss[2:-2, 2:-2])
            ),
        )

        vs.dS_rz_num_error = update(
            vs.dS_rz_num_error,
            at[2:-2, 2:-2],
            npx.abs(
                (vs.S_rz[2:-2, 2:-2, vs.tau] - vs.S_rz[2:-2, 2:-2, vs.taum1])
                - (
                    vs.inf_mat_rz[2:-2, 2:-2]
                    + vs.inf_mp_rz[2:-2, 2:-2]
                    + vs.inf_sc_rz[2:-2, 2:-2]
                    + vs.cpr_rz[2:-2, 2:-2]
                    + vs.re_rg[2:-2, 2:-2]
                    - vs.transp[2:-2, 2:-2]
                    - vs.evap_soil[2:-2, 2:-2]
                    - vs.q_rz[2:-2, 2:-2]
                    - vs.re_rl[2:-2, 2:-2]
                )
            ),
        )

        vs.dS_ss_num_error = update(
            vs.dS_ss_num_error,
            at[2:-2, 2:-2],
            npx.abs(
                (vs.S_ss[2:-2, 2:-2, vs.tau] - vs.S_ss[2:-2, 2:-2, vs.taum1])
                - (
                    vs.inf_mp_ss[2:-2, 2:-2]
                    + vs.q_rz[2:-2, 2:-2]
                    + vs.re_rl[2:-2, 2:-2]
                    - vs.re_rg[2:-2, 2:-2]
                    - vs.q_ss[2:-2, 2:-2]
                    - vs.cpr_rz[2:-2, 2:-2]
                )
            ),
        )

    elif (
        not settings.enable_lateral_flow
        and settings.enable_groundwater_boundary
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_groundwater
        and not settings.enable_offline_transport
    ):
        vs.dS_num_error = update(
            vs.dS_num_error,
            at[2:-2, 2:-2],
            npx.abs(
                (vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1])
                - (
                    vs.prec[2:-2, 2:-2, vs.tau]
                    - vs.q_sur[2:-2, 2:-2]
                    - vs.aet[2:-2, 2:-2]
                    - vs.q_ss[2:-2, 2:-2]
                    + vs.cpr_ss[2:-2, 2:-2]
                )
            ),
        )

    elif (
        settings.enable_offline_transport
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not (settings.enable_groundwater_boundary | settings.enable_crop_phenology)
    ):
        vs.dS_num_error = update(
            vs.dS_num_error,
            at[2:-2, 2:-2],
            npx.abs(
                (npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1))
                - (
                    vs.inf_mat_rz[2:-2, 2:-2]
                    + vs.inf_pf_rz[2:-2, 2:-2]
                    + vs.inf_pf_ss[2:-2, 2:-2]
                    - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2)
                    - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2)
                    - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)
                )
                * settings.h
            ),
        )
    elif (
        settings.enable_offline_transport
        and settings.enable_crop_phenology
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_groundwater_boundary
    ):
        vs.dS_num_error = update(
            vs.dS_num_error,
            at[2:-2, 2:-2],
            npx.abs(
                (npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1))
                - (
                    vs.inf_mat_rz[2:-2, 2:-2]
                    + vs.inf_pf_rz[2:-2, 2:-2]
                    + vs.inf_pf_ss[2:-2, 2:-2]
                    - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2)
                    - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2)
                    - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)
                )
                * settings.h
            ),
        )
    return KernelOutput(
        dS_num_error=vs.dS_num_error,
        dS_rz_num_error=vs.dS_rz_num_error,
        dS_ss_num_error=vs.dS_ss_num_error,
    )


@roger_kernel
def calc_dC_num_error(state):
    """
    Calculates numerical error of solute balance
    """
    vs = state.variables
    settings = state.settings

    if (
        settings.enable_offline_transport
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_lateral_flow
        and not settings.enable_groundwater_boundary
        and (settings.enable_deuterium or settings.enable_oxygen18)
    ):
        vs.dC_num_error = update(
            vs.dC_num_error,
            at[2:-2, 2:-2],
            npx.abs(
                (
                    npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.tau]
                    - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.taum1]
                )
                - (
                    vs.inf_mat_rz[2:-2, 2:-2]
                    * npx.where(npx.isnan(vs.C_inf_mat_rz[2:-2, 2:-2]), 0, vs.C_inf_mat_rz[2:-2, 2:-2])
                    + vs.inf_pf_rz[2:-2, 2:-2]
                    * npx.where(npx.isnan(vs.C_inf_pf_rz[2:-2, 2:-2]), 0, vs.C_inf_pf_rz[2:-2, 2:-2])
                    + vs.inf_pf_ss[2:-2, 2:-2]
                    * npx.where(npx.isnan(vs.C_inf_pf_ss[2:-2, 2:-2]), 0, vs.C_inf_pf_ss[2:-2, 2:-2])
                    - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_evap_soil[2:-2, 2:-2]), 0, vs.C_evap_soil[2:-2, 2:-2])
                    - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_transp[2:-2, 2:-2]), 0, vs.C_transp[2:-2, 2:-2])
                    - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_q_ss[2:-2, 2:-2]), 0, vs.C_q_ss[2:-2, 2:-2])
                )
                * settings.h
            ),
        )

    elif (
        settings.enable_offline_transport
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_lateral_flow
        and not settings.enable_groundwater_boundary
        and settings.enable_virtualtracer
    ):
        vs.dC_num_error = update(
            vs.dC_num_error,
            at[2:-2, 2:-2],
            npx.abs(
                (
                    npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.tau]
                    - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.taum1]
                )
                - (
                    vs.inf_mat_rz[2:-2, 2:-2]
                    * npx.where(npx.isnan(vs.C_inf_mat_rz[2:-2, 2:-2]), 0, vs.C_inf_mat_rz[2:-2, 2:-2])
                    + vs.inf_pf_rz[2:-2, 2:-2]
                    * npx.where(npx.isnan(vs.C_inf_pf_rz[2:-2, 2:-2]), 0, vs.C_inf_pf_rz[2:-2, 2:-2])
                    + vs.inf_pf_ss[2:-2, 2:-2]
                    * npx.where(npx.isnan(vs.C_inf_pf_ss[2:-2, 2:-2]), 0, vs.C_inf_pf_ss[2:-2, 2:-2])
                    - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_evap_soil[2:-2, 2:-2]), 0, vs.C_evap_soil[2:-2, 2:-2])
                    - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_transp[2:-2, 2:-2]), 0, vs.C_transp[2:-2, 2:-2])
                    - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_q_ss[2:-2, 2:-2]), 0, vs.C_q_ss[2:-2, 2:-2])
                )
                * settings.h
            ),
        )

    elif (
        settings.enable_offline_transport
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and settings.enable_crop_phenology
        and not settings.enable_lateral_flow
        and not settings.enable_groundwater_boundary
        and settings.enable_virtualtracer
    ):
        vs.dC_num_error = update(
            vs.dC_num_error,
            at[2:-2, 2:-2],
            npx.abs(
                (
                    npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.tau]
                    - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.taum1]
                )
                - (
                    vs.inf_mat_rz[2:-2, 2:-2]
                    * npx.where(npx.isnan(vs.C_inf_mat_rz[2:-2, 2:-2]), 0, vs.C_inf_mat_rz[2:-2, 2:-2])
                    + vs.inf_pf_rz[2:-2, 2:-2]
                    * npx.where(npx.isnan(vs.C_inf_pf_rz[2:-2, 2:-2]), 0, vs.C_inf_pf_rz[2:-2, 2:-2])
                    + vs.inf_pf_ss[2:-2, 2:-2]
                    * npx.where(npx.isnan(vs.C_inf_pf_ss[2:-2, 2:-2]), 0, vs.C_inf_pf_ss[2:-2, 2:-2])
                    - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_evap_soil[2:-2, 2:-2]), 0, vs.C_evap_soil[2:-2, 2:-2])
                    - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_transp[2:-2, 2:-2]), 0, vs.C_transp[2:-2, 2:-2])
                    - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_q_ss[2:-2, 2:-2]), 0, vs.C_q_ss[2:-2, 2:-2])
                )
                * settings.h
            ),
        )

        vs.dC_rz_num_error = update(
            vs.dC_rz_num_error,
            at[2:-2, 2:-2],
            npx.abs(
                (
                    npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_rz[2:-2, 2:-2, vs.tau]
                    - npx.sum(vs.sa_rz[2:-2, 2:-2, vs.taum1, :], axis=-1) * vs.C_rz[2:-2, 2:-2, vs.taum1]
                )
                - (
                    vs.inf_mat_rz[2:-2, 2:-2]
                    * npx.where(npx.isnan(vs.C_inf_mat_rz[2:-2, 2:-2]), 0, vs.C_inf_mat_rz[2:-2, 2:-2])
                    + vs.inf_pf_rz[2:-2, 2:-2]
                    * npx.where(npx.isnan(vs.C_inf_pf_rz[2:-2, 2:-2]), 0, vs.C_inf_pf_rz[2:-2, 2:-2])
                    + npx.sum(vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_cpr_rz[2:-2, 2:-2]), 0, vs.C_cpr_rz[2:-2, 2:-2])
                    + npx.sum(vs.re_rg[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rg[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_re_rg[2:-2, 2:-2]), 0, vs.C_re_rg[2:-2, 2:-2])
                    - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_evap_soil[2:-2, 2:-2]), 0, vs.C_evap_soil[2:-2, 2:-2])
                    - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_transp[2:-2, 2:-2]), 0, vs.C_transp[2:-2, 2:-2])
                    - npx.sum(vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_q_rz[2:-2, 2:-2]), 0, vs.C_q_rz[2:-2, 2:-2])
                    - npx.sum(vs.re_rl[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rl[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_re_rl[2:-2, 2:-2]), 0, vs.C_re_rl[2:-2, 2:-2])
                )
                * settings.h
            ),
        )

        vs.dC_ss_num_error = update(
            vs.dC_ss_num_error,
            at[2:-2, 2:-2],
            npx.abs(
                (
                    npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_ss[2:-2, 2:-2, vs.tau]
                    - npx.sum(vs.sa_ss[2:-2, 2:-2, vs.taum1, :], axis=-1) * vs.C_ss[2:-2, 2:-2, vs.taum1]
                )
                - (
                    vs.inf_pf_ss[2:-2, 2:-2]
                    * npx.where(npx.isnan(vs.C_inf_pf_ss[2:-2, 2:-2]), 0, vs.C_inf_pf_ss[2:-2, 2:-2])
                    + npx.sum(vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_q_rz[2:-2, 2:-2]), 0, vs.C_q_rz[2:-2, 2:-2])
                    + npx.sum(vs.re_rl[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rl[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_re_rl[2:-2, 2:-2]), 0, vs.C_re_rl[2:-2, 2:-2])
                    - npx.sum(vs.re_rg[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rg[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_re_rg[2:-2, 2:-2]), 0, vs.C_re_rg[2:-2, 2:-2])
                    - npx.sum(vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_cpr_rz[2:-2, 2:-2]), 0, vs.C_cpr_rz[2:-2, 2:-2])
                    - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_q_ss[2:-2, 2:-2]), 0, vs.C_q_ss[2:-2, 2:-2])
                )
                * settings.h
            ),
        )

    elif (
        settings.enable_offline_transport
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_lateral_flow
        and not settings.enable_groundwater_boundary
        and (settings.enable_bromide or settings.enable_chloride or settings.enable_nitrate)
    ):
        vs.dC_num_error = update(
            vs.dC_num_error,
            at[2:-2, 2:-2],
            npx.abs(
                (
                    npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.tau]
                    - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.taum1]
                )
                - (
                    vs.inf_mat_rz[2:-2, 2:-2]
                    * npx.where(npx.isnan(vs.C_inf_mat_rz[2:-2, 2:-2]), 0, vs.C_inf_mat_rz[2:-2, 2:-2])
                    + vs.inf_pf_rz[2:-2, 2:-2]
                    * npx.where(npx.isnan(vs.C_inf_pf_rz[2:-2, 2:-2]), 0, vs.C_inf_pf_rz[2:-2, 2:-2])
                    + vs.inf_pf_ss[2:-2, 2:-2]
                    * npx.where(npx.isnan(vs.C_inf_pf_ss[2:-2, 2:-2]), 0, vs.C_inf_pf_ss[2:-2, 2:-2])
                    - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_transp[2:-2, 2:-2]), 0, vs.C_transp[2:-2, 2:-2])
                    - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)
                    * npx.where(npx.isnan(vs.C_q_ss[2:-2, 2:-2]), 0, vs.C_q_ss[2:-2, 2:-2])
                )
                * settings.h
            ),
        )

    return KernelOutput(
        dC_num_error=vs.dC_num_error,
        dC_rz_num_error=vs.dC_rz_num_error,
        dC_ss_num_error=vs.dC_ss_num_error,
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

    check = True

    if (
        settings.enable_lateral_flow
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_groundwater_boundary
        and not settings.enable_groundwater
        and not settings.enable_offline_transport
    ):
        check1 = global_and(
            npx.all(
                npx.isclose(
                    vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1],
                    vs.prec[2:-2, 2:-2, vs.tau]
                    - vs.q_sur[2:-2, 2:-2]
                    - vs.aet[2:-2, 2:-2]
                    - vs.q_ss[2:-2, 2:-2]
                    - vs.q_sub[2:-2, 2:-2],
                    atol=settings.atol,
                    rtol=settings.rtol,
                )
            )
        )
        check2 = global_and(
            npx.all(
                (vs.S_fp_rz[2:-2, 2:-2] > -settings.atol)
                & (vs.S_lp_rz[2:-2, 2:-2] > -settings.atol)
                & (vs.S_fp_ss[2:-2, 2:-2] > -settings.atol)
                & (vs.S_lp_ss[2:-2, 2:-2] > -settings.atol)
            )
        )
        check3 = global_and(
            npx.all(
                (vs.S_fp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ufc_rz[2:-2, 2:-2])
                & (vs.S_lp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ac_rz[2:-2, 2:-2])
                & (vs.S_fp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ufc_ss[2:-2, 2:-2])
                & (vs.S_lp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ac_ss[2:-2, 2:-2])
            )
        )
        # dS = vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1]
        # dF = vs.prec[2:-2, 2:-2, vs.tau] - vs.q_sur[2:-2, 2:-2] - vs.aet[2:-2, 2:-2] - vs.q_ss[2:-2, 2:-2]
        # print(vs.S[2:-2, 2:-2, vs.tau])
        check = check1 & check2 & check3

    elif (
        settings.enable_lateral_flow
        and settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_groundwater_boundary
        and not settings.enable_groundwater
        and not settings.enable_offline_transport
    ):
        check1 = global_and(
            npx.all(
                npx.isclose(
                    vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1],
                    vs.prec[2:-2, 2:-2, vs.tau]
                    - vs.q_sur_out[2:-2, 2:-2]
                    + vs.q_sur_in[2:-2, 2:-2]
                    - vs.aet[2:-2, 2:-2]
                    - vs.q_ss[2:-2, 2:-2]
                    - vs.q_sub_out[2:-2, 2:-2]
                    + vs.q_sub_in[2:-2, 2:-2],
                    atol=settings.atol,
                    rtol=settings.rtol,
                )
            )
        )
        check2 = global_and(
            npx.all(
                (vs.S_fp_rz[2:-2, 2:-2] > -settings.atol)
                & (vs.S_lp_rz[2:-2, 2:-2] > -settings.atol)
                & (vs.S_fp_ss[2:-2, 2:-2] > -settings.atol)
                & (vs.S_lp_ss[2:-2, 2:-2] > -settings.atol)
            )
        )
        check3 = global_and(
            npx.all(
                (vs.S_fp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ufc_rz[2:-2, 2:-2])
                & (vs.S_lp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ac_rz[2:-2, 2:-2])
                & (vs.S_fp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ufc_ss[2:-2, 2:-2])
                & (vs.S_lp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ac_ss[2:-2, 2:-2])
            )
        )
        # dS = vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1]
        # dF = vs.prec[2:-2, 2:-2, vs.tau] - vs.q_sur[2:-2, 2:-2] - vs.aet[2:-2, 2:-2] - vs.q_ss[2:-2, 2:-2]
        # print(vs.S[2:-2, 2:-2, vs.tau])
        check = check1 & check2 & check3

    elif (
        settings.enable_lateral_flow
        and settings.enable_groundwater_boundary
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_offline_transport
    ):
        check1 = global_and(
            npx.all(
                npx.isclose(
                    vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1],
                    vs.prec[2:-2, 2:-2, vs.tau]
                    - vs.q_sur[2:-2, 2:-2]
                    - vs.aet[2:-2, 2:-2]
                    - vs.q_ss[2:-2, 2:-2]
                    - vs.q_sub[2:-2, 2:-2]
                    + vs.cpr_ss[2:-2, 2:-2],
                    atol=settings.atol,
                    rtol=settings.rtol,
                )
            )
        )
        check2 = global_and(
            npx.all(
                (vs.S_fp_rz[2:-2, 2:-2] > -settings.atol)
                & (vs.S_lp_rz[2:-2, 2:-2] > -settings.atol)
                & (vs.S_fp_ss[2:-2, 2:-2] > -settings.atol)
                & (vs.S_lp_ss[2:-2, 2:-2] > -settings.atol)
            )
        )
        check3 = global_and(
            npx.all(
                (vs.S_fp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ufc_rz[2:-2, 2:-2])
                & (vs.S_lp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ac_rz[2:-2, 2:-2])
                & (vs.S_fp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ufc_ss[2:-2, 2:-2])
                & (vs.S_lp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ac_ss[2:-2, 2:-2])
            )
        )
        check = check1 & check2 & check3

    elif (
        settings.enable_lateral_flow
        and settings.enable_groundwater
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_offline_transport
    ):
        check1 = global_and(
            npx.all(
                npx.isclose(
                    vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1],
                    vs.prec[2:-2, 2:-2, vs.tau]
                    - vs.q_sur[2:-2, 2:-2]
                    - vs.aet[2:-2, 2:-2]
                    - vs.q_sub[2:-2, 2:-2]
                    - vs.q_gw[2:-2, 2:-2]
                    - vs.q_leak[2:-2, 2:-2],
                    atol=settings.atol,
                    rtol=settings.rtol,
                )
            )
        )
        check2 = global_and(
            npx.all(
                (vs.S_fp_rz[2:-2, 2:-2] > -settings.atol)
                & (vs.S_lp_rz[2:-2, 2:-2] > -settings.atol)
                & (vs.S_fp_ss[2:-2, 2:-2] > -settings.atol)
                & (vs.S_lp_ss[2:-2, 2:-2] > -settings.atol)
            )
        )
        check3 = global_and(
            npx.all(
                (vs.S_fp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ufc_rz[2:-2, 2:-2])
                & (vs.S_lp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ac_rz[2:-2, 2:-2])
                & (vs.S_fp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ufc_ss[2:-2, 2:-2])
                & (vs.S_lp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ac_ss[2:-2, 2:-2])
            )
        )
        check = check1 & check2 & check3

    elif (
        settings.enable_groundwater_boundary
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_lateral_flow
        and not settings.enable_offline_transport
    ):
        check1 = global_and(
            npx.all(
                npx.isclose(
                    vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1],
                    vs.prec[2:-2, 2:-2, vs.tau]
                    - vs.q_sur[2:-2, 2:-2]
                    - vs.aet[2:-2, 2:-2]
                    - vs.q_ss[2:-2, 2:-2]
                    + vs.cpr_ss[2:-2, 2:-2],
                    atol=settings.atol,
                    rtol=settings.rtol,
                )
            )
        )
        check2 = global_and(
            npx.all(
                (vs.S_fp_rz[2:-2, 2:-2] > -settings.atol)
                & (vs.S_lp_rz[2:-2, 2:-2] > -settings.atol)
                & (vs.S_fp_ss[2:-2, 2:-2] > -settings.atol)
                & (vs.S_lp_ss[2:-2, 2:-2] > -settings.atol)
            )
        )
        check3 = global_and(
            npx.all(
                (vs.S_fp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ufc_rz[2:-2, 2:-2])
                & (vs.S_lp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ac_rz[2:-2, 2:-2])
                & (vs.S_fp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ufc_ss[2:-2, 2:-2])
                & (vs.S_lp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ac_ss[2:-2, 2:-2])
            )
        )
        check = check1 & check2 & check3

    elif (
        settings.enable_film_flow
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_offline_transport
    ):
        check1 = global_and(
            npx.all(
                npx.isclose(
                    vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1],
                    vs.prec[2:-2, 2:-2, vs.tau]
                    - vs.q_sur[2:-2, 2:-2]
                    - vs.aet[2:-2, 2:-2]
                    - vs.q_ss[2:-2, 2:-2]
                    - vs.ff_drain[2:-2, 2:-2],
                    atol=settings.atol,
                    rtol=settings.rtol,
                )
            )
        )
        check2 = global_and(
            npx.all(
                (vs.S_fp_rz[2:-2, 2:-2] > -settings.atol)
                & (vs.S_lp_rz[2:-2, 2:-2] > -settings.atol)
                & (vs.S_fp_ss[2:-2, 2:-2] > -settings.atol)
                & (vs.S_lp_ss[2:-2, 2:-2] > -settings.atol)
            )
        )
        check3 = global_and(
            npx.all(
                (vs.S_fp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ufc_rz[2:-2, 2:-2])
                & (vs.S_lp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ac_rz[2:-2, 2:-2])
                & (vs.S_fp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ufc_ss[2:-2, 2:-2])
                & (vs.S_lp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ac_ss[2:-2, 2:-2])
            )
        )
        check = check1 & check2 & check3

    elif (
        not settings.enable_lateral_flow
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_groundwater_boundary
        and not settings.enable_groundwater
        and not settings.enable_offline_transport
    ):
        check1 = global_and(
            npx.all(
                npx.isclose(
                    vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1],
                    vs.prec[2:-2, 2:-2, vs.tau] - vs.q_sur[2:-2, 2:-2] - vs.aet[2:-2, 2:-2] - vs.q_ss[2:-2, 2:-2],
                    atol=settings.atol,
                    rtol=settings.rtol,
                )
            )
        )
        check2 = global_and(
            npx.all(
                (vs.S_fp_rz[2:-2, 2:-2] > -settings.atol)
                & (vs.S_lp_rz[2:-2, 2:-2] > -settings.atol)
                & (vs.S_fp_ss[2:-2, 2:-2] > -settings.atol)
                & (vs.S_lp_ss[2:-2, 2:-2] > -settings.atol)
            )
        )
        check3 = global_and(
            npx.all(
                (vs.S_fp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ufc_rz[2:-2, 2:-2])
                & (vs.S_lp_rz[2:-2, 2:-2] - settings.atol <= vs.S_ac_rz[2:-2, 2:-2])
                & (vs.S_fp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ufc_ss[2:-2, 2:-2])
                & (vs.S_lp_ss[2:-2, 2:-2] - settings.atol <= vs.S_ac_ss[2:-2, 2:-2])
            )
        )
        check = check1 & check2 & check3

    elif (
        settings.enable_offline_transport
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_groundwater_boundary
        and not (
            settings.enable_deuterium
            or settings.enable_oxygen18
            or settings.enable_bromide
            or settings.enable_chloride
            or settings.enable_nitrate
            or settings.enable_virtualtracer
        )
    ):
        check = global_and(
            npx.all(
                npx.isclose(
                    npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1)
                    - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1),
                    (
                        vs.inf_mat_rz[2:-2, 2:-2]
                        + vs.inf_pf_rz[2:-2, 2:-2]
                        + vs.inf_pf_ss[2:-2, 2:-2]
                        - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2)
                        - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2)
                        - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)
                    )
                    * settings.h,
                    atol=settings.atol,
                    rtol=settings.rtol,
                )
            )
        )

    elif (
        settings.enable_offline_transport
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_groundwater_boundary
        and (settings.enable_deuterium or settings.enable_oxygen18)
    ):
        check1 = global_and(
            npx.all(
                npx.isclose(
                    npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1)
                    - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1),
                    (
                        vs.inf_mat_rz[2:-2, 2:-2]
                        + vs.inf_pf_rz[2:-2, 2:-2]
                        + vs.inf_pf_ss[2:-2, 2:-2]
                        - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2)
                        - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2)
                        - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)
                    )
                    * settings.h,
                    atol=settings.atol,
                    rtol=settings.rtol,
                )
            )
        )
        check2 = global_and(
            npx.all(
                npx.isclose(
                    npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.tau]
                    - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.taum1],
                    (
                        vs.inf_mat_rz[2:-2, 2:-2]
                        * npx.where(npx.isnan(vs.C_inf_mat_rz[2:-2, 2:-2]), 0, vs.C_inf_mat_rz[2:-2, 2:-2])
                        + vs.inf_pf_rz[2:-2, 2:-2]
                        * npx.where(npx.isnan(vs.C_inf_pf_rz[2:-2, 2:-2]), 0, vs.C_inf_pf_rz[2:-2, 2:-2])
                        + vs.inf_pf_ss[2:-2, 2:-2]
                        * npx.where(npx.isnan(vs.C_inf_pf_ss[2:-2, 2:-2]), 0, vs.C_inf_pf_ss[2:-2, 2:-2])
                        - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2)
                        * npx.where(npx.isnan(vs.C_evap_soil[2:-2, 2:-2]), 0, vs.C_evap_soil[2:-2, 2:-2])
                        - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2)
                        * npx.where(npx.isnan(vs.C_transp[2:-2, 2:-2]), 0, vs.C_transp[2:-2, 2:-2])
                        - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)
                        * npx.where(npx.isnan(vs.C_q_ss[2:-2, 2:-2]), 0, vs.C_q_ss[2:-2, 2:-2])
                    )
                    * settings.h,
                    atol=settings.atol,
                    rtol=settings.rtol,
                )
            )
        )
        check = check1 & check2

        # compare storage volume to saturated storage volume
        check3 = global_and(
            npx.all(
                (
                    npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1)
                    <= (vs.S_sat_rz[2:-2, 2:-2] + vs.S_sat_ss[2:-2, 2:-2])
                    - (vs.S_pwp_rz[2:-2, 2:-2] + vs.S_pwp_ss[2:-2, 2:-2])
                )
                & (npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) >= 0)
            )
        )
        check4 = global_and(
            npx.all(
                (npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1) <= vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
                & (npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1) >= 0)
            )
        )
        check5 = global_and(
            npx.all(
                (npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1) <= vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2])
                & (npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1) >= 0)
            )
        )
        check_bounds = check3 & check4 & check5

        if rs.loglevel == "warning" and rs.backend == "numpy" and not (check & check_bounds):
            check11 = npx.isclose(
                npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1),
                vs.inf_mat_rz[2:-2, 2:-2]
                + vs.inf_pf_rz[2:-2, 2:-2]
                + vs.inf_pf_ss[2:-2, 2:-2]
                - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2)
                - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2)
                - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2),
                atol=settings.atol,
                rtol=settings.rtol,
            )
            check22 = npx.isclose(
                npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.tau]
                - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.taum1],
                vs.inf_mat_rz[2:-2, 2:-2]
                * npx.where(npx.isnan(vs.C_inf_mat_rz[2:-2, 2:-2]), 0, vs.C_inf_mat_rz[2:-2, 2:-2])
                + vs.inf_pf_rz[2:-2, 2:-2]
                * npx.where(npx.isnan(vs.C_inf_pf_rz[2:-2, 2:-2]), 0, vs.C_inf_pf_rz[2:-2, 2:-2])
                + vs.inf_pf_ss[2:-2, 2:-2]
                * npx.where(npx.isnan(vs.C_inf_pf_ss[2:-2, 2:-2]), 0, vs.C_inf_pf_ss[2:-2, 2:-2])
                - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2)
                * npx.where(npx.isnan(vs.C_evap_soil[2:-2, 2:-2]), 0, vs.C_evap_soil[2:-2, 2:-2])
                - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2)
                * npx.where(npx.isnan(vs.C_transp[2:-2, 2:-2]), 0, vs.C_transp[2:-2, 2:-2])
                - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)
                * npx.where(npx.isnan(vs.C_q_ss[2:-2, 2:-2]), 0, vs.C_q_ss[2:-2, 2:-2]),
                atol=settings.atol,
                rtol=settings.rtol,
            )
            check33 = (
                npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1)
                <= (vs.S_sat_rz[2:-2, 2:-2] + vs.S_sat_ss[2:-2, 2:-2])
                - (vs.S_pwp_rz[2:-2, 2:-2] + vs.S_pwp_ss[2:-2, 2:-2])
            ) & (npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) >= 0)
            check44 = vs.sa_rz[2:-2, 2:-2, vs.tau, :] >= 0
            check55 = vs.sa_ss[2:-2, 2:-2, vs.tau, :] >= 0

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

    elif (
        settings.enable_offline_transport
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_groundwater_boundary
        and (settings.enable_bromide or settings.enable_chloride)
    ):
        check1 = global_and(
            npx.all(
                npx.isclose(
                    npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1)
                    - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1),
                    vs.inf_mat_rz[2:-2, 2:-2]
                    + vs.inf_pf_rz[2:-2, 2:-2]
                    + vs.inf_pf_ss[2:-2, 2:-2]
                    - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2)
                    - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2)
                    - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2),
                    atol=settings.atol,
                    rtol=settings.rtol,
                )
            )
        )
        check2 = global_and(
            npx.all(
                npx.isclose(
                    npx.sum(vs.msa_s[2:-2, 2:-2, vs.tau, :], axis=-1)
                    - npx.sum(vs.msa_s[2:-2, 2:-2, vs.taum1, :], axis=-1),
                    vs.inf_mat_rz[2:-2, 2:-2] * vs.C_inf_mat_rz[2:-2, 2:-2]
                    + vs.inf_pf_rz[2:-2, 2:-2] * vs.C_inf_pf_rz[2:-2, 2:-2]
                    + vs.inf_pf_ss[2:-2, 2:-2] * vs.C_inf_pf_ss[2:-2, 2:-2]
                    - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2)
                    * vs.C_transp[2:-2, 2:-2]
                    - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)
                    * vs.C_q_ss[2:-2, 2:-2],
                    atol=settings.atol,
                    rtol=settings.rtol,
                )
            )
        )

        check = check1 & check2

    elif (
        settings.enable_offline_transport
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_groundwater_boundary
        and settings.enable_virtualtracer
    ):
        check1 = global_and(
            npx.all(
                npx.isclose(
                    npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1)
                    - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1),
                    vs.inf_mat_rz[2:-2, 2:-2]
                    + vs.inf_pf_rz[2:-2, 2:-2]
                    + vs.inf_pf_ss[2:-2, 2:-2]
                    - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2)
                    - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2)
                    - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2),
                    atol=settings.atol,
                    rtol=settings.rtol,
                )
            )
        )
        check2 = global_and(
            npx.all(
                npx.isclose(
                    npx.sum(vs.msa_s[2:-2, 2:-2, vs.tau, :], axis=-1)
                    - npx.sum(vs.msa_s[2:-2, 2:-2, vs.taum1, :], axis=-1),
                    vs.inf_mat_rz[2:-2, 2:-2] * vs.C_inf_mat_rz[2:-2, 2:-2]
                    + vs.inf_pf_rz[2:-2, 2:-2] * vs.C_inf_pf_rz[2:-2, 2:-2]
                    + vs.inf_pf_ss[2:-2, 2:-2] * vs.C_inf_pf_ss[2:-2, 2:-2]
                    - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2)
                    * vs.C_transp[2:-2, 2:-2]
                    - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2)
                    * vs.C_evap_soil[2:-2, 2:-2]
                    - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)
                    * vs.C_q_ss[2:-2, 2:-2],
                    atol=settings.atol,
                    rtol=settings.rtol,
                )
            )
        )

        check = check1 & check2

    elif (
        settings.enable_offline_transport
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
        and not settings.enable_groundwater_boundary
        and settings.enable_nitrate
    ):
        check1 = global_and(
            npx.all(
                npx.isclose(
                    npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1)
                    - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1),
                    vs.inf_mat_rz[2:-2, 2:-2]
                    + vs.inf_pf_rz[2:-2, 2:-2]
                    + vs.inf_pf_ss[2:-2, 2:-2]
                    - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2)
                    - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2)
                    - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2),
                    atol=settings.atol,
                    rtol=settings.rtol,
                )
            )
        )
        check2 = global_and(
            npx.all(
                npx.isclose(
                    npx.sum(vs.msa_s[2:-2, 2:-2, vs.tau, :], axis=-1)
                    - npx.sum(vs.msa_s[2:-2, 2:-2, vs.taum1, :], axis=-1),
                    vs.inf_mat_rz[2:-2, 2:-2] * vs.C_inf_mat_rz[2:-2, 2:-2]
                    + vs.inf_pf_rz[2:-2, 2:-2] * vs.C_inf_pf_rz[2:-2, 2:-2]
                    + vs.inf_pf_ss[2:-2, 2:-2] * vs.C_inf_pf_ss[2:-2, 2:-2]
                    + npx.sum(vs.ma_s[2:-2, 2:-2, :], axis=2)
                    - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2)
                    * vs.C_transp[2:-2, 2:-2]
                    - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)
                    * vs.C_q_ss[2:-2, 2:-2]
                    - npx.sum(vs.mr_s[2:-2, 2:-2, :], axis=2),
                    atol=settings.atol,
                    rtol=settings.rtol,
                )
            )
        )

        check = check1 & check2

    elif (
        settings.enable_offline_transport
        and settings.enable_groundwater_boundary
        and not settings.enable_routing_1D
        and not settings.enable_routing_2D
    ):
        pass

    return check
