# SVAT 18O
vs = model.state.variables
dS = npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1)
fluxes = vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2] - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)

dS_rz = npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_rz[2:-2, 2:-2, vs.taum1, :], axis=-1)
fluxes_rz = vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) - npx.sum(vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :], axis=2) + npx.sum(vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :], axis=2)

dS_ss = npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_ss[2:-2, 2:-2, vs.taum1, :], axis=-1)
fluxes_ss = vs.inf_pf_ss[2:-2, 2:-2] + npx.sum(vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :], axis=2) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2) - npx.sum(vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :], axis=2)

dC = npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.tau] - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.taum1]
cfluxes = vs.inf_mat_rz[2:-2, 2:-2] * npx.where(npx.isnan(vs.C_inf_mat_rz[2:-2, 2:-2]), 0, vs.C_inf_mat_rz[2:-2, 2:-2]) + vs.inf_pf_rz[2:-2, 2:-2] * npx.where(npx.isnan(vs.C_inf_pf_rz[2:-2, 2:-2]), 0, vs.C_inf_pf_rz[2:-2, 2:-2]) + vs.inf_pf_ss[2:-2, 2:-2] * npx.where(npx.isnan(vs.C_inf_pf_ss[2:-2, 2:-2]), 0, vs.C_inf_pf_ss[2:-2, 2:-2]) - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_evap_soil[2:-2, 2:-2]), 0, vs.C_evap_soil[2:-2, 2:-2]) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_transp[2:-2, 2:-2]), 0, vs.C_transp[2:-2, 2:-2]) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_q_ss[2:-2, 2:-2]), 0, vs.C_q_ss[2:-2, 2:-2])

dC_rz = npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_rz[2:-2, 2:-2, vs.tau] - npx.sum(vs.sa_rz[2:-2, 2:-2, vs.taum1, :], axis=-1) * vs.C_rz[2:-2, 2:-2, vs.taum1]
cfluxes_rz = vs.inf_mat_rz[2:-2, 2:-2] * npx.where(npx.isnan(vs.C_inf_mat_rz[2:-2, 2:-2]), 0, vs.C_inf_mat_rz[2:-2, 2:-2]) + vs.inf_pf_rz[2:-2, 2:-2] * npx.where(npx.isnan(vs.C_inf_pf_rz[2:-2, 2:-2]), 0, vs.C_inf_pf_rz[2:-2, 2:-2]) - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_evap_soil[2:-2, 2:-2]), 0, vs.C_evap_soil[2:-2, 2:-2]) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_transp[2:-2, 2:-2]), 0, vs.C_transp[2:-2, 2:-2]) - npx.sum(vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_q_rz[2:-2, 2:-2]), 0, vs.C_q_rz[2:-2, 2:-2]) + npx.sum(vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_cpr_rz[2:-2, 2:-2]), 0, vs.C_cpr_rz[2:-2, 2:-2])

dC_ss = npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_ss[2:-2, 2:-2, vs.tau] - npx.sum(vs.sa_ss[2:-2, 2:-2, vs.taum1, :], axis=-1) * vs.C_ss[2:-2, 2:-2, vs.taum1]
cfluxes_ss = vs.inf_pf_ss[2:-2, 2:-2] * npx.where(npx.isnan(vs.C_inf_pf_ss[2:-2, 2:-2]), 0, vs.C_inf_pf_ss[2:-2, 2:-2]) + npx.sum(vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_q_rz[2:-2, 2:-2]), 0, vs.C_q_rz[2:-2, 2:-2]) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_q_ss[2:-2, 2:-2]), 0, vs.C_q_ss[2:-2, 2:-2]) - npx.sum(vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_cpr_rz[2:-2, 2:-2]), 0, vs.C_cpr_rz[2:-2, 2:-2])

vs.C_rz = update(
    vs.C_rz,
    at[2:-2, 2:-2, vs.tau], transport.calc_conc_iso_storage(state, vs.sa_rz, vs.msa_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
)
M_rz1 = npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_rz[2:-2, 2:-2, vs.tau]

vs.C_rz = update(
    vs.C_rz,
    at[2:-2, 2:-2, vs.tau], transport.calc_conc_iso_storage(state, vs.sa_rz, vs.msa_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
)
print(npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_rz[2:-2, 2:-2, vs.tau] - M_rz1)
print(npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_evap_soil[2:-2, 2:-2]), 0, vs.C_evap_soil[2:-2, 2:-2]))

vs.C_rz = update(
    vs.C_rz,
    at[2:-2, 2:-2, vs.tau], transport.calc_conc_iso_storage(state, vs.sa_rz, vs.msa_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
)
print(npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_rz[2:-2, 2:-2, vs.tau] - M_rz1)
print(npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_transp[2:-2, 2:-2]), 0, vs.C_transp[2:-2, 2:-2]))

vs.C_rz = update(
    vs.C_rz,
    at[2:-2, 2:-2, vs.tau], transport.calc_conc_iso_storage(state, vs.sa_rz, vs.msa_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
)
print(npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_rz[2:-2, 2:-2, vs.tau] - M_rz1)
print(npx.sum(vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_q_rz[2:-2, 2:-2]), 0, vs.C_q_rz[2:-2, 2:-2]))

vs.C_ss = update(
    vs.C_ss,
    at[2:-2, 2:-2, vs.tau], transport.calc_conc_iso_storage(state, vs.sa_ss, vs.msa_ss)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
)
