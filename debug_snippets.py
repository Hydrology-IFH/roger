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

# SVATCROP bromide
vs = model.state.variables
dS = npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1)
fluxes = vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2] - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)

dS_rz = npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_rz[2:-2, 2:-2, vs.taum1, :], axis=-1)
fluxes_rz = vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) - npx.sum(vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :], axis=2) + npx.sum(vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :], axis=2) - npx.sum(vs.re_rl[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rl[2:-2, 2:-2, :], axis=2) + npx.sum(vs.re_rg[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rg[2:-2, 2:-2, :], axis=2)

dS_ss = npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_ss[2:-2, 2:-2, vs.taum1, :], axis=-1)
fluxes_ss = vs.inf_pf_ss[2:-2, 2:-2] + npx.sum(vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :], axis=2) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2) - npx.sum(vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :], axis=2) - npx.sum(vs.re_rg[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rg[2:-2, 2:-2, :], axis=2) + npx.sum(vs.re_rl[2:-2, 2:-2, npx.newaxis] * vs.tt_re_rl[2:-2, 2:-2, :], axis=2)


dM = vs.M_s[2:-2, 2:-2, vs.tau] - vs.M_s[2:-2, 2:-2, vs.taum1]
mfluxes = vs.M_inf_mat_rz[2:-2, 2:-2] + vs.M_inf_pf_rz[2:-2, 2:-2] - vs.M_transp[2:-2, 2:-2] - vs.M_q_ss[2:-2, 2:-2]

# oneD model
check1 = npx.isclose(vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1], vs.prec[2:-2, 2:-2, vs.tau] - vs.q_sur[2:-2, 2:-2] - vs.aet[2:-2, 2:-2] - vs.q_ss[2:-2, 2:-2] - vs.q_sub[2:-2, 2:-2], atol=1e-02)
rows = npx.where(check1 == False)[0]
dS = vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1]
dF = vs.prec[2:-2, 2:-2, vs.tau] - vs.q_sur[2:-2, 2:-2] - vs.aet[2:-2, 2:-2] - vs.q_ss[2:-2, 2:-2] - vs.q_sub[2:-2, 2:-2]


rows = npx.where(mask == True)[0]

row = 2108
vs.S_fp_rz[row,2], vs.S_lp_rz[row,2]
vs.S_ufc_rz[row,2], vs.S_ac_rz[row,2]
vs.S_fp_ss[row,2], vs.S_lp_ss[row,2]
vs.S_ufc_ss[row,2], vs.S_ac_ss[row,2]

vs.S_fp_rz[rows,2], vs.S_lp_rz[rows,2]
vs.S_ufc_rz[rows,2], vs.S_ac_rz[rows,2]
vs.S_fp_ss[rows,2], vs.S_lp_ss[rows,2]
vs.S_ufc_ss[rows,2], vs.S_ac_ss[rows,2]

check2 = (vs.S_fp_rz[2:-2, 2:-2] > -1e-9) & (vs.S_lp_rz[2:-2, 2:-2] > -1e-9) & (vs.S_fp_ss[2:-2, 2:-2] > -1e-9) & (vs.S_lp_ss[2:-2, 2:-2] > -1e-9)
rows = npx.where(check2 == False)[0]

check3 = (vs.S_fp_rz[2:-2, 2:-2] <= vs.S_ufc_rz[2:-2, 2:-2]) & (vs.S_lp_rz[2:-2, 2:-2] <= vs.S_ac_rz[2:-2, 2:-2]) & (vs.S_fp_ss[2:-2, 2:-2] <= vs.S_ufc_ss[2:-2, 2:-2]) & (vs.S_lp_ss[2:-2, 2:-2] <= vs.S_ac_ss[2:-2, 2:-2])
rows = npx.where(check3 == False)[0] + 2

row = 2108
print(vs.S_fp_rz[row,2], vs.S_lp_rz[row,2], vs.S_ufc_rz[row,2], vs.S_ac_rz[row,2])

# add missing initial values
# write states of best model run
import h5netcdf
import numpy as onp
from pathlib import Path
base_path = Path("/Users/robinschwemmle/Desktop/PhD/models/roger/examples/plot_scale/rietholzbach/svat_transport")
states_hm_file = base_path / "states_hm.nc"
with h5netcdf.File(states_hm_file, 'a', decode_vlen_strings=False) as f:
    var_obj = f.variables.get('S_rz')
    var_obj1 = f.variables.get('S_sat_rz')
    vals = onp.array(var_obj)
    vals1 = onp.array(var_obj1)
    var_obj[:, :, 0] = onp.where(0.46 * 400 > vals1[:, :, 1], vals1[:, :, 1], 0.46 * 400)
    var_obj = f.variables.get('S_ss')
    var_obj1 = f.variables.get('S_sat_ss')
    vals = onp.array(var_obj)
    vals1 = onp.array(var_obj1)
    var_obj[:, :, 0] = onp.where(0.44 * 1600 > vals1[:, :, 1], vals1[:, :, 1], 0.44 * 1600)

base_path = Path("/Users/robinschwemmle/Desktop/PhD/models/roger/examples/plot_scale/rietholzbach/svat_monte_carlo")
states_hm_file = base_path / "states_hm_monte_carlo.nc"
with h5netcdf.File(states_hm_file, 'a', decode_vlen_strings=False) as f:
    var_obj = f.variables.get('S_rz')
    var_obj1 = f.variables.get('S_sat_rz')
    vals = onp.array(var_obj)
    vals1 = onp.array(var_obj1)
    var_obj[:, :, 0] = onp.where(0.46 * 400 > vals1[:, :, 1], vals1[:, :, 1], 0.46 * 400)
    var_obj = f.variables.get('S_ss')
    var_obj1 = f.variables.get('S_sat_ss')
    vals = onp.array(var_obj)
    vals1 = onp.array(var_obj1)
    var_obj[:, :, 0] = onp.where(0.44 * 1600 > vals1[:, :, 1], vals1[:, :, 1], 0.44 * 1600)

base_path = Path("/Users/robinschwemmle/Desktop/PhD/models/roger/examples/plot_scale/rietholzbach/svat_sensitivity")
states_hm_file = base_path / "states_hm_sensitivity.nc"
with h5netcdf.File(states_hm_file, 'a', decode_vlen_strings=False) as f:
    var_obj = f.variables.get('S_rz')
    var_obj1 = f.variables.get('S_sat_rz')
    vals = onp.array(var_obj)
    vals1 = onp.array(var_obj1)
    var_obj[:, :, 0] = onp.where(0.46 * 400 > vals1[:, :, 1], vals1[:, :, 1], 0.46 * 400)
    var_obj = f.variables.get('S_ss')
    var_obj1 = f.variables.get('S_sat_ss')
    vals = onp.array(var_obj)
    vals1 = onp.array(var_obj1)
    var_obj[:, :, 0] = onp.where(0.44 * 1600 > vals1[:, :, 1], vals1[:, :, 1], 0.44 * 1600)


# resubmit
tms = transport_model_structure.replace("_", " ")
model = SVATTRANSPORTSetup(override=dict(
        restart_input_filename=f'SVATTRANSPORT_{transport_model_structure}.warmup_restart.h5',
        restart_output_filename=None,
        ))
if tms not in ['complete-mixing', 'piston']:
    model._set_nsamples(nsamples)
else:
    if rs.mpi_comm:
        model._set_nsamples(rst.proc_num)
model._set_tm_structure(tms)
identifier = f'SVATTRANSPORT_{transport_model_structure}'
model._set_identifier(identifier)
input_path = model._base_path / "input"
model._set_input_dir(input_path)
forcing_path = model._input_dir / "forcing_tracer.nc"
if not os.path.exists(forcing_path):
    write_forcing_tracer(input_path, 'd18O')
model.setup()
model.run()
return


print(vs.S_fp_rz[2,2], vs.S_lp_rz[2,2], vs.S_fp_ss[2,2], vs.S_lp_ss[2,2])


h = (vs.ha[2,2]/((theta/vs.theta_sat[2,2])**(1/vs.lambda_bc[2,2])))
k = (vs.ks[2,2]/(1 + (theta/vs.theta_sat[2,2])**(-vs.m_bc[2,2])))
perc = k/vs.ks[2,2]
z = vs.z_gw[2,2, vs.tau] * 1000 - vs.z_soil[2,2]

(npx.power((z)/(-vs.ha[2,2]*10.2), -vs.n_salv[2,2])/(1 + (vs.n_salv[2,2] - 1) * npx.power((z)/(-vs.ha[2,2]*10.2), -vs.n_salv[2,2])))

(npx.power((z)/(-vs.ha]*10.2), -vs.n_salv)/(1 + (vs.n_salv - 1) * npx.power((z)/(-vs.ha*10.2), -vs.n_salv)))
