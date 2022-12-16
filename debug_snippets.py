# SVAT 18O
vs = model.state.variables
dS = npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1)
fluxes = (vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2] - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)) * settings.h

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

# SVAT-CROP model
check4 = (vs.S_fp_rz[2:-2, 2:-2] - 1e-9 <= vs.S_ufc_rz[2:-2, 2:-2]) & (vs.S_lp_rz[2:-2, 2:-2] - 1e-9 <= vs.S_ac_rz[2:-2, 2:-2]) & (vs.S_fp_ss[2:-2, 2:-2] - 1e-9 <= vs.S_ufc_ss[2:-2, 2:-2]) & (vs.S_lp_ss[2:-2, 2:-2] - 1e-9 <= vs.S_ac_ss[2:-2, 2:-2])
rows = npx.where(check4 == False)[0]
dS = vs.S[2:-2, 2:-2, vs.tau] - vs.S[2:-2, 2:-2, vs.taum1]
dF = vs.prec[2:-2, 2:-2, vs.tau] - vs.q_sur[2:-2, 2:-2] - vs.aet[2:-2, 2:-2] - vs.q_ss[2:-2, 2:-2] - vs.q_sub[2:-2, 2:-2]

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


from roger import logger
logger.add("out.log")



if rs.loglevel == 'debug' and rs.backend == 'numpy':
    check11 = npx.isclose(npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1),
                                            vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2] -
                                            npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2), atol=settings.atol)
    check22 = npx.isclose(npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.tau] - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.taum1],
                                            npx.sum(vs.inf_mat_rz[2:-2, 2:-2] * vs.tt_inf_mat_rz[2:-2, 2:-2, :] * npx.where(npx.isnan(vs.C_inf_mat_rz[2:-2, 2:-2]), 0, vs.C_inf_mat_rz[2:-2, 2:-2]) + vs.inf_pf_rz[2:-2, 2:-2] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] * npx.where(npx.isnan(vs.C_inf_pf_rz[2:-2, 2:-2]), 0, vs.C_inf_pf_rz[2:-2, 2:-2]) + vs.inf_pf_ss[2:-2, 2:-2] * vs.tt_inf_pf_rz[2:-2, 2:-2, :] * npx.where(npx.isnan(vs.C_inf_pf_ss[2:-2, 2:-2]), 0, vs.C_inf_pf_ss[2:-2, 2:-2]) -
                                            vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :] * npx.where(npx.isnan(vs.C_evap_soil[2:-2, 2:-2]), 0, vs.C_evap_soil[2:-2, 2:-2]) - vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :] * npx.where(npx.isnan(vs.C_transp[2:-2, 2:-2]), 0, vs.C_transp[2:-2, 2:-2]) - vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :] * npx.where(npx.isnan(vs.C_q_ss[2:-2, 2:-2]), 0, vs.C_q_ss[2:-2, 2:-2]), axis=2), atol=settings.atol)
    check33 = (npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) <= (vs.S_sat_rz + vs.S_sat_ss) - (vs.S_pwp_rz + vs.S_pwp_ss)) & (npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) >= 0)

    if not check11.all():
        logger.debug(f"water balance diverged at iteration {vs.itt}")
        rows11 = npx.where(check11 == False)[0].tolist()
        logger.debug(f"Water balance diverged at {rows11}")
        dS = npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1)
        dS_rz = npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_rz[2:-2, 2:-2, vs.taum1, :], axis=-1)
        dS_ss = npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1) - npx.sum(vs.sa_ss[2:-2, 2:-2, vs.taum1, :], axis=-1)
        fluxes = vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2] - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)
        fluxes_rz = vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + npx.sum(vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :], axis=2) - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) - npx.sum(vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :], axis=2)
        fluxes_ss = vs.inf_pf_ss[2:-2, 2:-2] + npx.sum(vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_q_rz[2:-2, 2:-2, :], axis=2) - npx.sum(vs.cpr_rz[2:-2, 2:-2, npx.newaxis] * vs.tt_cpr_rz[2:-2, 2:-2, :], axis=2) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2)
        logger.debug(f"dS: {dS[0][0]}; flux: {fluxes[0][0]}")
        logger.debug(f"dS_rz: {dS_rz[0][0]}; flux_rz: {fluxes_rz[0][0]}")
        logger.debug(f"dS_ss: {dS_ss[0][0]}; flux_ss: {fluxes_ss[0][0]}")
    if not check22.all():
        logger.debug(f"solute balance diverged at iteration {vs.itt}")
        rows22 = npx.where(check22 == False)[0].tolist()
        logger.debug(f"Solute balance diverged at {rows22}")
        dM = npx.sum(npx.where(npx.isnan(vs.msa_s[2:-2, 2:-2, vs.tau, :]), 0, vs.msa_s[2:-2, 2:-2, vs.tau, :]), axis=-1) - npx.sum(npx.where(npx.isnan(vs.msa_s[2:-2, 2:-2, vs.taum1, :]), 0, vs.msa_s[2:-2, 2:-2, vs.taum1, :]), axis=-1)
        dM_rz = npx.sum(npx.where(npx.isnan(vs.msa_rz[2:-2, 2:-2, vs.tau, :]), 0, vs.msa_rz[2:-2, 2:-2, vs.tau, :]), axis=-1) - npx.sum(npx.where(npx.isnan(vs.msa_rz[2:-2, 2:-2, vs.taum1, :]), 0, vs.msa_rz[2:-2, 2:-2, vs.taum1, :]), axis=-1)
        dM_ss = npx.sum(npx.where(npx.isnan(vs.msa_ss[2:-2, 2:-2, vs.tau, :]), 0, vs.msa_ss[2:-2, 2:-2, vs.tau, :]), axis=-1) - npx.sum(npx.where(npx.isnan(vs.msa_ss[2:-2, 2:-2, vs.taum1, :]), 0, vs.msa_ss[2:-2, 2:-2, vs.taum1, :]), axis=-1)
        mfluxes = npx.sum(npx.where(npx.isnan(vs.mtt_inf_mat_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_mat_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_inf_pf_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_inf_pf_ss[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_ss[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_evap_soil[2:-2, 2:-2, :]), 0, vs.mtt_evap_soil[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_transp[2:-2, 2:-2, :]), 0, vs.mtt_transp[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_q_ss[2:-2, 2:-2, :]), 0, vs.mtt_q_ss[2:-2, 2:-2, :]), axis=-1)
        mfluxes_rz = npx.sum(npx.where(npx.isnan(vs.mtt_inf_mat_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_mat_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_inf_pf_rz[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_rz[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_cpr_rz[2:-2, 2:-2, :]), 0, vs.mtt_cpr_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_evap_soil[2:-2, 2:-2, :]), 0, vs.mtt_evap_soil[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_transp[2:-2, 2:-2, :]), 0, vs.mtt_transp[2:-2, 2:-2, :]) -npx.where(npx.isnan(vs.mtt_q_rz[2:-2, 2:-2, :]), 0, vs.mtt_q_rz[2:-2, 2:-2, :]), axis=-1)
        mfluxes_ss = npx.sum(npx.where(npx.isnan(vs.mtt_inf_pf_ss[2:-2, 2:-2, :]), 0, vs.mtt_inf_pf_ss[2:-2, 2:-2, :]) + npx.where(npx.isnan(vs.mtt_q_rz[2:-2, 2:-2, :]), 0, vs.mtt_q_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_cpr_rz[2:-2, 2:-2, :]), 0, vs.mtt_cpr_rz[2:-2, 2:-2, :]) - npx.where(npx.isnan(vs.mtt_q_ss[2:-2, 2:-2, :]), 0, vs.mtt_q_ss[2:-2, 2:-2, :]), axis=-1)
        dC = npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.tau] - npx.sum(vs.sa_s[2:-2, 2:-2, vs.taum1, :], axis=-1) * vs.C_s[2:-2, 2:-2, vs.taum1]
        cfluxes = vs.inf_mat_rz[2:-2, 2:-2] * npx.where(npx.isnan(vs.C_inf_mat_rz[2:-2, 2:-2]), 0, vs.C_inf_mat_rz[2:-2, 2:-2]) + vs.inf_pf_rz[2:-2, 2:-2] * npx.where(npx.isnan(vs.C_inf_pf_rz[2:-2, 2:-2]), 0, vs.C_inf_pf_rz[2:-2, 2:-2]) + vs.inf_pf_ss[2:-2, 2:-2] * npx.where(npx.isnan(vs.C_inf_pf_ss[2:-2, 2:-2]), 0, vs.C_inf_pf_ss[2:-2, 2:-2]) - npx.sum(vs.evap_soil[2:-2, 2:-2, npx.newaxis] * vs.tt_evap_soil[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_evap_soil[2:-2, 2:-2]), 0, vs.C_evap_soil[2:-2, 2:-2]) - npx.sum(vs.transp[2:-2, 2:-2, npx.newaxis] * vs.tt_transp[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_transp[2:-2, 2:-2]), 0, vs.C_transp[2:-2, 2:-2]) - npx.sum(vs.q_ss[2:-2, 2:-2, npx.newaxis] * vs.tt_q_ss[2:-2, 2:-2, :], axis=2) * npx.where(npx.isnan(vs.C_q_ss[2:-2, 2:-2]), 0, vs.C_q_ss[2:-2, 2:-2])
        logger.debug(f"dC: {dC[0][0]}; cflux: {cfluxes[0][0]}")
        logger.debug(f"dM: {dM[0][0]}; mflux: {mfluxes[0][0]}")
        logger.debug(f"dM_rz: {dM_rz[0][0]}; mflux_rz: {mfluxes_rz[0][0]}")
        logger.debug(f"dM_ss: {dM_ss[0][0]}; mflux_ss: {mfluxes_ss[0][0]}")
    if not check33.all():
        logger.debug(f"StorAge is out of bounds at iteration {vs.itt}")


tms = transport_model_structure.replace("_", " ")
model = SVATTRANSPORTSetup()
model._set_tm_structure(tms)
identifier = f'SVATTRANSPORT_{transport_model_structure}1'
model._set_identifier(identifier)
model._sample_params(nsamples)
rows = [12, 13, 14]
params = model._params[rows, :]
model._params = params
model._nrows = len(rows)
input_path = model._base_path / "input"
model._set_input_dir(input_path)
write_forcing_tracer(input_path, 'd18O')
model.setup()
model.warmup()
model.run()
return

tms = transport_model_structure.replace("_", " ")
model = SVATTRANSPORTSetup()
restart_file = "SVATTRANSPORT_preferential.warmup_restart.h5"
model = SVATTRANSPORTSetup(override=dict(
        restart_input_filename=restart_file,
    ))
model._set_tm_structure(tms)
identifier = f'SVATTRANSPORT_{transport_model_structure}1'
model._set_identifier(identifier)
model._sample_params(nsamples)
rows = [12, 13, 14]
params = model._params[rows, :]
model._params = params
model._nrows = len(rows)
input_path = model._base_path / "input"
model._set_input_dir(input_path)
write_forcing_tracer(input_path, 'd18O')
model.setup()
model.run()
return


with self.state.settings.unlock():
    restart_file = self.state.settings.restart_output_filename
    self.state.settings.restart_output_filename = f'{self.state.settings.identifier}.warmup_restart.h5'
restart.write_restart(self.state, force=True)
with self.state.settings.unlock():
    self.state.settings.restart_output_filename = restart_file

vs = state.variables
sa_rz1 = vs.sa_rz[2:-2, 2:-2, 1, :]
sa_ss1 = vs.sa_ss[2:-2, 2:-2, 1, :]
sa_s1 = vs.sa_s[2:-2, 2:-2, 1, :]
msa_rz1 = vs.msa_rz[2:-2, 2:-2, 1, :]
msa_ss1 = vs.msa_ss[2:-2, 2:-2, 1, :]
csa_rz1 = vs.csa_rz[2:-2, 2:-2, 1, :]
csa_ss1 = vs.csa_ss[2:-2, 2:-2, 1, :]
msa_s1 = vs.msa_s[2:-2, 2:-2, 1, :]
ca_rz1 = vs.msa_rz[2:-2, 2:-2, 1, :] / vs.sa_rz[2:-2, 2:-2, 1, :]
ca_ss1 = vs.msa_ss[2:-2, 2:-2, 1, :] / vs.sa_ss[2:-2, 2:-2, 1, :]
cfluxa = mtt[2:-2, 2:-2, :] / (tt[2:-2, 2:-2, :] * flux[2:-2, 2:-2, npx.newaxis])
SA_rz1 = vs.SA_rz[2:-2, 2:-2, 1, :]
SA_ss1 = vs.SA_ss[2:-2, 2:-2, 1, :]

sa_rz2 = vs.sa_rz[2:-2, 2:-2, 1, :]
sa_ss2 = vs.sa_ss[2:-2, 2:-2, 1, :]
msa_rz2 = vs.msa_rz[2:-2, 2:-2, 1, :]
msa_ss2 = vs.msa_ss[2:-2, 2:-2, 1, :]

cond1 = (sa_rz1 <= 0) & (msa_rz1 =! 0)
cond2 = (sa_rz1 > 0) & (msa_rz1 == 0)
cond3 = (sa_ss1 <= 0) & (msa_ss1 =! 0)
cond4 = (sa_ss1 > 0) & (msa_ss1 == 0)


msa_rz0 = msan_rz[2:-2, 2:-2, 0, :]
msa_ss0 = msan_ss[2:-2, 2:-2, 0, :]

tt_transp = vs.tt_transp[2:-2, 2:-2, :]
mtt_transp = vs.mtt_transp[2:-2, 2:-2, :]


with model.state.settings.unlock():
    model.state.settings.warmup_done = True


# local benchmark
python run_benchmarks.py --sizes 1000. --sizes 10000. --sizes 100000. --sizes 200000. --backends numpy --backends jax --backends numpy-mpi --backends jax-mpi --nproc 2 --only oneD_benchmark.py --debug --local


python run_benchmarks.py --sizes 10000. --backends numpy --backends numpy-mpi --nproc 4 --only SVAT_benchmark.py --debug --local

python run_benchmarks.py --sizes 50000. --backends numpy --backends numpy-mpi --nproc 2 --only DUMMY_benchmark.py --debug --local

python run_benchmarks.py --sizes 1000 --backends numpy --backends numpy-mpi --nproc 4 --only SVATOXYGEN18_benchmark.py --debug --local

python run_benchmarks.py --sizes 1000. --backends numpy --backends numpy-mpi --nproc 4 --only SVAT_benchmark.py --debug --local

python run_benchmarks.py --sizes 4. --sizes 12. --sizes 100. --sizes 1000. --sizes 10000. --sizes 100000. --backends numpy --backends jax --backends numpy-mpi --backends jax-mpi --nproc 4 --pmem 4000 --only SVAT_benchmark.py --debug --local
python run_benchmarks.py --sizes 4. --sizes 12. --sizes 100. --sizes 500. --sizes 1000. --backends numpy --backends jax --backends numpy-mpi --backends jax-mpi --nproc 4 --pmem 4000 --only SVATOXYGEN18_benchmark.py --timesteps 20 --debug --local



python run_benchmarks.py --sizes 100. --sizes 1000. --sizes 10000. --sizes 100000. --backends numpy --backends jax --backends numpy-mpi --backends jax-mpi --backends jax-gpu --nproc 20 --pmem 4000 --only SVATOXYGEN18_benchmark.py --debug


python run_benchmarks.py --sizes 100000. --backends numpy-mpi --backends jax-mpi --nproc 4 --pmem 32000 --only SVATOXYGEN18_benchmark.py --debug
python run_benchmarks.py --sizes 100000. --backends numpy-mpi --backends jax-mpi --nproc 8 --pmem 16000 --only SVATOXYGEN18_benchmark.py --debug
python run_benchmarks.py --sizes 100000. --backends numpy-mpi --backends jax-mpi --nproc 16 --pmem 8000 --only SVATOXYGEN18_benchmark.py --debug
python run_benchmarks.py --sizes 100000. --backends numpy-mpi --backends jax-mpi --nproc 28 --pmem 4000 --only SVATOXYGEN18_benchmark.py --debug
python run_benchmarks.py --sizes 100000. --backends numpy-mpi --backends jax-mpi --nproc 56 --pmem 4000 --only SVATOXYGEN18_benchmark.py --debug
python run_benchmarks.py --sizes 100000. --backends numpy-mpi --backends jax-mpi --nproc 112 --pmem 4000 --only SVATOXYGEN18_benchmark.py --debug


from roger.tools.make_toy_data import make_toy_forcing_tracer
make_toy_forcing_tracer(model._base_path, tracer="Br", start_date='11/01/2021', ndays=365)
make_toy_forcing_tracer(model._base_path, tracer="d18O", start_date='11/01/2021', ndays=365)
make_toy_forcing_tracer(model._base_path, tracer="d18O", start_date='11/01/2019', ndays=1096)
make_toy_forcing_tracer(model._base_path, tracer="d18O", start_date='11/01/2017', ndays=1826)
make_toy_forcing_tracer(model._base_path, tracer="d18O", start_date='11/01/2012', ndays=3652)
make_toy_forcing_tracer(model._base_path, tracer="d18O", start_date='11/01/1994', ndays=10227)


mpirun -n 4 python -m pytest distributed_test.py

def _bfill_3d(self, state, arr):
    idx_shape = tuple([slice(None)] + [npx.newaxis] * (3 - 2 - 1))
    idx = allocate(state.dimensions, ("x", "y", "t"), dtype=int)
    arr1 = allocate(state.dimensions, ("x", 1, 1), dtype=int)
    arr2 = allocate(state.dimensions, (1, "y", 1), dtype=int)
    arr3 = allocate(state.dimensions, ("x", "y", "t"), dtype=int)
    arr_fill = allocate(state.dimensions, ("x", "y", "t"))
    idx = update(
        idx,
        at[2:-2, 2:-2, :], npx.where(npx.isfinite(arr), npx.arange(npx.shape(arr)[2])[idx_shape], 0)[2:-2, 2:-2, :],
    )
    idx = update(
        idx,
        at[2:-2, 2:-2, :], _bfill(idx)[2:-2, 2:-2, :],
    )
    arr1 = update(
        arr1,
        at[:, 0, 0], npx.arange(npx.shape(arr)[0]),
    )
    arr2 = update(
        arr2,
        at[0, :, 0], npx.arange(npx.shape(arr)[1]),
    )
    arr3 = update(
        arr3,
        at[:, :, :], idx,
    )
    arr_fill = update(
        arr_fill,
        at[:, :, :], arr[arr1, arr2, arr3],
    )

    return arr_fill


@roger_kernel
def _bfill(loop_arr):
    def loop_body(i, loop_arr):
        j = loop_arr.shape[2] - i
        loop_arr = update(
            loop_arr,
            at[:, :, j-1], npx.where(loop_arr[:, :, j-1] == 0, loop_arr[:, :, j], loop_arr[:, :, j-1]),
        )

        return loop_arr

    loop_arr = for_loop(1, loop_arr.shape[2], loop_body, loop_arr)
    loop_arr = update(
        loop_arr,
        at[:, :, -1], npx.where(loop_arr[:, :, -1] == 0, loop_arr[:, :, -2], loop_arr[:, :, -1]),
    )

    return loop_arr
