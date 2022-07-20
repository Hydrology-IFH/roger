from pathlib import Path
import os
import h5netcdf
import numpy as onp
from roger.cli.roger_run_base import roger_base_cli


@roger_base_cli
def main():
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, update_add, at, for_loop
    from roger.core.utilities import _get_row_no
    import roger.lookuptables as lut
    from roger.tools.setup import write_forcing

    class SVATFILMSetup(RogerSetup):
        """A SVAT model including gravity-driven infiltration and percolation.
        """
        _base_path = Path(__file__).parent
        _input_dir = None

        def _set_input_dir(self, path):
            if os.path.exists(path):
                self._input_dir = path
            else:
                self._input_dir = path
                if not os.path.exists(self._input_dir):
                    os.mkdir(self._input_dir)

        def _read_var_from_nc(self, var, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables[var]
                return npx.array(var_obj)

        def _get_nittevent(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['nitt_event']
                return onp.int32(onp.array(var_obj)[0])

        def _get_nitt(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['Time']
                return len(onp.array(var_obj))

        def _get_runlen(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['Time']
                return onp.array(var_obj)[-1] * 60 * 60

        def _get_nevent_ff(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['nevent_ff']
                return onp.int32(onp.array(var_obj)[0])

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = "SVATFILM"

            settings.nx, settings.ny, settings.nz = 1, 1, 1
            settings.nitt = self._get_nitt(self._input_dir, 'forcing.nc')
            settings.nittevent = self._get_nittevent(self._input_dir, 'forcing.nc')
            settings.nittevent_p1 = settings.nittevent + 1
            settings.runlen = self._get_runlen(self._input_dir, 'forcing.nc')
            settings.nittevent_ff = 5 * 24 * 6
            settings.nittevent_ff_p1 = settings.nittevent_ff + 1
            settings.nevent_ff = self._get_nevent_ff(self._input_dir, 'forcing.nc')

            settings.dx = 1
            settings.dy = 1
            settings.dz = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0

            settings.enable_film_flow = True
            settings.enable_macropore_lower_boundary_condition = False

            settings.ff_tc = 0.15

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "DT_SECS",
                "DT",
                "YEAR",
                "MONTH",
                "DOY",
                "dt_secs",
                "dt",
                "year",
                "month",
                "doy",
                "t",
                "itt",
                "x",
                "y",
            ],
        )
        def set_grid(self, state):
            vs = state.variables

            # temporal grid
            vs.DT_SECS = update(vs.DT_SECS, at[:], self._read_var_from_nc("dt", self._input_dir, 'forcing.nc'))
            vs.DT = update(vs.DT, at[:], vs.DT_SECS / (60 * 60))
            vs.YEAR = update(vs.YEAR, at[:], self._read_var_from_nc("year", self._input_dir, 'forcing.nc'))
            vs.MONTH = update(vs.MONTH, at[:], self._read_var_from_nc("month", self._input_dir, 'forcing.nc'))
            vs.DOY = update(vs.DOY, at[:], self._read_var_from_nc("doy", self._input_dir, 'forcing.nc'))
            vs.dt_secs = vs.DT_SECS[vs.itt]
            vs.dt = vs.DT[vs.itt]
            vs.year = vs.YEAR[vs.itt]
            vs.month = vs.MONTH[vs.itt]
            vs.doy = vs.DOY[vs.itt]
            vs.t = update(vs.t, at[:], npx.cumsum(vs.DT))
            # spatial grid
            dx = allocate(state.dimensions, ("x"))
            dx = update(dx, at[:], 1)
            dy = allocate(state.dimensions, ("y"))
            dy = update(dy, at[:], 1)
            vs.x = update(vs.x, at[3:-2], npx.cumsum(dx[3:-2]))
            vs.y = update(vs.y, at[3:-2], npx.cumsum(dy[3:-2]))

        @roger_routine
        def set_look_up_tables(self, state):
            vs = state.variables

            vs.lut_ilu = update(vs.lut_ilu, at[:, :], lut.ARR_ILU)
            vs.lut_gc = update(vs.lut_gc, at[:, :], lut.ARR_GC)
            vs.lut_gcm = update(vs.lut_gcm, at[:, :], lut.ARR_GCM)
            vs.lut_is = update(vs.lut_is, at[:, :], lut.ARR_IS)
            vs.lut_rdlu = update(vs.lut_rdlu, at[:, :], lut.ARR_RDLU)

        @roger_routine
        def set_topography(self, state):
            pass

        @roger_routine
        def set_parameters_setup(self, state):
            vs = state.variables

            vs.lu_id = update(vs.lu_id, at[2:-2, 2:-2], 8)
            vs.slope = update(vs.slope, at[2:-2, 2:-2], 0)
            vs.z_soil = update(vs.z_soil, at[2:-2, 2:-2], 1350)
            vs.dmpv = update(vs.dmpv, at[2:-2, 2:-2], 50)
            vs.lmpv = update(vs.lmpv, at[2:-2, 2:-2], 1000)
            vs.theta_ac = update(vs.theta_ac, at[2:-2, 2:-2], 0.13)
            vs.theta_ufc = update(vs.theta_ufc, at[2:-2, 2:-2], 0.24)
            vs.theta_pwp = update(vs.theta_pwp, at[2:-2, 2:-2], 0.23)
            vs.ks = update(vs.ks, at[2:-2, 2:-2], 25)
            vs.kf = update(vs.kf, at[2:-2, 2:-2], 2500)
            vs.a_ff = update(vs.a_ff, at[2:-2, 2:-2], 0.19)
            vs.c_ff = update(vs.c_ff, at[2:-2, 2:-2], 0.001)

        @roger_routine
        def set_parameters(self, state):
            vs = state.variables

            if (vs.MONTH[vs.itt] != vs.MONTH[vs.itt - 1]) & (vs.itt > 1):
                vs.update(set_parameters_monthly_kernel(state))

        @roger_routine
        def set_initial_conditions_setup(self, state):
            pass

        @roger_routine
        def set_initial_conditions(self, state):
            vs = state.variables

            vs.S_int_top = update(vs.S_int_top, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.swe_top = update(vs.swe_top, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.S_int_ground = update(vs.S_int_ground, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.swe_ground = update(vs.swe_ground, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.S_dep = update(vs.S_dep, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.S_snow = update(vs.S_snow, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.swe = update(vs.swe, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.theta_rz = update(vs.theta_rz, at[2:-2, 2:-2, :vs.taup1], 0.4)
            vs.theta_ss = update(vs.theta_ss, at[2:-2, 2:-2, :vs.taup1], 0.47)

        @roger_routine
        def set_forcing_setup(self, state):
            vs = state.variables

            vs.PREC = update(vs.PREC, at[2:-2, 2:-2, :], self._read_var_from_nc("PREC", self._input_dir, 'forcing.nc'))
            vs.EVENT_ID = update(vs.EVENT_ID, at[2:-2, 2:-2, :], self._read_var_from_nc("EVENT_ID", self._input_dir, 'forcing.nc'))
            vs.EVENT_ID_FF = update(vs.EVENT_ID_FF, at[2:-2, 2:-2, :], self._read_var_from_nc("EVENT_ID_FF", self._input_dir, 'forcing.nc'))

        @roger_routine
        def set_forcing(self, state):
            vs = state.variables
            settings = state.settings

            vs.ta = update(vs.ta, at[2:-2, 2:-2, :], self._read_var_from_nc("TA", self._input_dir, 'forcing.nc')[:, :, vs.itt])
            vs.pet = update(vs.pet, at[2:-2, 2:-2, :], self._read_var_from_nc("PET", self._input_dir, 'forcing.nc')[:, :, vs.itt])
            vs.pet_res = update(vs.pet_res, at[2:-2, 2:-2], vs.pet[2:-2, 2:-2])

            vs.itt_event_ff = update(
                vs.itt_event_ff,
                at[:], vs.itt - vs.event_start_ff,
            )
            arr_itt = allocate(state.dimensions, ("x", "y"))
            arr_itt = update(
                arr_itt,
                at[2:-2, 2:-2], vs.itt,
            )
            cond1 = ((vs.EVENT_ID_FF[2:-2, 2:-2, vs.itt-1] == 0) & (vs.EVENT_ID_FF[2:-2, 2:-2, vs.itt] >= 1) & (arr_itt >= 1))
            if cond1.any():
                vs.event_no_ff = vs.event_no_ff + 1
                # number of event
                vs.event_id_ff = npx.max(vs.EVENT_ID_FF[2:-2, 2:-2, vs.itt])
                # iteration at event start
                vs.event_start_ff = update(vs.event_start_ff, at[vs.event_no_ff - 1], vs.itt)
                # iteration at event end
                vs.event_end_ff = update(vs.event_end_ff, at[vs.event_no_ff - 1], vs.itt + settings.nittevent_ff)
                if (vs.event_end_ff[vs.event_no_ff - 1] >= settings.nitt):
                    vs.event_end_ff = update(vs.event_end_ff, at[vs.event_no_ff - 1], settings.nitt)
                vs.rain_event = update(
                    vs.rain_event,
                    at[2:-2, 2:-2, :], 0,
                )
                vs.rain_event = update(
                    vs.rain_event,
                    at[2:-2, 2:-2, 0:vs.event_end_ff[vs.event_no_ff - 1]-vs.event_start_ff[vs.event_no_ff - 1]], npx.where(vs.EVENT_ID_FF == vs.event_id_ff, vs.PREC, 0)[2:-2, 2:-2, vs.event_start_ff[vs.event_no_ff - 1]:vs.event_end_ff[vs.event_no_ff - 1]],
                )
                vs.rain_event_sum = update(
                    vs.rain_event_sum,
                    at[2:-2, 2:-2], npx.sum(vs.rain_event[2:-2, 2:-2, :], axis=-1),
                )
                vs.rain_event_csum = update(
                    vs.rain_event_csum,
                    at[2:-2, 2:-2, :], npx.cumsum(vs.rain_event[2:-2, 2:-2, :], axis=-1),
                )
                # subtract interception storage
                vs.rain_event = update(
                    vs.rain_event,
                    at[2:-2, 2:-2, 0], vs.rain_event[2:-2, 2:-2, 0] - (vs.S_int_top_tot[2:-2, 2:-2] - vs.S_int_top[2:-2, 2:-2, vs.tau]) + (vs.S_int_ground_tot[2:-2, 2:-2] - vs.S_int_ground[2:-2, 2:-2, vs.tau]),
                )
                vs.rain_event = update(
                    vs.rain_event,
                    at[2:-2, 2:-2, 1:], npx.diff(npx.where(npx.cumsum(vs.rain_event[2:-2, 2:-2, :], axis=-1) < 0, 0, npx.cumsum(vs.rain_event[2:-2, 2:-2, :], axis=-1)), axis=-1),
                )
                vs.rain_event = update(
                    vs.rain_event,
                    at[2:-2, 2:-2, 0], npx.where(vs.rain_event[2:-2, 2:-2, 0] < 0, 0, vs.rain_event[2:-2, 2:-2, 0]),
                )

            vs.update(set_forcing_kernel(state))

        @roger_routine
        def set_diagnostics(self, state):
            pass

        @roger_routine
        def after_timestep(self, state):
            vs = state.variables

            vs.update(after_timestep_kernel(state))
            vs.update(after_timestep_film_flow_kernel(state))

    @roger_kernel
    def set_parameters_monthly_kernel(state):
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
                at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.lut_ilu[row_no, vs.month], S_int_top_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
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
                at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.lut_ilu[row_no, vs.month], S_int_ground_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
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
                at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.lut_gc[row_no, vs.month], ground_cover[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
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
                at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.lut_gc[row_no, vs.month] / vs.lut_gcm[row_no, 1], basal_transp_coeff[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
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
                at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], 1 - ((vs.lut_gc[row_no, vs.month] / vs.lut_gcm[row_no, 1]) * vs.lut_gcm[row_no, 1]), basal_evap_coeff[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
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

        vs.lai = update(
            vs.lai,
            at[2:-2, 2:-2], npx.log(1 / (1 - vs.ground_cover[2:-2, 2:-2, vs.tau])) / npx.log(1 / 0.7) * vs.maskCatch[2:-2, 2:-2]
        )

        vs.throughfall_coeff_top = update(
            vs.throughfall_coeff_top,
            at[2:-2, 2:-2], npx.where(npx.isin(vs.lu_id[2:-2, 2:-2], npx.array([10, 11, 12])), npx.where(vs.lai[2:-2, 2:-2] > 1, 0, 1 - vs.lai[2:-2, 2:-2]), 0) * vs.maskCatch[2:-2, 2:-2]
        )

        vs.throughfall_coeff_ground = update(
            vs.throughfall_coeff_ground,
            at[2:-2, 2:-2], npx.where(npx.isin(vs.lu_id[2:-2, 2:-2], npx.arange(500, 598)), npx.where(vs.lai[2:-2, 2:-2] > 1, 0, 1 - vs.lai[2:-2, 2:-2]), 0) * vs.maskCatch[2:-2, 2:-2]
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

    @roger_kernel
    def set_forcing_kernel(state):
        vs = state.variables

        # update precipitation with available interception storage while film flow event
        mask_noff = (vs.EVENT_ID_FF[:, :, vs.itt] == 0)
        vs.prec = update(vs.prec, at[2:-2, 2:-2], 0)
        vs.prec = update_add(vs.prec, at[2:-2, 2:-2],
                             npx.where((vs.EVENT_ID_FF[2:-2, 2:-2, vs.itt] >= 1) & ((vs.S_int_top_tot[2:-2, 2:-2] - vs.S_int_top[2:-2, 2:-2, vs.tau]) + (vs.S_int_ground_tot[2:-2, 2:-2] - vs.S_int_ground[2:-2, 2:-2, vs.tau]) > 0),
                             npx.where(vs.PREC[2:-2, 2:-2, vs.itt] > (vs.S_int_top_tot[2:-2, 2:-2] - vs.S_int_top[2:-2, 2:-2, vs.tau]) + (vs.S_int_ground_tot[2:-2, 2:-2] - vs.S_int_ground[2:-2, 2:-2, vs.tau]), (vs.S_int_top_tot[2:-2, 2:-2] - vs.S_int_top[2:-2, 2:-2, vs.tau]) + (vs.S_int_ground_tot[2:-2, 2:-2] - vs.S_int_ground[2:-2, 2:-2, vs.tau]), vs.PREC[2:-2, 2:-2, vs.itt]), 0))
        vs.prec = update(vs.prec, at[2:-2, 2:-2], npx.where(mask_noff[2:-2, 2:-2], vs.PREC[2:-2, 2:-2, vs.itt], 0))
        vs.ta = update(vs.ta, at[2:-2, 2:-2, vs.tau], vs.TA[2:-2, 2:-2, vs.itt])
        vs.pet = update(vs.pet, at[2:-2, 2:-2], vs.PET[2:-2, 2:-2, vs.itt])
        vs.pet_res = update(vs.pet, at[2:-2, 2:-2], vs.PET[2:-2, 2:-2, vs.itt])

        vs.dt_secs = vs.DT_SECS[vs.itt]
        vs.dt = vs.DT[vs.itt]
        vs.year = vs.YEAR[vs.itt]
        vs.month = vs.MONTH[vs.itt]
        vs.doy = vs.DOY[vs.itt]

        # reset fluxes at beginning of time step
        vs.q_sur = update(
            vs.q_sur,
            at[2:-2, 2:-2], 0,
        )

        vs.q_hof = update(
            vs.q_hof,
            at[2:-2, 2:-2], 0,
        )

        return KernelOutput(
            prec=vs.prec,
            ta=vs.ta,
            pet=vs.pet,
            pet_res=vs.pet_res,
            dt=vs.dt,
            dt_secs=vs.dt_secs,
            year=vs.year,
            month=vs.month,
            doy=vs.doy,
            q_sur=vs.q_sur,
            q_hof=vs.q_hof,
        )

    @roger_kernel
    def after_timestep_kernel(state):
        vs = state.variables

        vs.ta = update(
            vs.ta,
            at[2:-2, 2:-2, vs.taum1], vs.ta[2:-2, 2:-2, vs.tau],
        )
        vs.z_root = update(
            vs.z_root,
            at[2:-2, 2:-2, vs.taum1], vs.z_root[2:-2, 2:-2, vs.tau],
        )
        vs.ground_cover = update(
            vs.ground_cover,
            at[2:-2, 2:-2, vs.taum1], vs.ground_cover[2:-2, 2:-2, vs.tau],
        )
        vs.S_sur = update(
            vs.S_sur,
            at[2:-2, 2:-2, vs.taum1], vs.S_sur[2:-2, 2:-2, vs.tau],
        )
        vs.S_int_top = update(
            vs.S_int_top,
            at[2:-2, 2:-2, vs.taum1], vs.S_int_top[2:-2, 2:-2, vs.tau],
        )
        vs.S_int_ground = update(
            vs.S_int_ground,
            at[2:-2, 2:-2, vs.taum1], vs.S_int_ground[2:-2, 2:-2, vs.tau],
        )
        vs.S_dep = update(
            vs.S_dep,
            at[2:-2, 2:-2, vs.taum1], vs.S_dep[2:-2, 2:-2, vs.tau],
        )
        vs.S_snow = update(
            vs.S_snow,
            at[2:-2, 2:-2, vs.taum1], vs.S_snow[2:-2, 2:-2, vs.tau],
        )
        vs.swe = update(
            vs.swe,
            at[2:-2, 2:-2, vs.taum1], vs.swe[2:-2, 2:-2, vs.tau],
        )
        vs.S_rz = update(
            vs.S_rz,
            at[2:-2, 2:-2, vs.taum1], vs.S_rz[2:-2, 2:-2, vs.tau],
        )
        vs.S_ss = update(
            vs.S_ss,
            at[2:-2, 2:-2, vs.taum1], vs.S_ss[2:-2, 2:-2, vs.tau],
        )
        vs.S_s = update(
            vs.S_s,
            at[2:-2, 2:-2, vs.taum1], vs.S_s[2:-2, 2:-2, vs.tau],
        )
        vs.S = update(
            vs.S,
            at[2:-2, 2:-2, vs.taum1], vs.S[2:-2, 2:-2, vs.tau],
        )
        vs.z_sat = update(
            vs.z_sat,
            at[2:-2, 2:-2, vs.taum1], vs.z_sat[2:-2, 2:-2, vs.tau],
        )
        vs.z_wf = update(
            vs.z_wf,
            at[2:-2, 2:-2, vs.taum1], vs.z_wf[2:-2, 2:-2, vs.tau],
        )
        vs.z_wf_t0 = update(
            vs.z_wf_t0,
            at[2:-2, 2:-2, vs.taum1], vs.z_wf_t0[2:-2, 2:-2, vs.tau],
        )
        vs.z_wf_t1 = update(
            vs.z_wf_t1,
            at[2:-2, 2:-2, vs.taum1], vs.z_wf_t1[2:-2, 2:-2, vs.tau],
        )
        vs.y_mp = update(
            vs.y_mp,
            at[2:-2, 2:-2, vs.taum1], vs.y_mp[2:-2, 2:-2, vs.tau],
        )
        vs.y_sc = update(
            vs.y_sc,
            at[2:-2, 2:-2, vs.taum1], vs.y_sc[2:-2, 2:-2, vs.tau],
        )
        vs.prec_event_sum = update(
            vs.prec_event_sum,
            at[2:-2, 2:-2, vs.taum1], vs.prec_event_sum[2:-2, 2:-2, vs.tau],
        )
        vs.t_event_sum = update(
            vs.t_event_sum,
            at[2:-2, 2:-2, vs.taum1], vs.t_event_sum[2:-2, 2:-2, vs.tau],
        )
        vs.theta_rz = update(
            vs.theta_rz,
            at[2:-2, 2:-2, vs.taum1], vs.theta_rz[2:-2, 2:-2, vs.tau],
        )
        vs.theta_ss = update(
            vs.theta_ss,
            at[2:-2, 2:-2, vs.taum1], vs.theta_ss[2:-2, 2:-2, vs.tau],
        )
        vs.theta = update(
            vs.theta,
            at[2:-2, 2:-2, vs.taum1], vs.theta[2:-2, 2:-2, vs.tau],
        )
        vs.k_rz = update(
            vs.k_rz,
            at[2:-2, 2:-2, vs.taum1], vs.k_rz[2:-2, 2:-2, vs.tau],
        )
        vs.k_ss = update(
            vs.k_ss,
            at[2:-2, 2:-2, vs.taum1], vs.k_ss[2:-2, 2:-2, vs.tau],
        )
        vs.k = update(
            vs.k,
            at[2:-2, 2:-2, vs.taum1], vs.k[2:-2, 2:-2, vs.tau],
        )
        vs.h_rz = update(
            vs.h_rz,
            at[2:-2, 2:-2, vs.taum1], vs.h_rz[2:-2, 2:-2, vs.tau],
        )
        vs.h_ss = update(
            vs.h_ss,
            at[2:-2, 2:-2, vs.taum1], vs.h_ss[2:-2, 2:-2, vs.tau],
        )
        vs.h = update(
            vs.h,
            at[2:-2, 2:-2, vs.taum1], vs.h[2:-2, 2:-2, vs.tau],
        )
        vs.z0 = update(
            vs.z0,
            at[2:-2, 2:-2, vs.taum1], vs.z0[2:-2, 2:-2, vs.tau],
        )

        return KernelOutput(
            ta=vs.ta,
            z_root=vs.z_root,
            ground_cover=vs.ground_cover,
            S_sur=vs.S_sur,
            S_int_top=vs.S_int_top,
            S_int_ground=vs.S_int_ground,
            S_dep=vs.S_dep,
            S_snow=vs.S_snow,
            swe=vs.swe,
            S_rz=vs.S_rz,
            S_ss=vs.S_ss,
            S_s=vs.S_s,
            S=vs.S,
            z_sat=vs.z_sat,
            z_wf=vs.z_wf,
            z_wf_t0=vs.z_wf_t0,
            z_wf_t1=vs.z_wf_t1,
            y_mp=vs.y_mp,
            y_sc=vs.y_sc,
            t_event_sum=vs.t_event_sum,
            prec_event_sum=vs.prec_event_sum,
            theta_rz=vs.theta_rz,
            theta_ss=vs.theta_ss,
            theta=vs.theta,
            h_rz=vs.h_rz,
            h_ss=vs.h_ss,
            h=vs.h,
            k_rz=vs.k_rz,
            k_ss=vs.k_ss,
            k=vs.k,
            z0=vs.z0,
        )

    @roger_kernel
    def after_timestep_film_flow_kernel(state):
        vs = state.variables

        vs.z_wf_ff = update(
            vs.z_wf_ff,
            at[2:-2, 2:-2, :, vs.taum1], vs.z_wf_ff[2:-2, 2:-2, :, vs.tau],
        )
        vs.z_pf_ff = update(
            vs.z_pf_ff,
            at[2:-2, 2:-2, :, vs.taum1], vs.z_pf_ff[2:-2, 2:-2, :, vs.tau],
        )
        vs.theta_rz_ff = update(
            vs.theta_rz_ff,
            at[2:-2, 2:-2, vs.taum1], vs.theta_rz_ff[2:-2, 2:-2, vs.tau],
        )
        vs.theta_ss_ff = update(
            vs.theta_ss_ff,
            at[2:-2, 2:-2, vs.taum1], vs.theta_ss_ff[2:-2, 2:-2, vs.tau],
        )
        vs.theta_ff = update(
            vs.theta_ff,
            at[2:-2, 2:-2, vs.taum1], vs.theta_ff[2:-2, 2:-2, vs.tau],
        )
        vs.z_pf = update(
            vs.z_pf,
            at[2:-2, 2:-2, vs.taum1], vs.z_pf[2:-2, 2:-2, vs.tau],
        )

        return KernelOutput(
            z_wf_ff=vs.z_wf_ff,
            z_pf_ff=vs.z_pf_ff,
            theta_rz_ff=vs.theta_rz_ff,
            theta_ss_ff=vs.theta_ss_ff,
            theta_ff=vs.theta_ff,
            z_pf=vs.z_pf,
        )

    model = SVATFILMSetup()
    path_input = model._base_path / "input"
    model._set_input_dir(path_input)
    write_forcing(path_input, enable_film_flow=True, z_soil=1350, a=0.19)
    model.setup()
    model.run()
    return


if __name__ == "__main__":
    main()
