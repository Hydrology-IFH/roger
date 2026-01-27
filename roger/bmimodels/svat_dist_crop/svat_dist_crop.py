from pathlib import Path
import h5netcdf
import pandas as pd
import os

from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput, runtime_settings
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at
from roger.core.surface import calc_parameters_surface_kernel
import roger.lookuptables as lut
import numpy as onp


class SVATDISTCROPSetup(RogerSetup):
    """A SVAT model including crop phenology/crop rotation."""

    def __init__(self, base_path=Path(), enable_groundwater_boundary=False, enable_soil_compaction=False, enable_irrigation=False):
        super().__init__()
        self._base_path = base_path
        self._input_dir = base_path / "input"
        self._output_dir = base_path / "output"
        self._file_config = base_path / "config_roger.yml"
        self._config = None
        self.enable_groundwater_boundary=enable_groundwater_boundary
        self.enable_soil_compaction=enable_soil_compaction
        self.enable_irrigation=enable_irrigation

    # custom helper functions
    def _read_var_from_nc(self, var, path_dir, file):
        nc_file = path_dir / file
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables[var]
            return npx.array(var_obj)

    def _read_var_from_csv(self, var, path_dir, file):
        csv_file = path_dir / file
        infile = pd.read_csv(csv_file, sep=";", skiprows=1, na_values=['', -9999, -9999.0])
        var_obj = infile.loc[:, var]
        if var == "lu_id":
            vals = npx.array(var_obj, dtype=runtime_settings.int_type)[:, npx.newaxis]
        else:
            vals = npx.array(var_obj)[:, npx.newaxis]
        return vals

    def _get_runlen(self, path_dir, file):
        nc_file = path_dir / file
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables["dt"]
            return onp.sum(onp.array(var_obj))

    def _get_time_origin(self, path_dir, file):
        nc_file = path_dir / file
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables["Time"].attrs["time_origin"]
            return str(var_obj)

    @roger_routine
    def set_settings(self, state):
        settings = state.settings
        settings.identifier = self._config["identifier"]

        # output frequency (in seconds)
        settings.output_frequency = self._config["OUTPUT_FREQUENCY"]
        # total grid numbers in x- and y-direction
        settings.nx, settings.ny = self._config["nx"], self._config["ny"]
        # derive total number of time steps from forcing
        settings.runlen = self._get_runlen(self._input_dir, "forcing.nc")
        settings.nitt_forc = len(self._read_var_from_nc("Time", self._input_dir, "forcing.nc"))
        station_ids = onp.unique(self._read_var_from_csv("STAT_ID", self._base_path, "parameters_roger.csv"))
        station_ids = station_ids[~onp.isnan(station_ids)]
        station_ids = station_ids[station_ids != -9999]
        settings.nstations = len(station_ids)

        # spatial discretization (in meters)
        settings.dx = self._config["dx"]
        settings.dy = self._config["dy"]

        # origin of spatial grid
        settings.x_origin = self._config["x_origin"]
        settings.y_origin = self._config["y_origin"]
        # origin of time steps (e.g. 01-01-2023)
        settings.time_origin = self._get_time_origin(self._input_dir, "forcing.nc")

        # enable specific processes
        settings.enable_macropore_lower_boundary_condition = False
        settings.enable_crop_water_stress = True
        settings.enable_crop_phenology = True
        settings.enable_crop_rotation = True
        settings.enable_macropore_lower_boundary_condition = False
        settings.enable_adaptive_time_stepping = True
        settings.enable_distributed_input = True
        settings.enable_soil_compaction = self.enable_soil_compaction
        settings.enable_groundwater_boundary = self.enable_groundwater_boundary
        settings.enable_irrigation = self.enable_irrigation
        if settings.enable_irrigation:
            settings.enable_net_irrigation = True
            settings.enable_crop_specific_irrigation_demand = True

        if settings.enable_crop_rotation:
            settings.ncrops = 3
            settings.ncr = 3

    @roger_routine
    def read_data(self, state):
        pass

    @roger_routine
    def set_grid(self, state):
        vs = state.variables
        settings = state.settings

        # spatial grid
        dx = allocate(state.dimensions, ("x"))
        dx = update(dx, at[:], settings.dx)
        dy = allocate(state.dimensions, ("y"))
        dy = update(dy, at[:], settings.dy)
        # distance from origin
        vs.x = update(vs.x, at[3:-2], settings.x_origin + npx.cumsum(dx[3:-2]))
        vs.y = update(vs.y, at[3:-2], settings.y_origin + npx.cumsum(dy[3:-2]))

    @roger_routine
    def set_look_up_tables(self, state):
        vs = state.variables

        vs.lut_ilu = update(vs.lut_ilu, at[:, :], lut.ARR_ILU)
        vs.lut_gc = update(vs.lut_gc, at[:, :], lut.ARR_GC)
        vs.lut_gcm = update(vs.lut_gcm, at[:, :], lut.ARR_GCM)
        vs.lut_is = update(vs.lut_is, at[:, :], lut.ARR_IS)
        vs.lut_rdlu = update(vs.lut_rdlu, at[:, :], lut.ARR_RDLU)
        vs.lut_crops = update(vs.lut_crops, at[:, :], lut.ARR_CP)

    @roger_routine
    def set_topography(self, state):
        vs = state.variables
        settings = state.settings

        # catchment mask (bool)
        z_soil = self._read_var_from_csv("z_soil", self._base_path, "parameters_roger.csv").reshape(settings.nx, settings.ny)
        vs.maskCatch = update(
            vs.maskCatch,
            at[2:-2, 2:-2],
            onp.isfinite(z_soil),
        )

    @roger_routine
    def set_parameters_setup(self, state):
        vs = state.variables
        settings = state.settings

        vs.lu_id = update(
            vs.lu_id,
            at[2:-2, 2:-2],
            self._read_var_from_csv("lu_id", self._base_path, "parameters_roger.csv").reshape(settings.nx, settings.ny),
        )

        vs.year = update(
            vs.year, at[1], self._read_var_from_nc("YEAR", self._input_dir, "forcing.nc")[vs.itt_forc]
        )
        vs.crop_type = update(
            vs.crop_type,
            at[2:-2, 2:-2, :],
            npx.where(vs.lu_id[2:-2, 2:-2] == 5, 599, 598)[:, :, npx.newaxis],
        )
        vs.crop_type = update(
            vs.crop_type,
            at[2:-2, 2:-2, 1],
            self._read_var_from_csv(f"{vs.year[1]}_summer", self._base_path, "crop_rotations.csv").reshape(settings.nx, settings.ny),
        )
        vs.crop_type = update(
            vs.crop_type,
            at[2:-2, 2:-2, 1],
            self._read_var_from_csv(f"{vs.year[1]}_winter", self._base_path, "crop_rotations.csv").reshape(settings.nx, settings.ny),
        )
        vs.lu_id = update(
            vs.lu_id,
            at[2:-2, 2:-2],
            npx.where(vs.lu_id[2:-2, 2:-2] == 5, vs.crop_type[2:-2, 2:-2, 0], vs.lu_id[2:-2, 2:-2]),
        )

        vs.sealing = update(
            vs.sealing,
            at[2:-2, 2:-2],
            self._read_var_from_csv("sealing", self._base_path, "parameters_roger.csv").reshape(settings.nx, settings.ny),
        )
        vs.z_soil = update(
            vs.z_soil,
            at[2:-2, 2:-2],
            self._read_var_from_csv("z_soil", self._base_path, "parameters_roger.csv").reshape(settings.nx, settings.ny),
        )
        vs.dmpv = update(
            vs.dmpv,
            at[2:-2, 2:-2],
            self._read_var_from_csv("dmpv", self._base_path, "parameters_roger.csv").reshape(settings.nx, settings.ny),
        )
        vs.lmpv = update(
            vs.lmpv,
            at[2:-2, 2:-2],
            self._read_var_from_csv("lmpv", self._base_path, "parameters_roger.csv").reshape(settings.nx, settings.ny),
        )
        vs.theta_ac = update(
            vs.theta_ac,
            at[2:-2, 2:-2],
            self._read_var_from_csv("theta_ac", self._base_path, "parameters_roger.csv").reshape(settings.nx, settings.ny),
        )
        vs.theta_ufc = update(
            vs.theta_ufc,
            at[2:-2, 2:-2],
            self._read_var_from_csv("theta_ufc", self._base_path, "parameters_roger.csv").reshape(settings.nx, settings.ny),
        )
        vs.theta_pwp = update(
            vs.theta_pwp,
            at[2:-2, 2:-2],
            self._read_var_from_csv("theta_pwp", self._base_path, "parameters_roger.csv").reshape(settings.nx, settings.ny),
        )
        vs.ks = update(
            vs.ks,
            at[2:-2, 2:-2],
            self._read_var_from_csv("ks", self._base_path, "parameters_roger.csv").reshape(settings.nx, settings.ny),
        )
        vs.kf = update(
            vs.kf,
            at[2:-2, 2:-2],
            self._read_var_from_csv("kf", self._base_path, "parameters_roger.csv").reshape(settings.nx, settings.ny),
        )
        vs.ta_offset = update(
            vs.ta_offset,
            at[2:-2, 2:-2],
            self._read_var_from_csv("ta_offset", self._base_path, "parameters_roger.csv").reshape(settings.nx, settings.ny),
        )
        vs.pet_weight = update(
            vs.pet_weight,
            at[2:-2, 2:-2],
            self._read_var_from_csv("pet_weight", self._base_path, "parameters_roger.csv").reshape(settings.nx, settings.ny),
        )
        vs.prec_weight = update(
            vs.prec_weight,
            at[2:-2, 2:-2],
            self._read_var_from_csv("prec_weight", self._base_path, "parameters_roger.csv").reshape(settings.nx, settings.ny),
        )
        # identifier of meteorological station
        vs.station_id = update(
            vs.station_id,
            at[2:-2, 2:-2],
            self._read_var_from_csv("STAT_ID", self._base_path, "parameters_roger.csv").reshape(settings.nx, settings.ny),
        )

        station_ids = [int(item) for item in os.listdir(self._base_path / "input" / "meteo_stations") if item != ".DS_Store"]
        vs.station_ids = update(
            vs.station_ids,
            at[:],
            station_ids,
        )
        if settings.enable_soil_compaction:
            list_crops = npx.arange(500, 598).tolist() + [599]
            mask_crops = npx.isin(vs.lu_id, npx.array(list_crops))
            # represent soil compaction by reducing ks and air capacity of subsoil
            vs.ks_ss = update(vs.ks_ss, at[2:-2, 2:-2], npx.where(mask_crops[2:-2, 2:-2], vs.ks[2:-2, 2:-2] * 0.2, vs.ks[2:-2, 2:-2]))  # reduce ks by an order of magnitude
            # reduce air capacity of subsoil to represent soil compaction
            # Mossadeghi-Björklund et al. (2019) Equation in Figure 3
            vs.theta_ac_ss = update(
                vs.theta_ac_ss, at[2:-2, 2:-2], npx.where(mask_crops[2:-2, 2:-2], (npx.log(vs.ks_ss[2:-2, 2:-2]/10)+0.61)/13.92, vs.theta_ac_ss[2:-2, 2:-2])
            )
            vs.theta_ac_ss = update(
                vs.theta_ac_ss, at[2:-2, 2:-2], npx.where(vs.theta_ac_ss[2:-2, 2:-2] > vs.theta_ac[2:-2, 2:-2], vs.theta_ac[2:-2, 2:-2], vs.theta_ac_ss[2:-2, 2:-2])
            )
            vs.theta_ac_ss = update(
                vs.theta_ac_ss, at[2:-2, 2:-2], npx.where(vs.theta_ac_ss[2:-2, 2:-2] <= 0, 0.02, vs.theta_ac_ss[2:-2, 2:-2])
            )

    @roger_routine
    def set_parameters(self, state):
        vs = state.variables

        if (vs.month[vs.tau] != vs.month[vs.taum1]) & (vs.itt > 1):
            vs.update(calc_parameters_surface_kernel(state))

    @roger_routine
    def set_initial_conditions_setup(self, state):
        pass

    @roger_routine
    def set_initial_conditions(self, state):
        vs = state.variables

        vs.theta_rz = update(
            vs.theta_rz,
            at[2:-2, 2:-2, : vs.taup1],
            vs.theta_pwp[2:-2, 2:-2, npx.newaxis] + vs.theta_ufc[2:-2, 2:-2, npx.newaxis],
        )
        vs.theta_ss = update(
            vs.theta_ss,
            at[2:-2, 2:-2, : vs.taup1],
            vs.theta_pwp[2:-2, 2:-2, npx.newaxis] + vs.theta_ufc[2:-2, 2:-2, npx.newaxis],
        )

        vs.update(set_initial_conditions_crops_kernel(state))

    @roger_routine
    def set_boundary_conditions_setup(self, state):
        pass

    @roger_routine
    def set_boundary_conditions(self, state):
        pass

    @roger_routine(
        dist_safe=False,
        local_variables=[
            "PREC_DIST",
            "TA_DIST",
            "TA_MIN_DIST",
            "TA_MAX_DIST",
            "PET_DIST",
            "YEAR",
            "MONTH",
            "DOY",
        ],
    )
    def set_forcing_setup(self, state):
        vs = state.variables

        vs.PREC_DIST = update(vs.PREC_DIST, at[:, :], self._read_var_from_nc("PREC", self._input_dir, 'forcing.nc'))
        vs.TA_DIST = update(vs.TA_DIST, at[:, :], self._read_var_from_nc("TA", self._input_dir, 'forcing.nc'))
        vs.TA_MIN_DIST = update(vs.TA_MIN_DIST, at[:, :], self._read_var_from_nc("TA_min", self._input_dir, 'forcing.nc'))
        vs.TA_MAX_DIST = update(vs.TA_MAX_DIST, at[:, :], self._read_var_from_nc("TA_max", self._input_dir, 'forcing.nc'))
        vs.PET_DIST = update(vs.PET_DIST, at[:, :], self._read_var_from_nc("PET", self._input_dir, 'forcing.nc'))
        vs.YEAR = update(
            vs.YEAR, at[:], self._read_var_from_nc("YEAR", self._input_dir, "forcing.nc")
        )
        vs.MONTH = update(
            vs.MONTH, at[:], self._read_var_from_nc("MONTH", self._input_dir, "forcing.nc")
        )
        vs.DOY = update(
            vs.DOY, at[:], self._read_var_from_nc("DOY", self._input_dir, "forcing.nc")
        )

    @roger_routine
    def set_forcing(self, state):
        vs = state.variables
        settings = state.settings

        if settings.enable_irrigation:
            vs.irrig = update(vs.irrig, at[2:-2, 2:-2], 0)

        condt = vs.time % (24 * 60 * 60) == 0
        if condt:
            precip = allocate(state.dimensions, ("x", "y", "timesteps_day"))
            ta = allocate(state.dimensions, ("x", "y", "timesteps_day"))
            ta_min = allocate(state.dimensions, ("x", "y", "timesteps_day"))
            ta_max = allocate(state.dimensions, ("x", "y", "timesteps_day"))
            pet = allocate(state.dimensions, ("x", "y", "timesteps_day"))
            for i, ii in enumerate(vs.station_ids):
                mask = (vs.station_id == ii)
                _precip = allocate(state.dimensions, ("x", "y", "timesteps_day"))
                _precip = update(_precip, at[:, :, :], vs.PREC_DIST[i, :][npx.newaxis, npx.newaxis, vs.itt_forc:vs.itt_forc + 6 * 24])
                precip = update(precip, at[:, :, :], npx.where(mask[:, :, npx.newaxis], _precip, precip))
                _ta = allocate(state.dimensions, ("x", "y", "timesteps_day"))
                _ta = update(_ta, at[:, :, :], vs.TA_DIST[i, :][npx.newaxis, npx.newaxis, vs.itt_forc:vs.itt_forc + 6 * 24])
                ta = update(ta, at[:, :, :], npx.where(mask[:, :, npx.newaxis], _ta, ta))
                _ta_min = allocate(state.dimensions, ("x", "y", "timesteps_day"))
                _ta_min = update(_ta_min, at[:, :, :], vs.TA_MIN_DIST[i, :][npx.newaxis, npx.newaxis, vs.itt_forc:vs.itt_forc + 6 * 24])
                ta_min = update(ta_min, at[:, :, :], npx.where(mask[:, :, npx.newaxis], _ta_min, ta_min))
                _ta_max = allocate(state.dimensions, ("x", "y", "timesteps_day"))
                _ta_max = update(_ta_max, at[:, :, :], vs.TA_MAX_DIST[i, :][npx.newaxis, npx.newaxis, vs.itt_forc:vs.itt_forc + 6 * 24])
                ta_max = update(ta_max, at[:, :, :], npx.where(mask[:, :, npx.newaxis], _ta_max, ta_max))
                _pet = allocate(state.dimensions, ("x", "y", "timesteps_day"))
                _pet = update(_pet, at[:, :, :], vs.PET_DIST[i, :][npx.newaxis, npx.newaxis, vs.itt_forc:vs.itt_forc + 6 * 24])
                pet = update(pet, at[:, :, :], npx.where(mask[:, :, npx.newaxis], _pet, pet))

            vs.itt_day = 0
            vs.year = update(
                vs.year, at[1], vs.YEAR[vs.itt_forc]
            )
            vs.month = update(
                vs.month, at[1], vs.MONTH[vs.itt_forc]
            )
            vs.doy = update(
                vs.doy, at[1], vs.DOY[vs.itt_forc]
            )
            vs.prec_day = update(
                vs.prec_day,
                at[2:-2, 2:-2, :],
                precip[2:-2, 2:-2, :]
                * vs.prec_weight[2:-2, 2:-2, npx.newaxis],
            )
            vs.ta_day = update(
                vs.ta_day,
                at[2:-2, 2:-2, :],
                ta[2:-2, 2:-2, :]
                + vs.ta_offset[2:-2, 2:-2, npx.newaxis],
            )
            vs.ta_min = update(
                vs.ta_min,
                at[2:-2, 2:-2, vs.tau],
                npx.min(ta_min[2:-2, 2:-2, :], axis=-1) + vs.ta_offset[2:-2, 2:-2],
            )
            vs.ta_max = update(
                vs.ta_max,
                at[2:-2, 2:-2, vs.tau],
                npx.max(ta_max[2:-2, 2:-2, :], axis=-1) + vs.ta_offset[2:-2, 2:-2],
            )
            vs.pet_day = update(
                vs.pet_day,
                at[2:-2, 2:-2, :],
                pet[2:-2, 2:-2, :]
                * vs.pet_weight[2:-2, 2:-2, npx.newaxis],
            )
            vs.itt_forc = vs.itt_forc + 6 * 24

            if settings.enable_irrigation:
                if vs.itt_forc < (settings.nitt_forc - 5 * 6 * 24):
                    vs.irrig = update(
                            vs.irrig, at[2:-2, 2:-2], 0
                        )
                    # irrigate if sum of rainfall for the next 5 days is less than 1 mm
                    sum_rainfall_next5days = npx.sum(vs.PREC[vs.itt_forc:vs.itt_forc + 5 * 6 * 24])
                    if sum_rainfall_next5days <= 20 and vs.month[1] in [4, 5] and (vs.irr_demand[2:-2, 2:-2] > 0).any():
                        mask_crops = npx.isin(vs.lu_id, npx.array([515, 550]))
                        vs.irrig = update(
                            vs.irrig, at[2:-2, 2:-2], npx.where((vs.irr_demand[2:-2, 2:-2] > 0) & mask_crops[2:-2, 2:-2], 30, vs.irrig[2:-2, 2:-2])
                        )
                    elif sum_rainfall_next5days <= 20 and vs.month[1] in [4, 5, 6] and (vs.irr_demand[2:-2, 2:-2] > 0).any():
                        mask_crops = npx.isin(vs.lu_id, npx.array([541, 542, 543, 544, 546, 556, 557, 558, 559, 560, 579]))
                        vs.irrig = update(
                            vs.irrig, at[2:-2, 2:-2], npx.where((vs.irr_demand[2:-2, 2:-2] > 0) & mask_crops[2:-2, 2:-2], 30, vs.irrig[2:-2, 2:-2])
                        )
                    elif sum_rainfall_next5days <= 20 and vs.month[1] in [4, 5, 6, 7] and (vs.irr_demand[2:-2, 2:-2] > 0).any():
                        mask_crops = npx.isin(vs.lu_id, npx.array([525, 539, 575, 510]))
                        vs.irrig = update(
                            vs.irrig, at[2:-2, 2:-2], npx.where((vs.irr_demand[2:-2, 2:-2] > 0) & mask_crops[2:-2, 2:-2], 30, vs.irrig[2:-2, 2:-2])
                        )
                        mask_crops = npx.isin(vs.lu_id, npx.array([563]))
                        vs.irrig = update(
                            vs.irrig, at[2:-2, 2:-2], npx.where((vs.irr_demand[2:-2, 2:-2] > 0) & mask_crops[2:-2, 2:-2], 40, vs.irrig[2:-2, 2:-2])
                        )
                    elif sum_rainfall_next5days <= 20 and vs.month[1] in [4, 5, 6, 7, 8] and (vs.irr_demand[2:-2, 2:-2] > 0).any():
                        mask_crops = npx.isin(vs.lu_id, npx.array([513]))
                        vs.irrig = update(
                            vs.irrig, at[2:-2, 2:-2], npx.where((vs.irr_demand[2:-2, 2:-2] > 0) & mask_crops[2:-2, 2:-2], 20, vs.irrig[2:-2, 2:-2])
                        )
                        mask_crops = npx.isin(vs.lu_id, npx.array([567]))
                        vs.irrig = update(
                            vs.irrig, at[2:-2, 2:-2], npx.where((vs.irr_demand[2:-2, 2:-2] > 0) & mask_crops[2:-2, 2:-2], 30, vs.irrig[2:-2, 2:-2])
                        )
                    # irrigate for 4 hours from 06:00 to 10:00
                    vs.prec_day = update_add(
                        vs.prec_day, at[2:-2, 2:-2, 6*6:10*6], vs.irrig[2:-2, 2:-2, npx.newaxis] / (6 * 4)
                    )

            if (vs.year[1] != vs.year[0]) & (vs.itt > 1):
                vs.itt_cr = vs.itt_cr + 2
                vs.crop_type = update(vs.crop_type, at[2:-2, 2:-2, 0], vs.crop_type[2:-2, 2:-2, 2])
                vs.crop_type = update(
                    vs.crop_type,
                    at[2:-2, 2:-2, 1],
                    self._read_var_from_csv(f"{vs.year[1]}_summer", self._base_path, "crop_rotations.csv").reshape(settings.nx, settings.ny),
                )
                vs.crop_type = update(
                    vs.crop_type,
                    at[2:-2, 2:-2, 1],
                    self._read_var_from_csv(f"{vs.year[1]}_winter", self._base_path, "crop_rotations.csv").reshape(settings.nx, settings.ny),
                )


    @roger_routine
    def set_diagnostics(self, state):
        diagnostics = state.diagnostics

        diagnostics["rate"].output_variables = self._config["OUTPUT_RATE"]
        # values are aggregated to daily
        diagnostics["rate"].output_frequency = self._config["OUTPUT_FREQUENCY"]
        diagnostics["rate"].sampling_frequency = 1
        diagnostics["rate"].base_output_path = self._output_dir

        diagnostics["collect"].output_variables = self._config["OUTPUT_COLLECT"]
        # values are aggregated to daily
        diagnostics["collect"].output_frequency = self._config["OUTPUT_FREQUENCY"]
        diagnostics["collect"].sampling_frequency = 1
        diagnostics["collect"].base_output_path = self._output_dir

        # maximum bias of deterministic/numerical solution at time step t
        diagnostics["maximum"].output_variables = ["dS_num_error"]
        diagnostics["maximum"].output_frequency = self._config["OUTPUT_FREQUENCY"]
        diagnostics["maximum"].sampling_frequency = 1
        diagnostics["maximum"].base_output_path = self._output_dir

    @roger_routine
    def after_timestep(self, state):
        vs = state.variables

        vs.update(after_timestep_kernel(state))
        vs.update(after_timestep_crops_kernel(state))


@roger_kernel
def set_initial_conditions_crops_kernel(state):
    vs = state.variables

    # calculate time since growing
    t_grow = allocate(state.dimensions, ("x", "y", "crops"))
    t_grow = update(
        t_grow,
        at[2:-2, 2:-2, :], npx.where(vs.z_root_crop[2:-2, 2:-2, vs.taum1, :] > 0, (-1 / vs.root_growth_rate[2:-2, 2:-2, :]) * npx.log(1 / ((vs.z_root_crop[2:-2, 2:-2, vs.taum1, :] / 1000 - vs.z_root_crop_max[2:-2, 2:-2, :] / 1000) * (-1 / (vs.z_root_crop_max[2:-2, 2:-2, :] / 1000 - vs.z_evap[2:-2, 2:-2, npx.newaxis] / 1000)))), 0)
    )

    vs.t_grow_cc = update(
        vs.t_grow_cc,
        at[2:-2, 2:-2, :2, :], t_grow[2:-2, 2:-2, npx.newaxis, :]
    )

    vs.t_grow_root = update(
        vs.t_grow_root,
        at[2:-2, 2:-2, :2, :], t_grow[2:-2, 2:-2, npx.newaxis, :]
    )

    return KernelOutput(
        t_grow_cc=vs.t_grow_cc,
        t_grow_root=vs.t_grow_root,
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
    # set to 0 for numerical errors
    vs.S_fp_rz = update(
        vs.S_fp_rz,
        at[2:-2, 2:-2], npx.where((vs.S_fp_rz[2:-2, 2:-2] > -1e-6) & (vs.S_fp_rz[2:-2, 2:-2] < 0), 0, vs.S_fp_rz[2:-2, 2:-2]),
    )
    vs.S_lp_rz = update(
        vs.S_lp_rz,
        at[2:-2, 2:-2], npx.where((vs.S_lp_rz[2:-2, 2:-2] > -1e-6) & (vs.S_lp_rz[2:-2, 2:-2] < 0), 0, vs.S_lp_rz[2:-2, 2:-2]),
    )
    vs.S_fp_ss = update(
        vs.S_fp_ss,
        at[2:-2, 2:-2], npx.where((vs.S_fp_ss[2:-2, 2:-2] > -1e-6) & (vs.S_fp_ss[2:-2, 2:-2] < 0), 0, vs.S_fp_ss[2:-2, 2:-2]),
    )
    vs.S_lp_ss = update(
        vs.S_lp_ss,
        at[2:-2, 2:-2], npx.where((vs.S_lp_ss[2:-2, 2:-2] > -1e-6) & (vs.S_lp_ss[2:-2, 2:-2] < 0), 0, vs.S_lp_ss[2:-2, 2:-2]),
    )
    vs.prec = update(
        vs.prec,
        at[2:-2, 2:-2, vs.taum1], vs.prec[2:-2, 2:-2, vs.tau],
    )
    vs.event_id = update(
        vs.event_id,
        at[vs.taum1], vs.event_id[vs.tau],
    )
    vs.year = update(
        vs.year,
        at[vs.taum1], vs.year[vs.tau],
    )
    vs.month = update(
        vs.month,
        at[vs.taum1], vs.month[vs.tau],
    )
    vs.doy = update(
        vs.doy,
        at[vs.taum1], vs.doy[vs.tau],
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
        prec=vs.prec,
        event_id=vs.event_id,
        year=vs.year,
        month=vs.month,
        doy=vs.doy,
        S_fp_rz=vs.S_fp_rz,
        S_lp_rz=vs.S_lp_rz,
        S_fp_ss=vs.S_fp_ss,
        S_lp_ss=vs.S_lp_ss,
    )

@roger_kernel
def after_timestep_crops_kernel(state):
    vs = state.variables

    vs.ta_min = update(vs.ta_min, at[2:-2, 2:-2, vs.taum1], vs.ta_min[2:-2, 2:-2, vs.tau])
    vs.ta_max = update(vs.ta_max, at[2:-2, 2:-2, vs.taum1], vs.ta_max[2:-2, 2:-2, vs.tau])
    vs.gdd_sum = update(vs.gdd_sum, at[2:-2, 2:-2, vs.taum1, :], vs.gdd_sum[2:-2, 2:-2, vs.tau, :])
    vs.t_grow_cc = update(vs.t_grow_cc, at[2:-2, 2:-2, vs.taum1, :], vs.t_grow_cc[2:-2, 2:-2, vs.tau, :])
    vs.t_grow_root = update(vs.t_grow_root, at[2:-2, 2:-2, vs.taum1, :], vs.t_grow_root[2:-2, 2:-2, vs.tau, :])
    vs.ccc = update(vs.ccc, at[2:-2, 2:-2, vs.taum1, :], vs.ccc[2:-2, 2:-2, vs.tau, :])
    vs.z_root_crop = update(vs.z_root_crop, at[2:-2, 2:-2, vs.taum1, :], vs.z_root_crop[2:-2, 2:-2, vs.tau, :])
    vs.re_rg_pwp = update(vs.re_rg_pwp, at[2:-2, 2:-2], 0)
    vs.re_rg = update(vs.re_rg, at[2:-2, 2:-2], 0)
    vs.re_rl_pwp = update(vs.re_rl_pwp, at[2:-2, 2:-2], 0)
    vs.re_rl = update(vs.re_rl, at[2:-2, 2:-2], 0)

    return KernelOutput(
        ta_min=vs.ta_min,
        ta_max=vs.ta_max,
        gdd_sum=vs.gdd_sum,
        t_grow_cc=vs.t_grow_cc,
        t_grow_root=vs.t_grow_root,
        ccc=vs.ccc,
        z_root_crop=vs.z_root_crop,
        re_rg_pwp=vs.re_rg_pwp,
        re_rg=vs.re_rg,
        re_rl_pwp=vs.re_rl_pwp,
        re_rl=vs.re_rl,
    )
