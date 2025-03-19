from collections import namedtuple

Setting = namedtuple("setting", ("default", "type", "description"))


def optional(type_):
    def wrapped(arg):
        if arg is None:
            return arg

        return type_(arg)

    return wrapped


PI = 3.14159265358979323846264338327950588

SETTINGS = {
    "identifier": Setting("UNNAMED", str, "Identifier of the current simulation"),
    # spatial and temporal discretization
    "nx": Setting(1, int, "Grid points in zonal (x) direction"),
    "ny": Setting(1, int, "Grid points in meridional (y) direction"),
    "nz": Setting(1, int, "Grid points in vertical (z) direction"),
    "dx": Setting(1, int, "Zonal grid spacing"),
    "dy": Setting(1, int, "Meridional grid spacing"),
    "dz": Setting(1, int, "Vertical grid spacing"),
    "nitt": Setting(1, int, "Number of total iterations"),
    "nitt_forc": Setting(1, int, "Number of total iterations of forcing"),
    "nevent_ff": Setting(1, int, "Number of films"),
    "nittevent_ff": Setting(1, int, "Number of total iterations of a single film flow event"),
    "nittevent_ff_p1": Setting(2, int, "Number of total iterations of a single film flow event for cumulated values"),
    "ages": Setting(1, int, "Number of water ages"),
    "nages": Setting(2, int, "Number of water ages to calculate cumulated distributions"),
    "ncrops": Setting(1, int, "Number of crops per year"),
    "ncrops_to_optimize": Setting(1, int, "Number of crops to_optimize"),
    "ncr": Setting(2, int, "Number of crop rotations"),
    "dt_mom": Setting(1.0, float, "Time step in hours for momentum"),
    "dt_ff": Setting(1 / 6, float, "Time step in hours for film flow"),
    "dt_gw": Setting(24.0, float, "Time step in hours for groundwater"),
    "dt_tracer": Setting(24.0, float, "Time step for tracers, can be larger than dt"),
    "runlen": Setting(0.0, float, "Length of simulation in seconds"),
    "runlen_warmup": Setting(0.0, float, "Length of warmup simulation in seconds"),
    "x_origin": Setting(0, float, "Grid origin in x-direction"),
    "y_origin": Setting(0, float, "Grid origin in y-direction"),
    "time_origin": Setting("1900-01-01 00:00:00", str, "time origin"),
    "AB_eps": Setting(0.1, float, "Deviation from Adam-Bashforth weighting"),
    "nsas": Setting(8, int, "Number of entries per grid cell containing SAS parameters"),
    "nstations": Setting(2, int, "Number of meteorological stations used for forcing"),
    "nflowdirs": Setting(8, int, "Number of flow directions per grid cell"),
    # Physical constants
    "pi": Setting(PI, float, "Pi"),
    "r_mp": Setting(2.5, float, "Macropore radius in mm"),
    "l_sc": Setting(10000, float, "Total length of shrinkage cracks in mm/m^2"),
    "sf": Setting(3, float, "Degree-day factor in -"),
    "ta_fm": Setting(0, float, "freeze-melt threshold in degC"),
    "rmax": Setting(30, float, "Retention capacity of liquid water in snow cover in %"),
    "throughfall_coeff": Setting(0.1, float, "throughfall coeffcient in -"),
    "end_event": Setting(21600, int, "Time after which no rainfall/snow melt occurs in seconds"),
    "hpi": Setting(5, int, "threshold for classification of heavy rainfall event in mm/10min"),
    "a_bc": Setting(2, int, "a parameter for Brooks-Corey"),
    "b_bc": Setting(2, int, "b parameter for Brooks-Corey"),
    "clay_min": Setting(0.01, float, "minimum clay content of soil"),
    "clay_max": Setting(0.71, float, "maximum clay content of soil"),
    "theta_ac_max": Setting(0.71, float, "maximum air capacity to calculate sand content of soil"),
    "theta_rew_min": Setting(0.02, float, "minimum soil water content at permanent wilting point in -"),
    "theta_rew_max": Setting(0.24, float, "maximum soil water content at permanent wilting point in -"),
    "zroot_to_zsoil_max": Setting(0.7, float, "maximum ratio of root zone depth to soil depth in -"),
    "rew_min": Setting(2, float, "minimum readily evaporable water in mm"),
    "rew_max": Setting(12, float, "maximum readily evaporable water in mm"),
    "z_evap_max": Setting(150, float, "maximum soil evaporation depth in mm"),
    "kf_max": Setting(3600, float, "maximum hydraulic conductivity of bedrock in mm/hour"),
    "transp_water_stress": Setting(0.75, float, "fraction of fine pore storage in -"),
    "ccc_decay_rate": Setting(0.005, float, "decay rate of crop canopy cover in -"),
    "basal_crop_coeff_min": Setting(0.15, float, "minimum basal crop coeffcient in -"),
    "ff_tc": Setting(0.15, float, "film flow termination criterium in -"),
    "VSMOW_conc18O": Setting(2005.2e-6, float, "oxygen-18 abundancy ratios according to VSMOW in -"),
    "d18O_min": Setting(-20, float, "potentially lowest oxygen-18 value in per mille"),
    "d18O_max": Setting(0, float, "potentially greatest oxygen-18 value in per mille"),
    "VSMOW_conc2H": Setting(155.76e-6, float, "deuterium abundancy ratios according to VSMOW in -"),
    "d2H_min": Setting(-160, float, "potentially lowest deuterium value in per mille"),
    "d2H_max": Setting(0, float, "potentially greatest deuterium value in per mille"),
    "cum_inf_for_N_input": Setting(20, float, "cumulated infiltration required for nitrogen input in mm"),
    "fraction_ufc_of_irrigation": Setting(0.45, float, "fraction of usable field capacity to define soil water deficit in -"),
    # Logical switches for general model setup
    "coord_degree": Setting(False, bool, "either spherical (True) or cartesian (False) coordinates"),
    "enable_distributed_input": Setting(False, bool, "enable distributed input"),
    "enable_film_flow": Setting(False, bool, "enable film flow process"),
    "enable_lateral_flow": Setting(False, bool, "enable lateral flow"),
    "enable_crop_phenology": Setting(False, bool, "enable crop phenology"),
    "enable_crop_rotation": Setting(False, bool, "enable crop rotation"),
    "enable_crop_specific_irrigation_demand": Setting(False, bool, "enable crop specific irrigation demand"),
    "enable_irrigation": Setting(False, bool, "enable crop irrigation"),
    "enable_net_irrigation": Setting(False, bool, "enable net crop irrigation i.e. interception and evaporation losses are not considered"),
    "enable_crop_partitioning": Setting(False, bool, "enable crop specific solute uptake"),
    "enable_crop_water_stress": Setting(False, bool, "enable crop water stress"),
    "enable_crop_optimization": Setting(False, bool, "enable crop-specific optimization"),
    "enable_offline_transport": Setting(False, bool, "enable offline transport"),
    "enable_groundwater_boundary": Setting(False, bool, "enable groundwater boundary"),
    "enable_groundwater": Setting(False, bool, "enable groundwater"),
    "enable_bromide": Setting(False, bool, "enable bromide"),
    "enable_chloride": Setting(False, bool, "enable enable_chloride"),
    "enable_oxygen18": Setting(False, bool, "enable oxygen-18"),
    "enable_deuterium": Setting(False, bool, "enable deuterium"),
    "enable_nitrate": Setting(False, bool, "enable nitrate"),
    "enable_virtualtracer": Setting(False, bool, "enable virtual tracer"),
    "enable_routing_1D": Setting(False, bool, "enable unidirectional routing"),
    "enable_routing_2D": Setting(False, bool, "enable bidirectional routing"),
    "enable_runon_infiltration": Setting(False, bool, "enable run-on infiltration"),
    "enable_urban": Setting(False, bool, "enable urban"),
    "enable_macropore_lower_boundary_condition": Setting(False, bool, "enable lower boundary condition of macropores"),
    "enable_adaptive_time_stepping": Setting(False, bool, "enable_adaptive_time_stepping"),
    "tm_structure": Setting("UNNAMED", str, "transport model structure"),
    "enable_age_statistics": Setting(False, bool, "enable calculation of age statistics"),
    "warmup_done": Setting(False, bool, "True if after model warmup"),
    "write_restart": Setting(False, bool, "Write HDF5 file for restart"),
    # numerical solver for SAS
    "sas_solver": Setting(None, optional(str), "numerical solver scheme for StorAge selection"),
    "sas_solver_substeps": Setting(1, int, "substeps to solver for StorAge selection numerically"),
    "h": Setting(1, float, "temporal increment of numerical solver (fraction of time step)"),
    "atol": Setting(1e-2, float, "absolute tolerance of solutions"),
    "rtol": Setting(1e-2, float, "relative tolerance of solutions"),
    # Restarts
    "restart_input_filename": Setting(
        None, optional(str), "File name of restart input. If not given, no restart data will be read."
    ),
    "restart_output_filename": Setting(
        "{identifier}_{itt:0>4d}.restart.h5",
        optional(str),
        "File name of restart output. May contain Python format syntax that is substituted with roger attributes.",
    ),
    "restart_frequency": Setting(0, float, "Frequency (in seconds) to write restart data"),
    # Output
    "output_frequency": Setting(86400, float, "Frequency (in seconds) to write output data"),
}


def check_setting_conflicts(settings):
    if settings.enable_groundwater and settings.enable_groundwater_boundary:
        raise RuntimeError(
            "use either the groundwater module or groundwater boundary condition (e.g. fixed groundwater head, time-variant groundwater head)"
        )

    if settings.enable_bromide and (
        settings.enable_chloride | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate
    ):
        raise RuntimeError("use single tracer")

    if settings.enable_chloride and (
        settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate
    ):
        raise RuntimeError("use single tracer")

    if settings.enable_oxygen18 and (
        settings.enable_bromide | settings.enable_chloride | settings.enable_deuterium | settings.enable_nitrate
    ):
        raise RuntimeError("use single tracer")

    if settings.enable_deuterium and (
        settings.enable_bromide | settings.enable_chloride | settings.enable_oxygen18 | settings.enable_nitrate
    ):
        raise RuntimeError("use single tracer")

    if settings.enable_nitrate and (
        settings.enable_bromide | settings.enable_chloride | settings.enable_oxygen18 | settings.enable_deuterium
    ):
        raise RuntimeError("use single tracer")

    if settings.enable_crop_rotation and not settings.enable_crop_phenology:
        raise RuntimeError("use crop rotation in combination with crop phenology")
