import abc

# do not import roger.core here!
from roger import settings, time, signals, distributed, progress, runtime_settings as rs, logger
from roger.state import get_default_state
from roger.plugins import load_plugin
from roger.routines import roger_routine, is_roger_routine
from roger.timer import timer_context


class RogerSetup(metaclass=abc.ABCMeta):
    """Main class for roger, used for building a model and running it.

    Note:
        This class is meant to be subclassed. Subclasses need to implement the
        methods :meth:`set_parameters`, :meth:`set_topography`, :meth:`set_grid`,
        :meth:`set_coriolis`, :meth:`set_initial_conditions`, :meth:`set_forcing`,
        and :meth:`set_diagnostics`.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from roger import RogerSetup
        >>>
        >>> class MyModel(RogerSetup):
        >>>     ...
        >>>
        >>> model = MyModel()
        >>> model.run()
        >>> plt.imshow(model.state.variables.theta[..., 1])
        >>> plt.show()

    """

    __roger_plugins__ = tuple()

    def __init__(self, override=None):
        self.override_settings = override or {}

        # this should be the first time the core routines are imported
        import roger.core  # noqa: F401

        self._plugin_interfaces = tuple(load_plugin(p) for p in self.__roger_plugins__)
        self._setup_done = False

        self.state = get_default_state(use_plugins=self.__roger_plugins__)

    @abc.abstractmethod
    def set_settings(self, state):
        """To be implemented by subclass.

        First function to be called during setup.
        Use this functions to set the model settings and define physical
        constants.

        Example:
          >>> def set_settings(self, state):
          >>>     settings = state.settings
          >>>     settings.nx, settings.ny, settings.nz = (360, 120, 2)
        """
        pass

    @abc.abstractmethod
    def set_look_up_tables(self, state):
        """To be implemented by subclass.

        Use this function to set the look-up tables.

        Example:
          >>> def set_look_up_tables(self, state):
          >>>     vs = state.variables
          >>>     vs.lut_ilu = update(vs.lut_ilu, at[:, :], lut.ARR_ILU)
        """
        pass

    @abc.abstractmethod
    def set_parameters_setup(self, state):
        """To be implemented by subclass.

        Use this function to set initial or constant model parameters.

        Example:
          >>> def set_parameters_setup(self, state):
          >>>     vs = state.variables
          >>>     vs.dmpv = update(vs.dmpv, at[:, :, :, vs.tau], npx.random.rand(vs.dmpv.shape[:-1]))
        """
        pass

    @abc.abstractmethod
    def set_parameters(self, state):
        """To be implemented by subclass.

        Use this function to modify the model parameters.

        Example:
          >>> def set_parameters(self, state):
          >>>     vs = state.variables
          >>>     vs.dmpv = update(vs.dmpv, at[:, :, :, vs.tau], npx.random.rand(vs.dmpv.shape[:-1]))
        """
        pass

    @abc.abstractmethod
    def set_initial_conditions_setup(self, state):
        """To be implemented by subclass.

        Use this function to set the initial conditions.

        Example:
          >>> @roger_method
          >>> def set_initial_conditions_setup(self, state):
          >>>     vs = state.variables
          >>>     vs.theta_rz = update(vs.theta_rz, at[:, :, :, vs.tau], 0.4)
        """
        pass

    @abc.abstractmethod
    def set_initial_conditions(self, state):
        """To be implemented by subclass.

        May be used to set initial conditions.

        Example:
          >>> @roger_method
          >>> def set_initial_conditions(self, state):
          >>>     vs = state.variables
          >>>     vs.theta_rz = update(vs.theta_rz, at[:, :, :, vs.tau], 0.4)
        """
        pass

    @abc.abstractmethod
    def set_boundary_conditions_setup(self, state):
        """To be implemented by subclass.

        Use this function to set the boundary conditions.

        Example:
          >>> @roger_method
          >>> def set_boundary_conditions_setup(self, state):
          >>>     vs = state.variables
          >>>     vs.z_gw = update(vs.z_gw, at[:, :, :, :vs.taup1], 5)
        """
        pass

    @abc.abstractmethod
    def set_boundary_conditions(self, state):
        """To be implemented by subclass.

        May be used to set boundary conditions.

        Example:
          >>> @roger_method
          >>> def set_boundary_conditions(self, state):
          >>>     vs = state.variables
          >>>     vs.z_gw = update(vs.z_gw, at[:, :, :, :vs.taup1], 5)
        """
        pass

    @abc.abstractmethod
    def set_grid(self, state):
        """To be implemented by subclass.

        Has to set the grid spacings :attr:`x` and :attr:`y`,
        along with the coordinates of the grid origin, :attr:`x_origin` and
        :attr:`y_origin`.

        Example:
          >>> @roger_method
          >>> def set_grid(self, state):
          >>>     vs = state.variables
          >>>     vs.x_origin, vs.y_origin = 0, 0
          >>>     vs.x = 1.
          >>>     vs.y = 1.
        """
        pass

    @abc.abstractmethod
    def set_topography(self, state):
        """To be implemented by subclass.

        Must specify the model topography by setting :attr:`slope`.

        Example:
          >>> @roger_method
          >>> def set_topography(self, state):
          >>>     vs = state.variables
          >>>     vs.slope = update(vs.slope, at[...], 10)
        """
        pass

    @abc.abstractmethod
    def set_forcing_setup(self, state):
        """To be implemented by subclass.

        Called within setup, e.g. through
        :attr:`PREC`, :attr:`TA`, :attr:`PET`.

        Example:
          >>> @roger_method
          >>> def set_forcing_setup(self, state):
          >>>     vs = state.variables
          >>>     vs.TA = vs._ta_data
        """
        pass

    @abc.abstractmethod
    def set_forcing(self, state):
        """To be implemented by subclass.

        Called before every time step to update the external forcing, e.g. through
        :attr:`prec`, :attr:`ta`, :attr:`pet`.

        Example:
          >>> @roger_method
          >>> def set_forcing(self, state):
          >>>     vs = state.variables
          >>>     vs.ta = vs.TA[:, :, vs.itt]
        """
        pass

    @abc.abstractmethod
    def set_diagnostics(self, vs, base_path=None):
        """To be implemented by subclass.

        Called before setting up the :ref:`diagnostics <diagnostics>`. Use this method e.g. to
        mark additional :ref:`variables <variables>` for output.

        Example:
          >>> @roger_method
          >>> def set_diagnostics(self, state):
          >>>     state.diagnostics['snapshot'].output_variables += ['drho', 'dsalt', 'dtemp']
        """
        pass

    @abc.abstractmethod
    def after_timestep(self, state):
        """Called at the end of each time step. Can be used to define custom, setup-specific
        events.
        """
        pass

    def _ensure_setup_done(self):
        if not self._setup_done:
            raise RuntimeError("setup() method has to be called before running the model")

    def setup(self):
        from roger import diagnostics, restart
        from roger.core import surface, soil

        setup_funcs = (
            self.set_parameters_setup,
            self.set_grid,
            self.set_topography,
            self.set_initial_conditions_setup,
            self.set_initial_conditions,
            self.set_boundary_conditions_setup,
            self.set_boundary_conditions,
            self.set_diagnostics,
            self.set_forcing_setup,
            self.after_timestep,
        )

        for f in setup_funcs:
            if not is_roger_routine(f):
                raise RuntimeError(
                    f"{f.__name__} method is not a roger routine. Please make sure to decorate it "
                    "with @roger_routine and try again."
                )

        logger.info("Running model setup")

        with self.state.timers["setup"]:
            with self.state.settings.unlock():
                self.set_settings(self.state)

                for setting, value in self.override_settings.items():
                    setattr(self.state.settings, setting, value)

            settings.check_setting_conflicts(self.state.settings)
            distributed.validate_decomposition(self.state.dimensions)

            self.state.initialize_variables()

            self.state.diagnostics.update(diagnostics.create_default_diagnostics(self.state))

            for plugin in self.state.plugin_interfaces:
                for diagnostic in plugin.diagnostics:
                    self.state.diagnostics[diagnostic.name] = diagnostic()

            self.set_grid(self.state)

            self.set_topography(self.state)

            self.set_look_up_tables(self.state)

            if not self.state.settings.restart_input_filename:
                self.set_parameters_setup(self.state)
                surface.calculate_parameters(self.state)
                soil.calculate_parameters(self.state)

                self.set_initial_conditions_setup(self.state)
                self.set_initial_conditions(self.state)
                surface.calculate_initial_conditions(self.state)
                soil.calculate_initial_conditions(self.state)

            self.set_diagnostics(self.state)
            diagnostics.initialize(self.state)

            self.set_boundary_conditions_setup(self.state)
            self.set_boundary_conditions(self.state)

            if not self.state.settings.restart_input_filename:
                self.set_forcing_setup(self.state)
            restart.read_restart(self.state)

        self._setup_done = True
        if not self.state.settings.enable_offline_transport:
            with self.state.settings.unlock():
                self.state.settings.warmup_done = True

        if self.state.settings.warmup_done:
            # write initial values to output
            diagnostics.diagnose(self.state)
            diagnostics.output(self.state)

        logger.success("Setup done\n")
        if (
            self.state.settings.enable_chloride
            | self.state.settings.enable_bromide
            | self.state.settings.enable_oxygen18
            | self.state.settings.enable_deuterium
            | self.state.settings.enable_nitrate
            | self.state.settings.enable_virtualtracer
        ):
            logger.warning(
                "IMPORTANT: Always check your logger output for warnings\n on diverging solutions. The occurence of warnings may\n require a post-evaluation of the accuracy of\n the numerical solution (e.g. calculate\n standard deviation of dS_num_error or dC_num_error).\n"
            )
        else:
            logger.warning(
                "IMPORTANT: Always check your logger output for warnings\n on diverging solutions. The occurence of warnings may\n require a post-evaluation of the accuracy of\n the numerical solution (e.g. calculate\n standard deviation of dS_num_error).\n"
            )

    @roger_routine
    def step(self, state):
        from roger import diagnostics, restart
        from roger.core import (
            surface,
            soil,
            root_zone,
            subsoil,
            groundwater,
            interception,
            snow,
            evapotranspiration,
            infiltration,
            film_flow,
            surface_runoff,
            subsurface_runoff,
            capillary_rise,
            crop,
            groundwater_flow,
            numerics,
            transport,
            adaptive_time_stepping,
        )

        self._ensure_setup_done()

        vs = state.variables
        settings = state.settings

        with state.timers["diagnostics"]:
            restart.write_restart(state)

        with state.timers["main"]:
            if not settings.enable_offline_transport:
                with state.timers["boundary conditions"]:
                    self.set_boundary_conditions(state)
                with state.timers["forcing"]:
                    self.set_forcing(state)
                if settings.enable_adaptive_time_stepping:
                    with state.timers["adaptive time-stepping"]:
                        adaptive_time_stepping.adaptive_time_stepping(state)
                with state.timers["time-variant parameters"]:
                    self.set_parameters(state)
                if settings.enable_crop_phenology:
                    with state.timers["crops"]:
                        crop.calculate_crop_phenology(state)
                        root_zone.calculate_root_zone(state)
                        subsoil.calculate_subsoil(state)
                        soil.calculate_soil(state)
                with state.timers["interception"]:
                    interception.calculate_interception(state)
                with state.timers["evapotranspiration"]:
                    evapotranspiration.calculate_evapotranspiration(state)
                with state.timers["snow"]:
                    snow.calculate_snow(state)
                with state.timers["infiltration"]:
                    infiltration.calculate_infiltration(state)
                if settings.enable_film_flow:
                    with state.timers["film flow"]:
                        film_flow.calculate_film_flow(state)
                with state.timers["surface runoff"]:
                    surface_runoff.calculate_surface_runoff(state)
                with state.timers["subsurface runoff"]:
                    subsurface_runoff.calculate_subsurface_runoff(state)
                with state.timers["capillary rise"]:
                    capillary_rise.calculate_capillary_rise(state)
                with state.timers["storage"]:
                    surface.calculate_surface(state)
                with state.timers["storage"]:
                    root_zone.calculate_root_zone(state)
                with state.timers["storage"]:
                    subsoil.calculate_subsoil(state)
                with state.timers["storage"]:
                    soil.calculate_soil(state)

                if settings.enable_groundwater_boundary:
                    with state.timers["groundwater recharge"]:
                        groundwater_flow.calculate_groundwater_recharge(state)

                if settings.enable_groundwater:
                    with state.timers["groundwater flow"]:
                        groundwater_flow.calculate_groundwater_flow(state)
                    with state.timers["storage"]:
                        groundwater.calculate_groundwater(state)

                with state.timers["storage"]:
                    numerics.calc_storage(state)

                vs.itt = vs.itt + 1
                vs.time = vs.time + vs.dt_secs
                # write output at end of time step
                with state.timers["diagnostics"]:
                    if not numerics.sanity_check(state):
                        logger.warning(
                            f"Solution diverged at iteration {vs.itt}.\n Please evaluate bias of deterministic/numerical solution.\n The bias is written to the model output."
                        )
                    numerics.calculate_num_error(state)

                    if settings.warmup_done:
                        if vs.time_for_diag >= settings.output_frequency:
                            vs.time_for_diag = 0
                        diagnostics.reset(state)
                        vs.time_for_diag = vs.time_for_diag + vs.dt_secs
                        diagnostics.diagnose(state)
                        diagnostics.output(state)

            elif settings.enable_offline_transport:
                # skip first iteration which contains initial values
                vs.itt = vs.itt + 1
                if settings.sas_solver == "deterministic":
                    vs.time = vs.time + vs.dt_secs

                with state.timers["main transport"]:
                    with state.timers["boundary conditions"]:
                        self.set_boundary_conditions(state)
                    with state.timers["forcing"]:
                        self.set_forcing(state)
                    with state.timers["time-variant parameters"]:
                        self.set_parameters(state)
                    with state.timers["StorAge selection"]:
                        transport.calculate_storage_selection(state)

        self.after_timestep(state)

        # NOTE: benchmarks parse this, do not change / remove
        if rs.profile_mode:
            logger.info(" Time step took {:.2f}s", state.timers["main"].last_time)

    def warmup(self, repeat=1):
        """Warmup routine of the simulation.

        Args
        -------
        repeat : int, optional
            Number of warmup runs. Default is 1.

        Note
        -------
        Make sure to call :meth:`setup` prior to this function.
        """
        from roger import diagnostics
        from roger.core import soil

        if self.state.settings.enable_offline_transport:
            with self.state.timers["warmup"]:
                logger.info("\nStarting warmup")
                for i in range(repeat):
                    self.run()
                    soil.rescale_SA(self.state)
                with self.state.variables.unlock():
                    self.state.variables.itt = 0
                    self.state.variables.time = 0

        with self.state.settings.unlock():
            self.state.settings.warmup_done = True
        if self.state.settings.warmup_done:
            # write initial values to output after warmup
            diagnostics.diagnose(self.state)
            diagnostics.output(self.state)

    def run(self, show_progress_bar=None):
        """Main routine of the simulation.

        Args
        -------
        show_progress_bar (:obj:`bool`, optional): Whether to show fancy progress bar via tqdm.
            By default, only show if stdout is a terminal and roger is running on a single process.

        Note
        -------
        Make sure to call :meth:`setup` (and :meth:`warmup` if offline transport)
        prior to this function.
        """
        from roger import restart

        vs = self.state.variables
        settings = self.state.settings

        if not settings.warmup_done:
            time_length, time_unit = time.format_time(settings.runlen_warmup)
            runlen = settings.runlen_warmup
        else:
            time_length, time_unit = time.format_time(settings.runlen)
            runlen = settings.runlen

        logger.info(f"\nStarting calculation for {time_length:.1f} {time_unit}")

        start_time = vs.time

        # disable timers for first iteration
        timer_context.active = False

        pbar = progress.get_progress_bar(self.state, use_tqdm=show_progress_bar)

        try:
            with signals.signals_to_exception(), pbar:
                while vs.time - start_time < runlen:
                    self.step(self.state)

                    if not timer_context.active:
                        timer_context.active = True

                    pbar.advance_time(vs.dt_secs)

        except:  # noqa: E722
            logger.critical(f"Stopping calculation at iteration {vs.itt}")
            raise

        else:
            if not settings.warmup_done:
                logger.success("Warmup done\n")
            else:
                logger.success("Calculation done\n")

        finally:
            if settings.write_restart:
                restart.write_restart(self.state, force=True)
            self._timing_summary()

    def _timing_summary(self):
        timing_summary = []

        if self.state.timers["main"].total_time > 0:
            timing_summary.extend(
                [
                    "",
                    "Timing summary:",
                    "(excluding first iteration)",
                    "---",
                    " setup time                 = {:.2f}s".format(self.state.timers["setup"].total_time),
                    " main loop time             = {:.2f}s".format(self.state.timers["main"].total_time),
                    "   boundary conditions      = {:.2f}s".format(self.state.timers["boundary conditions"].total_time),
                    "   forcing                  = {:.2f}s".format(self.state.timers["forcing"].total_time),
                    "   time-variant parameters  = {:.2f}s".format(
                        self.state.timers["time-variant parameters"].total_time
                    ),
                    "   interception             = {:.2f}s".format(self.state.timers["interception"].total_time),
                    "   evapotranspiration       = {:.2f}s".format(self.state.timers["evapotranspiration"].total_time),
                    "   snow                     = {:.2f}s".format(self.state.timers["snow"].total_time),
                    "   infiltration             = {:.2f}s".format(self.state.timers["infiltration"].total_time),
                    "   capillary rise           = {:.2f}s".format(self.state.timers["capillary rise"].total_time),
                    "   subsurface flow          = {:.2f}s".format(self.state.timers["subsurface runoff"].total_time),
                    "   groundwater flow         = {:.2f}s".format(self.state.timers["groundwater flow"].total_time),
                    "   crops                    = {:.2f}s".format(self.state.timers["crops"].total_time),
                    "   storage                  = {:.2f}s".format(self.state.timers["storage"].total_time),
                    "   routing                  = {:.2f}s".format(self.state.timers["routing"].total_time),
                ]
            )

        if self.state.timers["main transport"].total_time > 0:
            timing_summary.extend(
                [
                    "",
                    "Timing summary:",
                    "(excluding first iteration)",
                    "---",
                    " setup time                                      = {:.2f}s".format(
                        self.state.timers["setup"].total_time
                    ),
                    " warmup time                                     = {:.2f}s".format(
                        self.state.timers["warmup"].total_time
                    ),
                    " main loop time                                  = {:.2f}s".format(
                        self.state.timers["main transport"].total_time
                    ),
                    "   boundary conditions                           = {:.2f}s".format(
                        self.state.timers["boundary conditions"].total_time
                    ),
                    "   forcing                                       = {:.2f}s".format(
                        self.state.timers["forcing"].total_time
                    ),
                    "   redistribution after root growth/harvesting   = {:.2f}s".format(
                        self.state.timers["redistribution after root growth/harvesting"].total_time
                    ),
                    "   infiltration into root zone                   = {:.2f}s".format(
                        self.state.timers["infiltration into root zone"].total_time
                    ),
                    "   evapotranspiration                            = {:.2f}s".format(
                        self.state.timers["evapotranspiration"].total_time
                    ),
                    "   infiltration into subsoil                     = {:.2f}s".format(
                        self.state.timers["infiltration into subsoil"].total_time
                    ),
                    "   subsurface runoff of root zone                = {:.2f}s".format(
                        self.state.timers["subsurface runoff of root zone"].total_time
                    ),
                    "   subsurface runoff of subsoil                  = {:.2f}s".format(
                        self.state.timers["subsurface runoff of subsoil"].total_time
                    ),
                    "   capillary rise into root zone                 = {:.2f}s".format(
                        self.state.timers["capillary rise into root zone"].total_time
                    ),
                    "   capillary rise into subsoil                   = {:.2f}s".format(
                        self.state.timers["capillary rise into subsoil"].total_time
                    ),
                    "   ageing                                        = {:.2f}s".format(
                        self.state.timers["ageing"].total_time
                    ),
                    "   storage                                       = {:.2f}s".format(
                        self.state.timers["storage"].total_time
                    ),
                    "   nitrogen cycle                                = {:.2f}s".format(
                        self.state.timers["nitrogen cycle"].total_time
                    ),
                    "   routing                                       = {:.2f}s".format(
                        self.state.timers["routing"].total_time
                    ),
                ]
            )

        if rs.profile_mode:
            pass

        timing_summary.extend(
            [
                " diagnostics and I/O      = {:.2f}s".format(self.state.timers["diagnostics"].total_time),
            ]
        )

        timing_summary.extend(
            [
                "   {:<22} = {:.2f}s".format(plugin.name, self.state.timers[plugin.name].total_time)
                for plugin in self.state._plugin_interfaces
            ]
        )

        logger.debug("\n".join(timing_summary))

        if rs.profile_mode:
            print_profile_summary(self.state.profile_timers, self.state.timers["main"].total_time)


def print_profile_summary(profile_timers, main_loop_time):
    profile_timings = ["", "Profile timings:", "[total time spent (% of main loop)]", "---"]
    maxwidth = max(len(k) for k in profile_timers.keys())
    profile_format_string = "{{:<{}}} = {{:.2f}}s ({{:.2f}}%)".format(maxwidth)
    main_loop_time = max(main_loop_time, 1e-8)  # prevent division by 0

    for name, timer in profile_timers.items():
        this_time = timer.total_time
        if this_time == 0:
            continue

        profile_timings.append(profile_format_string.format(name, this_time, 100 * this_time / main_loop_time))

    logger.diagnostic("\n".join(profile_timings))
