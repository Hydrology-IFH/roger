"""The Basic Model Interface (BMI) Python specification.

This language specification is derived from the Scientific Interface
Definition Language (SIDL) file bmi.sidl located at
https://github.com/csdms/bmi.
"""

from pathlib import Path
from typing import Tuple
from bmipy import Bmi
from roger import signals, logger, progress, time

import numpy as np
import yaml


class BmiRoger(Bmi):
    def __init__(self, model=None):
        """Create a BmiRoger model that is ready for initialization."""
        self._model = model
        self._pbar = None

        self._input_var_names = ()
        self._output_var_names = ()

    def initialize(self, path):
        """Perform startup tasks for the model.

        Perform all tasks that take place before entering the model's time
        loop, including opening files and initializing the model state. Model
        inputs are read from a text-based configuration file, specified by
        `self._model._config_file`.

        Parameters
        ----------
        self._model._config_file : str, optional
            The path to the model configuration file.

        Notes
        -----
        Models should be refactored, if necessary, to use a
        configuration file. CSDMS does not impose any constraint on
        how configuration files are formatted, although YAML is
        recommended. A template of a model's configuration file
        with placeholder values is used by the BMI.
        """
        if not self._model:
            _file_config = path / "config.yaml"
            with open(_file_config, "r") as file:
                _config = yaml.safe_load(file)
            try:
                exec("from roger.bmimodels.{_config['model'].lower()} import {_config['model'].upper()}Setup")
                self._model = eval(f"{_config['model']}Setup()")
                self._model._config = _config
            except ValueError:
                logger.error(f"Model {_config['model']} not found")
                raise
        else:
            with open(self._model._file_config, "r") as file:
                self._model._config = yaml.safe_load(file)
                self._model._config["model"] = self._model.__class__.__name__.split("Setup")[0]
        if "event" in self._model._config["model"].split("_"):
            self._input_var_names = ("ta", "prec")
            from roger.tools.setup import write_forcing_event

            write_forcing_event(self._model.input_dir)
            self._model.setup()
            self._pbar = progress.get_progress_bar(self._model.state, use_tqdm=False)
        elif "transport" in self._model._config["model"].split("_"):
            self._input_var_names = ("ta", "prec", "pet")
            from roger.tools.setup import write_forcing_tracer

            self._model.setup()
            self._pbar = progress.get_progress_bar(self._model.state, use_tqdm=False)
            self._model.warmup()
        elif "bromide" in self._model._config["model"].split("_"):
            self._input_var_names = ("ta", "prec", "pet", "C_in")
            from roger.tools.setup import write_forcing_tracer

            write_forcing_tracer(self._model._input_dir, "Br")
            self._model.setup()
            self._pbar = progress.get_progress_bar(self._model.state, use_tqdm=False)
            self._model.warmup()
        elif "chloride" in self._model._config["model"].split("_"):
            self._input_var_names = ("ta", "prec", "pet", "C_in")
            from roger.tools.setup import write_forcing_tracer

            write_forcing_tracer(self._model._input_dir, "Cl")
            self._model.setup()
            self._pbar = progress.get_progress_bar(self._model.state, use_tqdm=False)
            self._model.warmup()
        elif "oxygen18" in self._model._config["model"].split("_"):
            self._input_var_names = ("ta", "prec", "pet", "C_iso_in")
            from roger.tools.setup import write_forcing_tracer

            write_forcing_tracer(self._model._input_dir, "d18O")
            self._model.setup()
            self._pbar = progress.get_progress_bar(self._model.state, use_tqdm=False)
            self._model.warmup()
        elif "nitrate" in self._model._config["model"].split("_"):
            self._input_var_names = ("ta", "prec", "pet", "C_in")
            from roger.tools.setup import write_forcing_tracer

            write_forcing_tracer(self._model._input_dir, "NO3")
            self._model.setup()
            self._pbar = progress.get_progress_bar(self._model.state, use_tqdm=False)
            self._model.warmup()
        else:
            self._input_var_names = ("ta", "prec", "pet")
            from roger.tools.setup import write_forcing

            write_forcing(Path(path) / "input")
            self._model.setup()
            self._pbar = progress.get_progress_bar(self._model.state, use_tqdm=False)
            self._pbar

        output_var_names = []
        for key in self._model._config.keys():
            if key in ["OUTPUT_RATE", "OUTPUT_COLLECT"]:
                output_var_names = output_var_names + self._model._config[key]
        self._output_var_names = tuple(output_var_names)

    def update(self) -> None:
        """Advance model state by one time step.

        Perform all tasks that take place within one pass through the model's
        time loop. This typically includes incrementing all of the model's
        state variables. If the model's state variables don't change in time,
        then they can be computed by the :func:`initialize` method and this
        method can return with no action.
        """
        from roger import restart

        if self._model.state.variables.time <= 0:
            self._model._ensure_warmup_done()

            if not self._model.state.settings.warmup_done:
                time_length, time_unit = time.format_time(self._model.state.settings.runlen_warmup)
                runlen = self._model.state.settings.runlen_warmup
            else:
                time_length, time_unit = time.format_time(self._model.state.settings.runlen)
                runlen = self._model.state.settings.runlen

            logger.info(f"\nStarting calculation for {time_length:.1f} {time_unit}")

        else:
            if not self._model.state.settings.warmup_done:
                runlen = self._model.state.settings.runlen_warmup
            else:
                runlen = self._model.state.settings.runlen

        try:
            with signals.signals_to_exception():
                self._pbar.start_time()
                self._model.step(self._model.state)
                self._pbar.advance_time(self._model.state.variables.dt_secs)
        except:  # noqa: E722
            logger.critical(f"Stopping calculation at iteration {self._model.state.variables.itt}")
            raise

        else:
            if self._model.state.variables.time >= runlen:
                if not self._model.state.settings.warmup_done:
                    logger.success("Warmup done\n")
                else:
                    logger.success("Calculation done\n")

        finally:
            if self._model.state.settings.write_restart:
                restart.write_restart(self._model.state, force=True)

    def update_until(self, timespan: int) -> None:
        """Advance the simulation until the given time.

        Parameters
        ----------
        timespan : int
            A time span in seconds to advance the simulation.
        """
        from roger import restart

        if self._model.state.variables.time <= 0:
            self._model._ensure_warmup_done()

            if not self._model.state.settings.warmup_done:
                time_length, time_unit = time.format_time(self._model.state.settings.runlen_warmup)
                runlen = self._model.state.settings.runlen_warmup
            else:
                time_length, time_unit = time.format_time(self._model.state.settings.runlen)
                runlen = self._model.state.settings.runlen

            logger.info(f"\nStarting calculation for {time_length:.1f} {time_unit}")

        else:
            if not self._model.state.settings.warmup_done:
                runlen = self._model.state.settings.runlen_warmup
            else:
                runlen = self._model.state.settings.runlen

        start_time = self._model.state.variables.time

        try:
            with signals.signals_to_exception():
                while self._model.state.variables.time - start_time < timespan:
                    self._pbar.start_time()
                    self._model.step(self._model.state)
                    self._pbar.advance_time(self._model.state.variables.dt_secs)

        except:  # noqa: E722
            logger.critical(f"Stopping calculation at iteration {self._model.state.variables.itt}")
            raise

        else:
            if self._model.state.variables.time >= runlen:
                if not self._model.state.settings.warmup_done:
                    logger.success("Warmup done\n")
                else:
                    logger.success("Calculation done\n")

        finally:
            if self._model.state.settings.write_restart:
                restart.write_restart(self._model.state, force=True)

    def finalize(self) -> None:
        """Perform tear-down tasks for the model.

        Perform all tasks that take place after exiting the model's time
        loop. This typically includes deallocating memory, closing files and
        printing reports.
        """
        self._model = None

    def get_component_name(self) -> str:
        """Name of the component.

        Returns
        -------
        str
            The name of the component.
        """
        return self._model._config["model"]

    def get_input_item_count(self) -> int:
        """Count of a model's input variables.

        Returns
        -------
        int
          The number of input variables.
        """
        return len(self._input_var_names)

    def get_output_item_count(self) -> int:
        """Count of a model's output variables.

        Returns
        -------
        int
          The number of output variables.
        """
        return len(self._output_var_names)

    def get_input_var_names(self) -> Tuple[str]:
        """List of a model's input variables.

        Input variable names must be CSDMS Standard Names, also known
        as *long variable names*.

        Returns
        -------
        list of str
            The input variables for the model.

        Notes
        -----
        Standard Names enable the CSDMS framework to determine whether
        an input variable in one model is equivalent to, or compatible
        with, an output variable in another model. This allows the
        framework to automatically connect components.

        Standard Names do not have to be used within the model.
        """
        return self._input_var_names

    def get_output_var_names(self) -> Tuple[str]:
        """List of a model's output variables.

        Output variable names must be CSDMS Standard Names, also known
        as *long variable names*.

        Returns
        -------
        list of str
            The output variables for the model.
        """
        return self._output_var_names

    def get_var_grid(self, name: str) -> int:
        """Get grid identifier for the given variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        int
          The grid identifier.
        """
        vs = self._model.state.variables.__active_vars__
        for grid_id, var_name_list in vs.items():
            if name in var_name_list:
                return grid_id

    def get_var_type(self, name: str) -> str:
        """Get data type of the given variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        str
            The Python variable type; e.g., ``str``, ``int``, ``float``.
        """
        return str(self.get_value_ptr(name).dtype)

    def get_var_units(self, name: str) -> str:
        """Get units of the given variable.

        Standard unit names, in lower case, should be used, such as
        ``meters`` or ``seconds``. Standard abbreviations, like ``m`` for
        meters, are also supported. For variables with compound units,
        each unit name is separated by a single space, with exponents
        other than 1 placed immediately after the name, as in ``m s-1``
        for velocity, ``W m-2`` for an energy flux, or ``km2`` for an
        area.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        str
            The variable units.

        Notes
        -----
        CSDMS uses the `UDUNITS`_ standard from Unidata.

        .. _UDUNITS: http://www.unidata.ucar.edu/software/udunits
        """
        return self._model.state.variables.__active_vars__[f"{name}"].units

    def get_var_itemsize(self, name: str) -> int:
        """Get memory use for each array element in bytes.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        int
            Item size in bytes.
        """
        return np.dtype(self.get_var_type(name)).itemsize

    def get_var_nbytes(self, name: str) -> int:
        """Get size, in bytes, of the given variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        int
            The size of the variable, counted in bytes.
        """
        return self.get_value_ptr(name).nbytes

    def get_var_location(self) -> str:
        """Get the grid element type that the a given variable is defined on.

        The grid topology can be composed of *nodes*, *edges*, and *faces*.

        *node*
            A point that has a coordinate pair or triplet: the most
            basic element of the topology.

        *edge*
            A line or curve bounded by two *nodes*.

        *face*
            A plane or surface enclosed by a set of edges. In a 2D
            horizontal application one may consider the word “polygon”,
            but in the hierarchy of elements the word “face” is most common.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        str
            The grid location on which the variable is defined. Must be one of
            `"node"`, `"edge"`, or `"face"`.

        Notes
        -----
        CSDMS uses the `ugrid conventions`_ to define unstructured grids.

        .. _ugrid conventions: http://ugrid-conventions.github.io/ugrid-conventions
        """
        return "node"

    def get_current_time(self) -> int:
        """Current time of the model.

        Returns
        -------
        float
            The current model time.
        """
        return self._model.state.variables.time

    def get_start_time(self) -> int:
        """Start time of the model.

        Model times should be of type float.

        Returns
        -------
        float
            The model start time.
        """
        return 0

    def get_end_time(self) -> int:
        """End time of the model.

        Returns
        -------
        float
            The maximum model time.
        """
        return self._model.state.settings.runlen

    def get_time_units(self) -> str:
        """Time units of the model.

        Returns
        -------
        str
            The model time unit; e.g., `days` or `s`.

        Notes
        -----
        CSDMS uses the UDUNITS standard from Unidata.
        """
        return "hours"

    def get_time_step(self) -> float:
        """Current time step of the model.

        The model time step should be of type float.

        Returns
        -------
        float
            The time step used in model.
        """
        return self._model.dt

    def get_value(self, name: str, dest: np.ndarray) -> np.ndarray:
        """Get a copy of values of the given variable.

        This is a getter for the model, used to access the model's
        current state. It returns a *copy* of a model variable, with
        the return type, size and rank dependent on the variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.

        Returns
        -------
        ndarray
            The same numpy array that was passed as an input buffer.
        """
        dest[:] = self.get_value_ptr(name)[2:-2, 2:-2, ...].flatten()
        return dest

    def get_value_ptr(self, name: str) -> np.ndarray:
        """Get a reference to values of the given variable.

        This is a getter for the model, used to access the model's
        current state. It returns a reference to a model variable,
        with the return type, size and rank dependent on the variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        array_like
            A reference to a model variable.
        """
        if name in list(self._model.state.diagnostics["rate"].var_meta.keys()):
            arr = self._model.state.diagnostics["rate"].variables.get(f"{name}")
        elif name in list(self._model.state.diagnostics["average"].var_meta.keys()):
            arr = self._model.state.diagnostics["average"].variables.get(f"{name}")
        elif name in list(self._model.state.diagnostics["maximum"].var_meta.keys()):
            arr = self._model.state.diagnostics["maximum"].variables.get(f"{name}")
        else:
            arr = self._model.state.variables.get(f"{name}")
        arr.setflags(write=1)
        return arr

    def get_value_at_indices(self, name: str, dest: np.ndarray, inds: np.ndarray) -> np.ndarray:
        """Get values at particular indices.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        inds : array_like
            The indices into the variable array.

        Returns
        -------
        array_like
            Value of the model variable at the given location.
        """
        dest[:] = self.get_value_ptr(name)[2:-2, 2:-2, ...].take(inds)
        return dest

    def set_value(self, name: str, src: np.ndarray) -> None:
        """Specify a new value for a model variable.

        This is the setter for the model, used to change the model's
        current state. It accepts, through *src*, a new value for a
        model variable, with the type, size and rank of *src*
        dependent on the variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.
        src : array_like
            The new value for the specified variable.
        """
        val = self.get_value_ptr(name)[2:-2, 2:-2, ...]
        if val.shape[-1] == 2:
            val[..., 1] = src.reshape((val.shape[0], val.shape[1]))
        else:
            val[:] = src.reshape(val.shape)

    def set_value_at_indices(self, name: str, inds: np.ndarray, src: np.ndarray) -> None:
        """Specify a new value for a model variable at particular indices.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.
        inds : array_like
            The indices into the variable array.
        src : array_like
            The new value for the specified variable.
        """
        val = self.get_value_ptr(name)
        val.flat[inds] = src

    # Grid information
    def get_grid_rank(self, name: str) -> int:
        """Get number of dimensions of the computational grid.

        Returns
        -------
        int
            Rank of the grid.
        """
        return len(self._model.state.variables.__active_vars__[f"{name}"].dims)

    def get_grid_size(self) -> int:
        """Get the total number of elements in the computational grid.

        Returns
        -------
        int
            Size of the grid.
        """
        return self._model.nx * self._model.ny

    def get_grid_type(self) -> str:
        """Get the grid type as a string.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        str
            Type of grid as a string.
        """
        return "uniform_rectilinear"

    # Uniform rectilinear
    def get_grid_shape(self, name: str):
        """Get dimensions of the computational grid.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.
        """
        return self.get_value_ptr(name)[2:-2, 2:-2, ...].shape

    def get_grid_spacing(self):
        """Spacing of rows and columns of uniform rectilinear grid."""
        return (self._model.state.settings.dx, self._model.state.settings.dy)

    def get_grid_origin(self):
        """Get coordinates for the lower-left corner of the computational grid."""
        return (self._model.state.settings.x_origin, self._model.state.settings.y_origin)

    # Non-uniform rectilinear, curvilinear
    def get_grid_x(self, grid: int, x: np.ndarray) -> np.ndarray:
        """Get coordinates of grid nodes in the x direction.

        Parameters
        ----------
        grid : int
            A grid identifier.
        x : ndarray of float, shape *(nrows,)*
            A numpy array to hold the x-coordinates of the grid node columns.

        Returns
        -------
        ndarray of float
            The input numpy array that holds the grid's column x-coordinates.
        """
        return self._model.state.settings.x_origin + self._model.state.settings.dx * np.arange(
            self._model.state.settings.nx
        )

    def get_grid_y(self, grid: int, y: np.ndarray) -> np.ndarray:
        """Get coordinates of grid nodes in the y direction.

        Parameters
        ----------
        grid : int
            A grid identifier.
        y : ndarray of float, shape *(ncols,)*
            A numpy array to hold the y-coordinates of the grid node rows.

        Returns
        -------
        ndarray of float
            The input numpy array that holds the grid's row y-coordinates.
        """
        return self._model.state.settings.y_origin + self._model.state.settings.dy * np.arange(
            self._model.state.settings.ny
        )

    def get_grid_z(self, grid: int, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError("get_grid_z")

    def get_grid_node_count(self):
        return self._model.state.settings.nx * self._model.state.settings.ny

    def get_grid_edge_count(self):
        raise NotImplementedError("get_grid_edge_count")

    def get_grid_face_count(self):
        raise NotImplementedError("get_grid_face_count")

    def get_grid_edge_nodes(self):
        raise NotImplementedError("get_grid_edge_nodes")

    def get_grid_face_edges(self):
        raise NotImplementedError("get_grid_face_edges")

    def get_grid_face_nodes(self):
        raise NotImplementedError("get_grid_face_nodes")

    def get_grid_nodes_per_face(self):
        raise NotImplementedError("get_grid_nodes_per_face")
