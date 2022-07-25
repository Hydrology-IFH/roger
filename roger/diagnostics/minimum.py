import os
import copy

from roger.diagnostics.base import RogerDiagnostic
from roger.variables import TIMESTEPS, allocate
from roger.core.operators import numpy as npx, update, at


class Minimum(RogerDiagnostic):
    """Minimum output diagnostic.

    All registered variables are stacked when :meth:`diagnose` is called,
    and returns minimum and output upon calling :meth:`output`.
    """

    name = "minimum"  #:
    output_path = "{identifier}.minimum.nc"  #: File to write to. May contain format strings that are replaced with Roger attributes.
    output_variables = None  #: Iterable containing all variables to be minimumd. Changes have no effect after ``initialize`` has been called.
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    sampling_frequency = None  #: Frequency (in seconds) in which variables are accumulated.

    def __init__(self, state):
        self.var_meta = {}
        self.output_variables = []

    def initialize(self, state):
        """Register all variables to be minimumd"""

        for var in self.output_variables:
            var_meta = copy.copy(state.var_meta[var])
            var_meta.time_dependent = True
            var_meta.write_to_restart = True

            if self._has_timestep_dim(state, var):
                var_meta.dims = var_meta.dims[:-1]

            self.var_meta[var] = var_meta

        self.initialize_variables(state)
        self.initialize_output(state)

    @staticmethod
    def _has_timestep_dim(state, var):
        if state.var_meta[var].dims is None:
            return False

        return state.var_meta[var].dims[-1] == TIMESTEPS[0]

    def diagnose(self, state):
        vs = state.variables
        min_vs = self.variables

        var_data = allocate(state.dimensions, ("x", "y", 2))

        for key in self.output_variables:
            if self._has_timestep_dim(state, key):
                var_data = update(
                    var_data,
                    at[:, :, 0], getattr(vs, key)[..., vs.taum1],
                )
                var_data = update(
                    var_data,
                    at[:, :, 1], getattr(vs, key)[..., vs.tau],
                )
                var_data = update(
                    var_data,
                    at[:, :, 0], npx.where((var_data[:, :, 0] == 0) & (var_data[:, :, 1] > 0), npx.nan, var_data[:, :, 0]),
                )
                setattr(min_vs, key, npx.nanmin(var_data, axis=-1))
            else:
                var_data = update(
                    var_data,
                    at[:, :, 0], getattr(min_vs, key),
                )
                var_data = update(
                    var_data,
                    at[:, :, 1], getattr(vs, key),
                )
                var_data = update(
                    var_data,
                    at[:, :, 0], npx.where((var_data[:, :, 0] == 0) & (var_data[:, :, 1] > 0), npx.nan, var_data[:, :, 0]),
                )
                setattr(min_vs, key, npx.nanmin(var_data, axis=-1))

    def output(self, state):
        """Write minimum to netcdf file and zero array"""
        min_vs = self.variables

        if not os.path.isfile(self.get_output_file_name(state)):
            self.initialize_output(state)

        for key in self.output_variables:
            val = getattr(min_vs, key)
            setattr(min_vs, key, val)

        self.write_output(state)
