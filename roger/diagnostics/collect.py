import os
import copy

from roger.diagnostics.base import RogerDiagnostic
from roger.variables import TIMESTEPS


class Collect(RogerDiagnostic):
    """Collect output diagnostic.

    All registered variables are collected when :meth:`diagnose` is called,
    and collected upon calling :meth:`output`.
    """

    name = "collect"  #:
    output_path = "{identifier}.collect.nc"  #: File to write to. May contain format strings that are replaced with Roger attributes.
    output_variables = None  #: Iterable containing all variables to be volumed. Changes have no effect after ``initialize`` has been called.
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    sampling_frequency = None  #: Frequency (in seconds) in which variables are accumulated.

    def __init__(self, state):
        self.var_meta = {}
        self.output_variables = []

    def initialize(self, state):
        """Register all variables to be collected at certain interval"""

        for var in self.output_variables:
            var_meta = copy.copy(state.var_meta[var])
            var_meta.time_dependent = True
            var_meta.write_to_restart = True

            if self._has_timestep_dim(state, var):
                var_meta.dims = var_meta.dims[:-1]

            if self._has_fourth_dim(state, var):
                var_meta.dims = tuple((var_meta.dims[0], var_meta.dims[1], var_meta.dims[-1]))

            self.var_meta[var] = var_meta

        self.initialize_variables(state)
        self.initialize_output(state)

    @staticmethod
    def _has_timestep_dim(state, var):
        if state.var_meta[var].dims is None:
            return False

        return state.var_meta[var].dims[-1] == TIMESTEPS[0]

    @staticmethod
    def _has_fourth_dim(state, var):
        if state.var_meta[var].dims is None:
            return False

        return state.var_meta[var].dims[-2] == TIMESTEPS[0]

    def reset(self):
        pass

    def diagnose(self, state):
        vs = state.variables
        clc_vs = self.variables

        for key in self.output_variables:
            if self._has_timestep_dim(state, key):
                setattr(clc_vs, key, getattr(vs, key)[..., vs.tau])
            elif self._has_fourth_dim(state, key):
                setattr(clc_vs, key, getattr(vs, key)[:, :, vs.tau, :])
            else:
                setattr(clc_vs, key, getattr(vs, key))

    def output(self, state):
        """Write Collect to netcdf file and zero array"""
        clc_vs = self.variables

        if not os.path.isfile(self.get_output_file_name(state)):
            self.initialize_output(state)

        for key in self.output_variables:
            val = getattr(clc_vs, key)
            setattr(clc_vs, key, val)

        self.write_output(state)