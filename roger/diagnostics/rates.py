import os
import copy

from roger.diagnostics.base import RogerDiagnostic
from roger.core.operators import numpy as npx
from roger.variables import TIMESTEPS, AGES, NEVENTS_FF


class Rates(RogerDiagnostic):
    """Time average output diagnostic.

    All registered variables are summed up when :meth:`diagnose` is called,
    and output upon calling :meth:`output`.
    """

    name = "rates"  #:
    output_path = "{identifier}.rates.nc"  #: File to write to. May contain format strings that are replaced with Roger attributes.
    output_variables = None  #: Iterable containing all variables to be averaged. Changes have no effect after ``initialize`` has been called.
    output_frequency = None  #: Frequency (in hours) in which output is written.
    sampling_frequency = None  #: Frequency (in hours) in which variables are accumulated.

    def __init__(self, state):
        self.var_meta = {}
        self.output_variables = []

    def initialize(self, state):
        """Register all variables to be added"""

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

    @staticmethod
    def _has_event_dim(state, var):
        if state.var_meta[var].dims is None:
            return False

        return state.var_meta[var].dims[-1] == NEVENTS_FF[0]

    @staticmethod
    def _has_age_dim(state, var):
        if state.var_meta[var].dims is None:
            return False

        return state.var_meta[var].dims[-1] == AGES[0]

    def diagnose(self, state):
        vs = state.variables
        rate_vs = self.variables

        for key in self.output_variables:
            var_data = getattr(rate_vs, key)
            if self._has_timestep_dim(state, key):
                setattr(rate_vs, key, var_data + getattr(vs, key)[..., vs.tau])
            else:
                setattr(rate_vs, key, var_data + getattr(vs, key))

    def output(self, state):
        """Write rates to netcdf file and zero array"""
        rate_vs = self.variables

        if not os.path.isfile(self.get_output_file_name(state)):
            self.initialize_output(state)

        for key in self.output_variables:
            val = getattr(rate_vs, key)
            setattr(rate_vs, key, val)

        self.write_output(state)

        for key in self.output_variables:
            val = getattr(rate_vs, key)
            setattr(rate_vs, key, 0 * val)
