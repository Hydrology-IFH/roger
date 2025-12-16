import os
import copy

from roger import time, logger
from roger.diagnostics.base import RogerDiagnostic


DEFAULT_OUTPUT_VARS = [
    "ta",
]


class Snapshot(RogerDiagnostic):
    """Writes snapshots of the current solution. Also reads and writes the main
    restart data required for restarting a Roger simulation.
    """

    output_path = "{identifier}.snapshot.nc"
    """File to write to. May contain format strings that are replaced with Roger attributes."""
    name = "snapshot"  #:
    output_frequency = None  #: Frequency (in seconds) in which output is written.

    def __init__(self, state):
        self.output_variables = []

        for var in DEFAULT_OUTPUT_VARS:
            active = state.var_meta[var].active
            if callable(active):
                active = active(state.settings)

            if active:
                self.output_variables.append(var)

    def initialize(self, state):
        vs = state.variables

        self.var_meta = {var: copy.copy(state.var_meta[var]) for var in self.output_variables}
        for var in self.var_meta.values():
            var.write_to_restart = False

        self.variables = vs
        self.initialize_output(state)

    def reset(self):
        pass

    def diagnose(self, state):
        pass

    def output(self, state):
        vs = state.variables

        time_length, time_unit = time.format_time(vs.time)
        logger.info(f" Writing snapshot at {time_length:.2f} {time_unit}")

        if not os.path.isfile(self.get_output_file_name(state)):
            self.initialize_output(state)

        self.write_output(state)
