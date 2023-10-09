from roger import logger

from roger.variables import Variable
from roger.core.operators import numpy as npx
from roger.diagnostics.base import RogerDiagnostic
from roger.distributed import global_sum


class WaterMonitor(RogerDiagnostic):
    """Diagnostic monitoring of water storage.

    Writes output to stdout (no binary output).
    """

    name = "water_monitor"
    output_frequency = None

    def __init__(self, state):
        self.var_meta = {
            "S_sur": Variable("S_sur", None, write_to_restart=True),
            "S_s": Variable("S_s", None, write_to_restart=True),
        }

    def initialize(self, state):
        self.initialize_variables(state)

    def reset(self):
        pass

    def diagnose(self, state):
        pass

    def output(self, state):
        """
        Diagnose water storage
        """
        vs = state.variables
        storage_vs = self.variables

        S_sur = global_sum(npx.sum(vs.S_sur[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2, npx.newaxis]))
        S_s = global_sum(npx.sum(vs.S_s[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2, npx.newaxis]))

        logger.diagnostic(f" Surface storage {S_sur} change to last {(S_sur - storage_vs.S_sur)}")
        logger.diagnostic(f" Soil storage {S_s} change to last {(S_s - storage_vs.S_s)}")

        storage_vs.S_sur = S_sur
        storage_vs.S_s = S_s
