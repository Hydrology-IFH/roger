from roger import logger

from roger.variables import Variable
from roger.core.operators import numpy as npx
from roger.diagnostics.base import RogerDiagnostic
from roger.distributed import global_sum


class TracerMonitor(RogerDiagnostic):
    """Diagnostic monitoring tracer contents / fluxes.

    Writes output to stdout (no binary output).
    """

    name = "tracer_monitor"
    output_frequency = None

    def __init__(self, state):
        self.var_meta = {
            "C_s": Variable("C_s", None, write_to_restart=True),
            "M_S": Variable("M_s", None, write_to_restart=True),
        }

    def initialize(self, state):
        self.initialize_variables(state)

    def diagnose(self, state):
        pass

    def output(self, state):
        """
        Diagnose tracer content
        """
        vs = state.variables
        tracer_vs = self.variables

        conc = global_sum(npx.sum(vs.C_s[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2, npx.newaxis]))
        mass = global_sum(npx.sum(vs.M_s[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2, npx.newaxis]))

        logger.diagnostic(
            f" Concentration {conc} change to last {(conc - tracer_vs.C_s)}"
        )
        logger.diagnostic(
            f" Mass {mass} change to last {(mass - tracer_vs.M_s)}"
        )

        tracer_vs.M = mass
        tracer_vs.C = conc
