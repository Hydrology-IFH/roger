import os
import numpy as np

from roger import roger_routine
from roger.setups.dummy import DUMMYSetup


def _normalize(*arrays):
    if any(a.size == 0 for a in arrays):
        return arrays

    norm = np.abs(arrays[0]).max()
    if norm == 0.0:
        return arrays

    return (a / norm for a in arrays)


class RestartSetup(DUMMYSetup):
    @roger_routine
    def set_diagnostics(self, state):
        for diag in state.diagnostics.values():
            diag.sampling_frequency = 1
            diag.output_frequency = float("inf")


def test_restart(tmpdir):
    os.chdir(tmpdir)

    timesteps_1 = 24 * 5
    timesteps_2 = 24 * 5

    restart_file = "restart.h5"

    dummy_no_restart = RestartSetup(
        override=dict(
            identifier="DUMMY_no_restart",
            restart_input_filename=None,
            restart_output_filename=restart_file,
            runlen=timesteps_1,
        )
    )
    dummy_no_restart.setup()
    dummy_no_restart.run()

    dummy_restart = RestartSetup(
        override=dict(
            identifier="DUMMY_restart",
            restart_input_filename=restart_file,
            restart_output_filename=None,
            runlen=timesteps_2,
        )
    )
    dummy_restart.setup()
    dummy_restart.run()

    with dummy_no_restart.state.settings.unlock():
        dummy_no_restart.state.settings.runlen = timesteps_2

    dummy_no_restart.run()

    state_1, state_2 = dummy_restart.state, dummy_no_restart.state

    for setting in state_1.settings.fields():
        if setting in ("identifier", "restart_input_filename", "restart_output_filename", "runlen"):
            continue

        s1 = state_1.settings.get(setting)
        s2 = state_2.settings.get(setting)
        assert s1 == s2

    def check_var(var):
        v1 = state_1.variables.get(var)
        v2 = state_2.variables.get(var)
        np.testing.assert_allclose(*_normalize(v1, v2), atol=1e-10, rtol=0)

    for var in state_1.variables.fields():
        if var in ("itt",):
            continue

        check_var(var)

    def check_diag_var(diag, var):
        v1 = state_1.diagnostics[diag].variables.get(var)
        v2 = state_2.diagnostics[diag].variables.get(var)
        np.testing.assert_allclose(*_normalize(v1, v2), atol=1e-10, rtol=0)

    for diag in state_1.diagnostics:
        if getattr(state_1.diagnostics[diag], "variables", None) is None:
            continue

        for var in state_1.diagnostics[diag].variables.fields():
            if var in ("itt",):
                continue

            check_diag_var(diag, var)
