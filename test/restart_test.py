import os
import numpy as np

from roger.setups.svat import SVATSetup


def _normalize(*arrays):
    if any(a.size == 0 for a in arrays):
        return arrays

    norm = np.abs(arrays[0]).max()
    if norm == 0.0:
        return arrays

    return (a / norm for a in arrays)


class RestartSetup(SVATSetup):
    pass


def test_restart(tmpdir):
    from roger.tools.make_toy_data import make_toy_forcing  # noqa: E402
    os.chdir(tmpdir)

    timesteps_1 = 5 * 24 * 60 * 60
    timesteps_2 = 10 * 24 * 60 * 60

    restart_file = "restart.h5"

    svat_no_restart = RestartSetup(
        override=dict(
            identifier="svat_no_restart",
            restart_input_filename=None,
            restart_output_filename=restart_file,
            write_restart=True,
            runlen=timesteps_1,
        )
    )
    make_toy_forcing(svat_no_restart._base_path, event_type='rain', ndays=10)
    svat_no_restart.setup()
    svat_no_restart.run()

    svat_restart = RestartSetup(
        override=dict(
            identifier="svat_restart",
            restart_input_filename=restart_file,
            restart_output_filename=None,
            write_restart=False,
            runlen=timesteps_2,
        )
    )
    svat_restart.setup()
    svat_restart.run()

    with svat_no_restart.state.settings.unlock():
        svat_no_restart.state.settings.runlen = timesteps_2

    svat_no_restart.run()

    state_1, state_2 = svat_restart.state, svat_no_restart.state

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
