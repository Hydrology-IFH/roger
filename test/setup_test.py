import pytest


@pytest.fixture(autouse=True)
def set_options():
    from veros import runtime_settings

    object.__setattr__(runtime_settings, "diskless_mode", True)


@pytest.mark.parametrize("float_type", ("float32", "float64"))
def test_setup_dummy(float_type):
    from veros import runtime_settings

    object.__setattr__(runtime_settings, "float_type", float_type)

    from veros.setups.dummy import DUMMYSetup

    sim = DUMMYSetup()
    sim.setup()

    with sim.state.settings.unlock():
        sim.state.settings.runlen = 10

    sim.run()
