import pytest


@pytest.fixture(autouse=True)
def set_options():
    from roger import runtime_settings

    object.__setattr__(runtime_settings, "diskless_mode", True)


@pytest.mark.parametrize("float_type", ("float32", "float64"))
def test_setup_svat_float_types(float_type):
    from roger import runtime_settings

    object.__setattr__(runtime_settings, "float_type", float_type)

    from roger.setups.svat import SVATSetup
    from roger.tools.make_toy_setup import make_setup

    sim = SVATSetup()
    make_setup(sim._base_path, event_type='rain', ndays=10,
               float_type=float_type)
    sim.setup()
    sim.run()


def test_setup_svat_transport():
    pass


def test_setup_svat_crop():
    pass


def test_setup_svat_crop_transport():
    pass


def test_setup_svat_film():
    pass


def test_setup_dist():
    pass


def test_setup_dist_event():
    pass
