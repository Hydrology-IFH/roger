from pathlib import Path
import pytest
import shutil

BASE_PATH = Path(__file__).parent


@pytest.mark.parametrize("float_type", ("float32", "float64"))
def test_setup_svat_float_types(float_type):
    from roger import runtime_settings

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "float_type", float_type)

    from roger.models.svat import SVATSetup
    from roger.tools.make_toy_data import make_toy_forcing

    sim = SVATSetup()
    make_toy_forcing(sim._base_path, event_type="rain", ndays=10,
                     float_type=float_type)
    sim.setup()
    sim.run()
    # delete toy forcing
    files = sim._base_path / "input"
    shutil.rmtree(files, ignore_errors=True)


@pytest.mark.parametrize("float_type", ("float32", "float64"))
def test_setup_svat_crop_float_types(float_type):
    from roger import runtime_settings

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "float_type", float_type)

    from roger.models.svat_crop import SVATCROPSetup
    from roger.tools.make_toy_data import make_toy_forcing

    sim = SVATCROPSetup()
    make_toy_forcing(sim._base_path, event_type="rain", ndays=10,
                     enable_crop_phenology=True,
                     float_type=float_type)
    sim.setup()
    sim.run()
    # delete toy forcing
    files = sim._base_path / "input"
    shutil.rmtree(files, ignore_errors=True)


@pytest.mark.parametrize("float_type", ("float32", "float64"))
def test_setup_oneD_float_types(float_type):
    from roger import runtime_settings

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "float_type", float_type)

    from roger.models.oneD import ONEDSetup
    from roger.tools.make_toy_data import make_toy_forcing

    sim = ONEDSetup()
    make_toy_forcing(sim._base_path, event_type="rain", ndays=10,
                     float_type=float_type)
    sim.setup()
    sim.run()
    # delete toy forcing
    files = sim._base_path / "input"
    shutil.rmtree(files, ignore_errors=True)


@pytest.mark.parametrize("float_type", ("float32", "float64"))
def test_setup_oneD_event_float_types(float_type):
    from roger import runtime_settings

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "float_type", float_type)

    from roger.models.oneD_event import ONEDEVENTSetup
    from roger.tools.make_toy_data import make_toy_forcing_event

    sim = ONEDEVENTSetup()
    make_toy_forcing_event(sim._base_path, event_type="rain",
                           float_type=float_type)
    sim.setup()
    sim.run()
    # delete toy forcing
    files = sim._base_path / "input"
    shutil.rmtree(files, ignore_errors=True)


@pytest.mark.parametrize("float_type", ("float32", "float64"))
def test_setup_svat_transport_bromide(float_type):
    from roger import runtime_settings

    object.__setattr__(runtime_settings, "diskless_mode", False)
    object.__setattr__(runtime_settings, "float_type", float_type)

    from roger.models.svat_transport_bromide import SVATTRANSPORTSetup
    from roger.tools.make_toy_data import make_toy_forcing_tracer
    from make_data_for_svat_transport import make_data

    make_data(float_type)
    sim = SVATTRANSPORTSetup()
    make_toy_forcing_tracer(sim._base_path, tracer="Br", ndays=10,
                            float_type=float_type)
    shutil.move(BASE_PATH / "states_hm.nc", sim._base_path / "input" / "states_hm.nc")
    sim.setup()
    sim.warmup()
    sim.run()
    # delete toy forcing
    files = sim._base_path / "input"
    shutil.rmtree(files, ignore_errors=True)


@pytest.mark.parametrize("float_type", ("float32", "float64"))
def test_setup_svat_transport_oxygen18(float_type):
    from roger import runtime_settings

    object.__setattr__(runtime_settings, "diskless_mode", False)
    object.__setattr__(runtime_settings, "float_type", float_type)

    from roger.models.svat_transport_oxygen18 import SVATTRANSPORTSetup
    from roger.tools.make_toy_data import make_toy_forcing_tracer
    from make_data_for_svat_transport import make_data

    make_data(float_type)
    sim = SVATTRANSPORTSetup()
    make_toy_forcing_tracer(sim._base_path, tracer="d18O", ndays=10,
                            float_type=float_type)
    shutil.move(BASE_PATH / "states_hm.nc", sim._base_path / "states_hm.nc")
    sim.setup()
    sim.warmup()
    sim.run()
    # delete toy forcing
    files = sim._base_path / "input"
    shutil.rmtree(files, ignore_errors=True)
