import pytest

from roger.state import RogerSettings, RogerVariables, RogerState


@pytest.fixture
def dummy_state():
    from roger.variables import VARIABLES, DIM_TO_SHAPE_VAR
    from roger.settings import SETTINGS

    return RogerState(VARIABLES, SETTINGS, DIM_TO_SHAPE_VAR)


@pytest.fixture
def dummy_settings():
    from roger.settings import SETTINGS

    return RogerSettings(SETTINGS)


@pytest.fixture
def dummy_variables():
    from roger.variables import VARIABLES, DIM_TO_SHAPE_VAR
    from roger.settings import SETTINGS

    dummy_state = RogerState(VARIABLES, SETTINGS, DIM_TO_SHAPE_VAR)
    dummy_state.initialize_variables()
    return dummy_state.variables


def test_lock_settings(dummy_settings):
    orig_val = dummy_settings.throughfall_coeff

    with pytest.raises(RuntimeError):
        dummy_settings.throughfall_coeff = 0

    assert dummy_settings.throughfall_coeff == orig_val

    with dummy_settings.unlock():
        dummy_settings.throughfall_coeff = 1

    assert dummy_settings.throughfall_coeff == 1


def test_settings_repr(dummy_settings):
    with dummy_settings.unlock():
        dummy_settings.throughfall_coeff = 1

    assert "throughfall_coeff = 1.0," in repr(dummy_settings)


def test_variables_repr(dummy_variables):
    from roger.core.operators import numpy as npx

    array_type = type(npx.array([]))
    assert f"tau = {array_type} with shape (), dtype int32," in repr(dummy_variables)


def test_to_xarray(dummy_state):
    pytest.importorskip("xarray")

    dummy_state.initialize_variables()
    ds = dummy_state.to_xarray()

    # settings
    assert tuple(ds.attrs.keys()) == tuple(dummy_state.settings.fields())
    assert tuple(ds.attrs.values()) == tuple(dummy_state.settings.values())

    # dimensions
    used_dims = set()
    for var, meta in dummy_state.var_meta.items():
        if var in dummy_state.variables:
            if meta.dims is None:
                continue

            used_dims |= set(meta.dims)

    assert set(ds.coords.keys()) == used_dims

    for dim in used_dims:
        assert int(ds.dims[dim]) == dummy_state.dimensions[dim]

    # variables
    for var in dummy_state.variables.fields():
        assert var in ds


def test_variable_init(dummy_state):
    with pytest.raises(RuntimeError):
        dummy_state.variables

    dummy_state.initialize_variables()

    assert isinstance(dummy_state.variables, RogerVariables)

    with pytest.raises(RuntimeError):
        dummy_state.initialize_variables()


def test_set_dimension(dummy_state):
    with dummy_state.settings.unlock():
        dummy_state.settings.nx = 10

    assert dummy_state.dimensions["x"] == 10

    dummy_state.dimensions["foobar"] = 42
    assert dummy_state.dimensions["foobar"] == 42

    with pytest.raises(RuntimeError):
        dummy_state.dimensions["x"] = 11

    assert dummy_state._dimensions["x"] == "nx"


def test_resize_dimension(dummy_state):
    from roger.state import resize_dimension

    with dummy_state.settings.unlock():
        dummy_state.settings.nx = 10

    dummy_state.initialize_variables()

    assert dummy_state.dimensions["x"] == 10
    assert dummy_state.variables.x.shape == (14,)

    resize_dimension(dummy_state, "x", 100)

    assert dummy_state.dimensions["x"] == 100
    assert dummy_state.variables.x.shape == (104,)


def test_timers(dummy_state):
    from roger.timer import Timer

    timer = dummy_state.timers["foobar"]
    assert isinstance(timer, Timer)
