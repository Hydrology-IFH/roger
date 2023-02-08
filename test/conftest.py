import os
import pytest


def pytest_addoption(parser):
    parser.addoption("--backend", choices=["numpy", "jax"], default="numpy", help="Numerical backend to test")


def pytest_configure(config):
    backend = config.getoption("--backend")
    os.environ["ROGER_BACKEND"] = backend


@pytest.fixture(autouse=True)
def set_random_seed():
    import numpy as np

    np.random.seed(42)
