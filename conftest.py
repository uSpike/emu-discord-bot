import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run-expensive", action="store_true", default=False,
        help="run expensive tests that call external APIs"
    )

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "expensive: mark test as expensive"
    )

def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-expensive"):
        # Run all tests including expensive ones
        return

    skip_expensive = pytest.mark.skip(reason="need --run-expensive option to run")
    for item in items:
        if "expensive" in item.keywords:
            item.add_marker(skip_expensive)