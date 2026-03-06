import pytest


def pytest_collection_modifyitems(config, items):
    """Skip e2e-marked tests unless explicitly selected with -m e2e."""
    if "e2e" in (config.getoption("-m", default="") or ""):
        return
    skip = pytest.mark.skip(
        reason="E2e tests require external services. Run with: pytest -m e2e"
    )
    for item in items:
        if item.get_closest_marker("e2e"):
            item.add_marker(skip)
