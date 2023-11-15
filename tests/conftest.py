"""Pytest fixtures for the whole repo."""
from pathlib import Path

import pytest
from pyinstrument import Profiler

TESTS_ROOT = Path(__file__).parent


@pytest.fixture(autouse=True)
def auto_profile(request):
    """Automatically run performance profiling on all tests.

    This fixture will run the profiler on all tests, and save the results to an html file in the .profiles directory.
    """
    profile_root = TESTS_ROOT / ".profiles"
    # Turn profiling on.
    profiler = Profiler()
    profiler.start()

    yield  # Run test.

    profiler.stop()
    profiler.print(color=True)
    profile_root.mkdir(exist_ok=True)
    results_file = profile_root / f"{request.node.name}.html"
    profiler.write_html(results_file)
