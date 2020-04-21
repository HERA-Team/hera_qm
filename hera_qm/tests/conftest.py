# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

"""Testing environment setup and teardown for pytest."""
import pytest
import urllib
from astropy.utils import iers
from astropy.time import Time


@pytest.fixture(autouse=True, scope="session")
def setup_and_teardown_package():
    """Make data/test directory to put test output files in."""

    # try to download the iers table. If it fails, turn off auto downloading for the tests
    # and turn it back on in teardown_package (done by extending auto_max_age)
    try:
        iers_a = iers.IERS_A.open(iers.IERS_A_URL)
        t1 = Time.now()
        t1.ut1
    except(urllib.error.URLError):
        iers.conf.auto_max_age = None

    # yield to allow tests to run
    yield

    iers.conf.auto_max_age = 30
