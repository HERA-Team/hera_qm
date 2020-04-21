# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

"""Tests for version.py."""
import sys
from io import StringIO
import subprocess
import hera_qm


def test_construct_version_info():
    # this test is a bit silly because it uses the nearly the same code as the original,
    # but it will detect accidental changes that could cause problems.
    # It does test that the __version__ attribute is set on hera_qm.
    # I can't figure out how to test the except clause in construct_version_info.
    git_origin = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'],
                                         stderr=subprocess.STDOUT).strip().decode()
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       stderr=subprocess.STDOUT).strip().decode()
    git_description = subprocess.check_output(['git', 'describe', '--dirty', '--tags', '--always']).strip().decode()
    git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                         stderr=subprocess.STDOUT).strip().decode()

    test_version_info = {'version': hera_qm.__version__, 'git_origin': git_origin,
                         'git_hash': git_hash, 'git_description': git_description,
                         'git_branch': git_branch}

    assert hera_qm.version.construct_version_info() == test_version_info


def test_main():
    version_info = hera_qm.version.construct_version_info()

    saved_stdout = sys.stdout
    try:
        out = StringIO()
        sys.stdout = out
        hera_qm.version.main()
        output = out.getvalue()
        assert output == ('Version = {v}\ngit origin = {o}\n'
                          'git branch = {b}\ngit description = {d}\n'
                          .format(v=version_info['version'],
                                  o=version_info['git_origin'],
                                  b=version_info['git_branch'],
                                  d=version_info['git_description']))

    finally:
        sys.stdout = saved_stdout
