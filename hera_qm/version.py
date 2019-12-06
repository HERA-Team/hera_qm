# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License
"""Generate version information for hera_qm."""

import json
import os
import subprocess

hera_qm_dir = os.path.dirname(os.path.realpath(__file__))


def _get_git_output(args, capture_stderr=False):
    """Get output from Git.

    Get output from Git, ensuring that it is of the ``str`` type,
    not bytes. This prevents headaches when dealing with python2/3
    compatibilitiy.

    Parameters
    ----------
    args : list
        A list of command line arguments to pass to git.
    capture_stderr : bool, optional
        If True, capture stderr as well as stdout. Default is False.

    Returns
    -------
    data : str
        A string containing the output of the command passed to git.

    """
    argv = ['git', '-C', hera_qm_dir] + args

    if capture_stderr:
        data = subprocess.check_output(argv, stderr=subprocess.STDOUT)
    else:
        data = subprocess.check_output(argv)

    data = data.strip()

    return data.decode('utf8')


def construct_version_info():
    """Construct hera_qm version information.

    Returns
    -------
    version_info : dict
        A dictionary containing the version information of hera_qm. The keys
        are as follows:

        version: the version defined in the VERSION file of the hera_qm repo.
        git_origin: the origin of the git repo.
        git_hash: the git hash of the installed module.
        git_description: the description of the module as provided by git.
        git_branch: the currently checkout out branch.

    """
    version_file = os.path.join(hera_qm_dir, 'VERSION')
    version = open(version_file).read().strip()

    try:
        git_origin = _get_git_output(['config', '--get', 'remote.origin.url'], capture_stderr=True)
        git_hash = _get_git_output(['rev-parse', 'HEAD'], capture_stderr=True)
        git_description = _get_git_output(['describe', '--dirty', '--tag', '--always'])
        git_branch = _get_git_output(['rev-parse', '--abbrev-ref', 'HEAD'], capture_stderr=True)
    except subprocess.CalledProcessError:  # pragma: no cover  - can't figure out how to test exception.
        try:
            # Check if a GIT_INFO file was created when installing package
            git_file = os.path.join(hera_qm_dir, 'GIT_INFO')
            with open(git_file) as data_file:
                data = [x for x in json.loads(data_file.read().strip())]
                git_origin = data[0]
                git_hash = data[1]
                git_description = data[2]
                git_branch = data[3]
        except IOError:
            git_origin = ''
            git_hash = ''
            git_description = ''
            git_branch = ''

    version_info = {'version': version, 'git_origin': git_origin,
                    'git_hash': git_hash, 'git_description': git_description,
                    'git_branch': git_branch}
    return version_info


version_info = construct_version_info()
version = version_info['version']
git_origin = version_info['git_origin']
git_hash = version_info['git_hash']
git_description = version_info['git_description']
git_branch = version_info['git_branch']

# String to add to history of any files written with this version of pyuvdata
hera_qm_version_str = ('hera_qm version: ' + version + '.')
if git_hash != '':
    hera_qm_version_str += ('  Git origin: ' + git_origin
                            + '.  Git hash: ' + git_hash
                            + '.  Git branch: ' + git_branch
                            + '.  Git description: ' + git_description + '.')
hera_qm_version_str = str(hera_qm_version_str)


def main():
    """Print module version information and exit."""
    print('Version = {0}'.format(version))
    print('git origin = {0}'.format(git_origin))
    print('git branch = {0}'.format(git_branch))
    print('git description = {0}'.format(git_description))


if __name__ == '__main__':
    main()
