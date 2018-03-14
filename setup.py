from setuptools import setup
import glob
import os.path as path
from os import listdir
import sys
import os
from hera_qm import version
import os.path as op
import json

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(op.join('hera_qm', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)


def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths
data_files = package_files('hera_qm', 'data')

setup_args = {
    'name': 'hera_qm',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/hera_qm',
    'license': 'BSD',
    'description': 'HERA Data Quality Metrics.',
    'package_dir': {'hera_qm': 'hera_qm'},
    'packages': ['hera_qm'],
    'include_package_data': True,
    'scripts': ['scripts/ant_metrics_run.py', 'scripts/xrfi_run.py',
                'scripts/firstcal_metrics_run.py', 'scripts/auto_view.py',
                'scripts/omnical_metrics_run.py', 'scripts/xrfi_apply.py',
                'scripts/delay_xrfi_run.py'],
    'version': version.version,
    'package_data': {'hera_qm': data_files},
    'install_requires': ['astropy>=2.0'],
    'zip_safe': False,
}


if __name__ == '__main__':
    apply(setup, (), setup_args)
