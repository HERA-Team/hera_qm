from setuptools import setup
import os
import sys
import os.path as op
import json
from pathlib import Path

sys.path.append('hera_qm')

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
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup_args = {
    'name': 'hera_qm',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/hera_qm',
    'license': 'BSD',
    'description': 'HERA Data Quality Metrics.',
    'long_description': long_description,
    'long_description_content_type': 'text/markdown',
    'package_dir': {'hera_qm': 'hera_qm'},
    'packages': ['hera_qm'],
    'include_package_data': True,
    'scripts': ['scripts/ant_metrics_run.py',
                'scripts/auto_metrics_run.py',
                'scripts/xrfi_run.py',
                'scripts/firstcal_metrics_run.py',
                'scripts/auto_view.py',
                'scripts/omnical_metrics_run.py',
                'scripts/xrfi_apply.py',
                'scripts/delay_xrfi_h1c_idr2_1_run.py',
                'scripts/xrfi_h1c_run.py',
                'scripts/xrfi_day_threshold_run.py',
                'scripts/xrfi_h3c_idr2_1_run.py',
                'scripts/xrfi_h3ca_rtp_run.py',
                'scripts/xrfi_run_data_only.py'
                ],
    'package_data': {'hera_qm': data_files},
    'setup_requires': ['pytest-runner'],
    'install_requires': [
        'astropy>=3.2.3',
        'h5py',
        'pyyaml',
        'numpy>=1.10',
        'pyuvdata',
    ],
    'tests_require': ['pytest'],
    'zip_safe': False,
}


if __name__ == '__main__':
    setup(**setup_args)
