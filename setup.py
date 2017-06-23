from setuptools import setup
import glob
import os.path as path
from os import listdir

__version__ = '0.0.0'

setup_args = {
    'name': 'hera_qm',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/hera_qm',
    'license': 'BSD',
    'description': 'HERA Data Quality Metrics.',
    'package_dir': {'hera_qm': 'hera_qm'},
    'packages': ['hera_qm'],
    #    'scripts': glob.glob('scripts/*'),
    'version': __version__,
    #'package_data': {'hera_qm': ['data/*', 'data/*/*', 'calibrations/*']},
    #    'install_requires': ['numpy>=1.10', 'scipy', 'pyuvdata', 'astropy>1.2', 'aipy']
    #    'dependency_links': ['https://github.com/zakiali/omnical/tarball/master#egg=omnical-dev',]
    'zip_safe': False,
}


if __name__ == '__main__':
    apply(setup, (), setup_args)
