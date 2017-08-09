# HERA Quality Metrics

[![Build Status](https://travis-ci.org/HERA-Team/hera_qm.svg?branch=master)](https://travis-ci.org/HERA-Team/hera_qm)
[![Coverage Status](https://coveralls.io/repos/github/HERA-Team/hera_qm/badge.svg?branch=master)](https://coveralls.io/github/HERA-Team/hera_qm?branch=master)

`hera_qm` is a python package for calculating quality metrics of HERA data.
It is integrated in the Real-Time Pipeline (RTP), automatically generating metrics
for all HERA data as it is taken. But `hera_qm` can also be used offline for
further analysis.

## Motivation
Data quality metrics are useful and needed throughout the analysis of interferometric data.
This repository is a centralized place for the HERA team to develop metrics to 1)
run on data in the RTP and deliver to the wider collaboration; 2) store these metrics
in the Monitor and Control database for easy access; and 3) use offline in individual
analyses. As a consequence of the first two goals, contributions to `hera_qm` will
be vetted by the community and require thorough unittests. However, the code base
will also be flexible to enable the third goal, and we welcome contributions (see below).

## Installation

### Dependencies
First install dependencies.

* numpy >= 1.10
* scipy
* astropy >= 2.0
* aipy
* pyuvdata (`pip install pyuvdata` or use https://github.com/HERA-Team/pyuvdata.git)
* hera_cal (https://github.com/HERA-Team/hera_cal.git)
* omnical (https://github.com/HERA-Team/omnical.git)
* linsolve (https://github.com/HERA-Team/linsolve.git)

For anaconda users, we suggest using conda to install astropy, numpy and scipy and conda-forge
for aipy (```conda install -c conda-forge aipy```).

### Installing hera_qm
Clone the repo using
`git clone https://github.com/HERA-Team/hera_qm.git`

Navigate into the directory and run `python setup.py install`.

### Running Tests
Requires installation of `nose` package.
From the source `hera_qm` directory run: `nosetests hera_qm`.

## Package Details and Usage
There are currently three primary modules which drive HERA quality metrics.

### ant_metrics
A module to handle visibility-based metrics designed to identify misbehaving antennas.
The primary class, `Antenna_Metrics` includes methods to calculate several metrics
to isolate cross-polarized antennas, dead antennas, or antennas which are forming
non-redundant visibilities when they should be. And example of using this module
is in `scripts/ant_metrics_example_notebook.ipynb`.

### firstcal_metrics
A module to calculate metrics based on firstcal delay solutions. These metrics
identify large variations in delay solutions across time or across the array
for a given time. An example of using this module is in `scripts/firstcal_metrics.ipynb`.

### xrfi
This module contains several algorithms for radio frequency interference (RFI)
detection and flagging. `xrfi.xrfi` provides the best RFI excision we currently have.
The function `xrfi.xrfi_run` demonstrates how to use and apply the RFI algorithms
to a UVData object.

## Known Issues and Planned Improvements
Issues are tracked in the [issue log](https://github.com/HERA-Team/hera_qm/issues).
Major current issues and planned improvements include:
* A unified metric class structure
* Develop Tsys calculations into metrics (HERA Memos 16 and 34)
* Develop closure quantities into metrics (HERA Memo 15)

## Contributing
Contributions to this package to introduce new functionality or address any of the
issues in the [issue log](https://github.com/HERA-Team/hera_qm/issues) are very welcome.
Please submit improvements as pull requests against the repo after verifying that
the existing tests pass and any new code is well covered by unit tests.

Bug reports or feature requests are also very welcome, please add them to the
issue log after verifying that the issue does not already exist.
Comments on existing issues are also welcome.
