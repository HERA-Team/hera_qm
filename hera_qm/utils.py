# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License
"""Module for general purpose utility functions."""

from __future__ import print_function, division, absolute_import
from functools import reduce
import re
import os
import warnings
import argparse
import numpy as np
from pyuvdata import UVCal, UVData
from pyuvdata import telescopes as uvtel
from pyuvdata import utils as uvutils
import copy
from six.moves import range
import six

if six.PY2:
    def _bytes_to_str(b):
        return b

    def _str_to_bytes(s):
        return s
else:
    def _bytes_to_str(b):
        return b.decode('utf8')

    def _str_to_bytes(s):
        return s.encode('utf8')


# argument-generating function for *_run wrapper functions
def get_metrics_ArgumentParser(method_name):
    """Get an ArgumentParser instance for working with metrics wrappers.

    Parameters
    ----------
    method_name : {"ant_metrics", "firstcal_metrics", "omnical_metrics", "xrfi_run",
                   "xrfi_apply", "xrfi_h1c_run", "delay_xrfi_h1c_idr2_1_run"}
        The target wrapper desired.

    Returns
    -------
    ap
        An argparse.ArgumentParser instance with the relevant options for the selected method
    """
    methods = ["ant_metrics", "firstcal_metrics", "omnical_metrics", "xrfi_h1c_run",
               "delay_xrfi_h1c_idr2_1_run", "xrfi_run", "xrfi_apply"]
    if method_name not in methods:
        raise AssertionError('method_name must be one of {}'.format(','.join(methods)))

    ap = argparse.ArgumentParser()

    if method_name == 'ant_metrics':
        ap.prog = 'ant_metrics.py'
        ap.add_argument('-p', '--pol', default='', type=str,
                        help="Comma-separated list of polarizations included. Default is ''")
        ap.add_argument('--crossCut', default=5.0, type=float,
                        help='Modified z-score cut for most cross-polarized antenna. Default 5 "sigmas"')
        ap.add_argument('--deadCut', default=5.0, type=float,
                        help='Modified z-score cut for most likely dead antenna. Default 5 "sigmas"')
        ap.add_argument('--alwaysDeadCut', default=10.0, type=float,
                        help='Modified z-score cut for antennas that are definitely dead. Default 10 "sigmas".'
                        'These are all thrown away at once without waiting to iteratively throw away only the worst offender.')
        ap.add_argument('--extension', default='.ant_metrics.json', type=str,
                        help='Extension to be appended to the file name. Default is ".ant_metrics.json"')
        ap.add_argument('--metrics_path', default='', type=str,
                        help='Path to save metrics file to. Default is same directory as file.')
        ap.add_argument('--vis_format', default='miriad', type=str,
                        help='File format for visibility files. Default is miriad.')
        ap.add_argument('-q', '--quiet', action='store_false', dest='verbose', default=True,
                        help='Silence feedback to the command line.')
        ap.add_argument('files', metavar='files', type=str, nargs='*', default=[],
                        help='*.uv files for which to calculate ant_metrics.')

        ap.add_argument('--run_mean_vij', action='store_true',
                        dest='run_mean_vij',
                        help=('Sets boolean flag to True. Flag determines if '
                              'mean_vij_metrics is run. Default: True'))
        ap.add_argument('--skip_mean_vij', action='store_false',
                        dest='run_mean_vij',
                        help=('Sets boolean flag to False. Flag determines if '
                              'mean_vij_metrics is run. Default: True'))
        ap.set_defaults(run_mean_vij=True)

        ap.add_argument('--run_red_corr', action='store_true',
                        dest='run_red_corr',
                        help=('Sets boolean flag to True. Flag determines if '
                              'red_corr_metrics is run. Default: True'))
        ap.add_argument('--skip_red_corr', action='store_false',
                        dest='run_red_corr',
                        help=('Sets boolean flag to False. Flag determines if '
                              'red_corr_metrics is run. Default: True'))
        ap.set_defaults(run_red_corr=True)

        ap.add_argument('--run_cross_pols', action='store_true',
                        dest='run_cross_pols',
                        help=('Sets boolean flag to True. Flag determines if '
                              'mean_Vij_cross_pol_metrics and '
                              'red_corr_cross_pol_metrics are run. '
                              'Default: True'))
        ap.add_argument('--skip_cross_pols', action='store_false',
                        dest='run_cross_pols',
                        help=('Sets boolean flag to False. Flag determines if '
                              'mean_Vij_cross_pol_metrics and '
                              'red_corr_cross_pol_metrics are run. '
                              'Default: True'))
        ap.set_defaults(run_cross_pols=True)

    elif method_name == 'firstcal_metrics':
        ap.prog = 'firstcal_metrics.py'
        ap.add_argument('--std_cut', default=0.5, type=float,
                        help='Delay standard deviation cut for good / bad determination. Default 0.5')
        ap.add_argument('--extension', default='.firstcal_metrics.hdf5', type=str,
                        help='Extension to be appended to the file name. Default is ".firstcal_metrics.json"')
        ap.add_argument('--filetype', default='hdf5', type=str,
                        help='Filetype used in write_metrics call, filetype should match the type at the end of extension argument')
        ap.add_argument('--metrics_path', default='', type=str,
                        help='Path to save metrics file to. Default is same directory as file.')
        ap.add_argument("--clobber", default=False, action="store_true",
                        help='overwrites existing firstcal_metrics file (default False)')
        ap.add_argument('files', metavar='files', type=str, nargs='*', default=[],
                        help='*.calfits files for which to calculate firstcal_metrics.')
    elif method_name == 'omnical_metrics':
        ap.prog = 'omnical_metrics.py'
        ap.add_argument('--fc_files', metavar='fc_files', type=str, default=None,
                        help='[optional] *.first.calfits files of firstcal solutions to perform omni-firstcal comparison metrics.'
                        ' If multiple pols exist in a single *.omni.calfits file, feed .pol.first.calfits fcfiles as comma-delimited.'
                        ' If operating on multiple .omni.calfits files, feed separate comma-delimited fcfiles as vertical bar-delimited.'
                        ' Ex1: omnical_metrics_run.py --fc_files=zen.xx.first.calfits,zen.yy.first.calfits zen.omni.calfits'
                        ' Ex2: omnical_metrics_run.py --fc_files=zen1.xx.first.calfits,zen1.yy.first.calfits|zen2.xx.first.calfits,zen2.yy.first.calfits zen1.omni.calfits zen2.omni.calfits')
        ap.add_argument('--no_bandcut', action='store_true', default=False,
                        help="flag to turn off cutting of frequency band edges before calculating metrics")
        ap.add_argument('--phs_std_cut', type=float, default=0.3,
                        help="set gain phase stand dev cut. see OmniCal_Metrics.run_metrics() for details.")
        ap.add_argument('--chisq_std_zscore_cut', type=float, default=4.0,
                        help="set chisq stand dev. zscore cut. see OmniCal_Metrics.run_metrics() for details.")
        ap.add_argument('--make_plots', action='store_true', default=False,
                        help="make .png plots of metrics")
        ap.add_argument('--extension', default='.omni_metrics.json', type=str,
                        help='Extension to be appended to the metrics file name. Default is ".omni_metrics.json"')
        ap.add_argument('--metrics_path', default='', type=str,
                        help='Path to save metrics file to. Default is same directory as file.')
        ap.add_argument('files', metavar='files', type=str, nargs='*', default=[],
                        help='*.omni.calfits files for which to calculate omnical_metrics.')
    elif method_name == 'xrfi_h1c_run':
        ap.prog = 'xrfi_h1c_run.py'
        ap.add_argument('--infile_format', default='miriad', type=str,
                        help='File format for input files. Not currently used while '
                        'we use generic read function in pyuvdata, But will '
                        'be implemented for partial io.')
        ap.add_argument('--extension', default='flags.h5', type=str,
                        help='Extension to be appended to input file name. Default is "flags.h5".')
        ap.add_argument('--summary', action='store_true', default=False,
                        help='Run summary of RFI flags and store in npz file.')
        ap.add_argument('--summary_ext', default='flag_summary.h5',
                        type=str, help='Extension to be appended to input file name'
                        ' for summary file. Default is "flag_summary.h5"')
        ap.add_argument('--xrfi_path', default='', type=str,
                        help='Path to save flag files to. Default is same directory as input file.')
        ap.add_argument('--algorithm', default='xrfi_simple', type=str,
                        help='RFI-flagging algorithm to use. Default is xrfi_simple.')
        ap.add_argument('--model_file', default=None, type=str, help='Model visibility '
                        'file to flag on.')
        ap.add_argument('--model_file_format', default='uvfits', type=str,
                        help='File format for input files. Not currently used while '
                        'we use generic read function in pyuvdata, But will '
                        'be implemented for partial io.')
        ap.add_argument('--calfits_file', default=None, type=str, help='Calfits file '
                        'to use to flag on gains and/or chisquared values.')
        ap.add_argument('--nsig_df', default=6.0, type=float, help='Number of sigma '
                        'above median value to flag in f direction for xrfi_simple. Default is 6.')
        ap.add_argument('--nsig_dt', default=6.0, type=float,
                        help='Number of sigma above median value to flag in t direction'
                        ' for xrfi_simple. Default is 6.')
        ap.add_argument('--nsig_all', default=0.0, type=float,
                        help='Number of overall sigma above median value to flag'
                        ' for xrfi_simple. Default is 0 (skip).')
        ap.add_argument('--kt_size', default=8, type=int,
                        help='Size of kernel in time dimension for detrend in xrfi '
                        'algorithm. Default is 8.')
        ap.add_argument('--kf_size', default=8, type=int,
                        help='Size of kernel in frequency dimension for detrend in '
                        'xrfi algorithm. Default is 8.')
        ap.add_argument('--sig_init', default=6.0, type=float,
                        help='Starting number of sigmas to flag on. Default is 6.')
        ap.add_argument('--sig_adj', default=2.0, type=float,
                        help='Number of sigmas to flag on for data adjacent to a flag. Default is 2.')
        ap.add_argument('--px_threshold', default=0.2, type=float,
                        help='Fraction of flags required to trigger a broadcast across'
                        ' baselines for a given (time, frequency) pixel. Default is 0.2.')
        ap.add_argument('--freq_threshold', default=0.5, type=float,
                        help='Fraction of channels required to trigger broadcast across'
                        ' frequency (single time). Default is 0.5.')
        ap.add_argument('--time_threshold', default=0.05, type=float,
                        help='Fraction of times required to trigger broadcast across'
                        ' time (single frequency). Default is 0.05.')
        ap.add_argument('--ex_ants', default=None, type=str,
                        help='Comma-separated list of antennas to exclude. Flags of visibilities '
                        'formed with these antennas will be set to True.')
        ap.add_argument('--metrics_file', default=None, type=str,
                        help='Metrics file that contains a list of excluded antennas. Flags of '
                        'visibilities formed with these antennas will be set to True.')
        ap.add_argument('filename', metavar='filename', nargs='*', type=str, default=[],
                        help='file for which to flag RFI (only one file allowed).')
    elif method_name == 'delay_xrfi_h1c_idr2_1_run':
        ap.prog = 'delay_xrfi_h1c_idr2_1_run.py'
        ap.add_argument('filename', metavar='filename', nargs='?', type=str, default=None,
                        help='file for which to flag RFI (only one file allowed).')
        xg = ap.add_argument_group(title='XRFI options', description='Options related to '
                                   'the RFI flagging routine.')
        xg.add_argument('--infile_format', default='miriad', type=str,
                        help='File format for input files. DEPRECATED, but kept for legacy.')
        xg.add_argument('--extension', default='flags.h5', type=str,
                        help='Extension to be appended to input file name. Default is "flags.h5".')
        xg.add_argument('--summary', action='store_true', default=False,
                        help='Run summary of RFI flags and store in npz file.')
        xg.add_argument('--summary_ext', default='flag_summary.h5',
                        type=str, help='Extension to be appended to input file name'
                        ' for summary file. Default is "flag_summary.h5"')
        xg.add_argument('--xrfi_path', default='', type=str,
                        help='Path to save flag files to. Default is same directory as input file.')
        xg.add_argument('--algorithm', default='xrfi_simple', type=str,
                        help='RFI-flagging algorithm to use. Default is xrfi_simple.')
        xg.add_argument('--model_file', default=None, type=str, help='Model visibility '
                        'file to flag on.')
        xg.add_argument('--model_file_format', default='uvfits', type=str,
                        help='File format for input files. DEPRECATED, but kept for legacy.')
        xg.add_argument('--calfits_file', default=None, type=str, help='Calfits file '
                        'to use to flag on gains and/or chisquared values.')
        xg.add_argument('--nsig_df', default=6.0, type=float, help='Number of sigma '
                        'above median value to flag in f direction for xrfi_simple. Default is 6.')
        xg.add_argument('--nsig_dt', default=6.0, type=float,
                        help='Number of sigma above median value to flag in t direction'
                        ' for xrfi_simple. Default is 6.')
        xg.add_argument('--nsig_all', default=0.0, type=float,
                        help='Number of overall sigma above median value to flag'
                        ' for xrfi_simple. Default is 0 (skip).')
        xg.add_argument('--kt_size', default=8, type=int,
                        help='Size of kernel in time dimension for detrend in xrfi '
                        'algorithm. Default is 8.')
        xg.add_argument('--kf_size', default=8, type=int,
                        help='Size of kernel in frequency dimension for detrend in '
                        'xrfi algorithm. Default is 8.')
        xg.add_argument('--sig_init', default=6.0, type=float,
                        help='Starting number of sigmas to flag on. Default is 6.')
        xg.add_argument('--sig_adj', default=2.0, type=float,
                        help='Number of sigmas to flag on for data adjacent to a flag. Default is 2.')
        xg.add_argument('--px_threshold', default=0.2, type=float,
                        help='Fraction of flags required to trigger a broadcast across'
                        ' baselines for a given (time, frequency) pixel. Default is 0.2.')
        xg.add_argument('--freq_threshold', default=0.5, type=float,
                        help='Fraction of channels required to trigger broadcast across'
                        ' frequency (single time). Default is 0.5.')
        xg.add_argument('--time_threshold', default=0.05, type=float,
                        help='Fraction of times required to trigger broadcast across'
                        ' time (single frequency). Default is 0.05.')
        xg.add_argument('--ex_ants', default='', type=str,
                        help='Comma-separated list of antennas to exclude. Flags of visibilities '
                        'formed with these antennas will be set to True.')
        xg.add_argument('--metrics_file', default='', type=str,
                        help='Metrics file that contains a list of excluded antennas. Flags of '
                        'visibilities formed with these antennas will be set to True.')
        dg = ap.add_argument_group(title='Delay filter options', description='Options '
                                   'related to the delay filter which is applied before flagging.')
        dg.add_argument("--standoff", type=float, default=15.0, help='fixed additional delay beyond the horizon (default 15 ns)')
        dg.add_argument("--horizon", type=float, default=1.0, help='proportionality constant for bl_len where 1.0 (default) is the horizon '
                        '(full light travel time)')
        dg.add_argument("--tol", type=float, default=1e-7, help='CLEAN algorithm convergence tolerance (default 1e-7). '
                        'NOTE: default is different from default when running delay_filter_run.py.')
        dg.add_argument("--window", type=str, default="tukey", help='window function for frequency filtering (default "tukey", '
                        'see aipy.dsp.gen_window for options')
        dg.add_argument("--skip_wgt", type=float, default=0.1, help='skips filtering rows with unflagged fraction ~< skip_wgt (default 0.1)')
        dg.add_argument("--maxiter", type=int, default=100, help='maximum iterations for aipy.deconv.clean to converge (default 100)')
        dg.add_argument("--alpha", type=float, default=.5, help='alpha parameter to use for Tukey window (ignored if window is not Tukey)')
        dg.add_argument('--waterfalls', default=None, type=str, help='comma separated '
                        'list of npz files containing waterfalls of flags to broadcast '
                        'to full flag array and apply before delay filter.')
    elif method_name == 'xrfi_run':
        ap.prog = 'xrfi_run.py'
        ap.add_argument('--ocalfits_file', default=None, type=str, help='Omnical '
                        'calfits file to use to flag on gains and chisquared values.')
        ap.add_argument('--acalfits_file', default=None, type=str, help='Abscal '
                        'calfits file to use to flag on gains and chisquared values.')
        ap.add_argument('--model_file', default=None, type=str, help='Model visibility '
                        'file to flag on.')
        ap.add_argument('--data_file', default=None, type=str, help='Raw visibility '
                        'data file to flag on.')
        ap.add_argument('--init_metrics_ext', default='init_xrfi_metrics.h5', type=str,
                        help='Extension to be appended to input file name '
                        'for initial metric object. Default is "init_xrfi_metrics.h5".')
        ap.add_argument('--init_flags_ext', default='init_flags.h5', type=str,
                        help='Extension to be appended to input file name '
                        'for initial flag object. Default is "init_flags.h5".')
        ap.add_argument('--final_metrics_ext', default='final_xrfi_metrics.h5', type=str,
                        help='Extension to be appended to input file name '
                        'for final metric object. Default is "final_xrfi_metrics.h5".')
        ap.add_argument('--final_flags_ext', default='final_flags.h5', type=str,
                        help='Extension to be appended to input file name '
                        'for final flag object. Default is "final_flags.h5".')
        ap.add_argument('--xrfi_path', default='', type=str,
                        help='Path to save flag files to. Default is same directory as input file.')
        ap.add_argument('--kt_size', default=8, type=int,
                        help='Size of kernel in time dimension for detrend in xrfi '
                        'algorithm. Default is 8.')
        ap.add_argument('--kf_size', default=8, type=int,
                        help='Size of kernel in frequency dimension for detrend in '
                        'xrfi algorithm. Default is 8.')
        ap.add_argument('--sig_init', default=6.0, type=float,
                        help='Starting number of sigmas to flag on. Default is 6.0.')
        ap.add_argument('--sig_adj', default=2.0, type=float,
                        help='Number of sigmas to flag on for data adjacent to a flag. Default is 2.0.')
        ap.add_argument('--freq_threshold', default=0.35, type=float,
                        help='Fraction of times required to trigger broadcast across'
                        ' times (single freq). Default is 0.35.')
        ap.add_argument('--time_threshold', default=0.5, type=float,
                        help='Fraction of channels required to trigger broadcast across'
                        ' frequency (single time). Default is 0.5.')
        ap.add_argument('--ex_ants', default=None, type=str,
                        help='Comma-separated list of antennas to exclude. Flags of visibilities '
                        'formed with these antennas will be set to True.')
        ap.add_argument('--metrics_file', default=None, type=str,
                        help='Metrics file that contains a list of excluded antennas. Flags of '
                        'visibilities formed with these antennas will be set to True.')
        ap.add_argument('--cal_ext', default='flagged_abs', type=str,
                        help='Extension to replace penultimate extension in calfits '
                        'file for output calibration including flags. Defaults is '
                        '"flagged_abs". For example, a input_cal of "foo.goo.calfits" '
                        'would result in "foo.flagged_abs.calfits".')
        ap.add_argument("--clobber", default=False, action="store_true",
                        help='overwrites existing files (default False)')
    elif method_name == 'xrfi_apply':
        ap.prog = 'xrfi_apply.py'
        ap.add_argument('--infile_format', default='miriad', type=str,
                        help='File format for input files. Not currently used while '
                        'we use generic read function in pyuvdata, But will '
                        'be implemented for partial io.')
        ap.add_argument('--xrfi_path', default='', type=str,
                        help='Path to save output to. Default is same directory as input file.')
        ap.add_argument('--outfile_format', default='miriad', type=str,
                        help='File format for output files. Default is miriad.')
        ap.add_argument('--extension', default='R', type=str,
                        help='Extension to be appended to input file name. Default is "R".')
        ap.add_argument('--overwrite', action='store_true', default=False,
                        help='Option to overwrite output file if it already exists.')
        ap.add_argument('--flag_file', default=None, type=str, help='npz file '
                        'containing full flag array to insert into data file.')
        ap.add_argument('--waterfalls', default=None, type=str, help='comma separated '
                        'list of npz files containing waterfalls of flags to broadcast '
                        'to full flag array and union with flag array in flag_file.')
        ap.add_argument('--output_uvflag', default=True, type=bool,
                        help='Whether to save a uvflag object with the final flag array. '
                        'The flag array will be identical to what is stored in the data.')
        ap.add_argument('--output_uvflag_ext', default='flags.h5', type=str,
                        help='Extension to be appended to input file name. Default is "flags.h5".')
        ap.add_argument('filename', metavar='filename', nargs='*', type=str, default=[],
                        help='file for which to flag RFI (only one file allowed).')
    return ap


def get_pol(fname):
    """Strip the filename of a HERA visibility to its polarization.

    Note
    ----
    This function assumes the file naming format:
        zen.ddddddd.ddddd.pp.*
    The 'd' values are the 7-digit Julian day and 5-digit fractional Julian
    date. The 'pp' is the polarization extracted. It need not be 2 characters,
    and the parser will return everying between the two periods.

    Parameters
    ----------
    fname : str
        The name of a file. Note that just the file name should be passed in, to
        avoid pathological problems like directories that may match the structure
        being searched for.

    Returns
    -------
    polarization : str
        The polarization label contained in the filename, e.g., "xx"
    """
    fn = re.findall(r'zen\.\d{7}\.\d{5}\..*', fname)[0]
    return fn.split('.')[3]


def generate_fullpol_file_list(files, pol_list):
    """Generate a list of unique JDs that have all four polarizations available.

    This function, when given a list of files, will look for the specified polarizations,
    and add the JD to the returned list if all polarizations were found. The return is a
    list of lists, where the outer list is a single JD and the inner list is a "full set"
    of polarizations, based on the polarization list provided.

    Parameters
    ----------
    files : list
        The list of files to look for.
    pol_list : list
        The list of polarizations to look for, as strings (e.g., ['xx', 'xy', 'yx',
        'yy']).

    Returns
    -------
    jd_list : list
        The list of lists of JDs where all supplied polarizations could be found.

    """
    # initialize
    file_list = []

    # Check if all input files are full-pol files
    # if so return the input files as the full list
    uvd = UVData()

    for filename in files:
        if filename.split('.')[-1] == 'uvh5':
            uvd.read_uvh5(filename, read_data=False)
        else:
            uvd.read(filename)
        # convert the polarization array to strings and compare with the
        # expected input.
        # If anyone file is not a full-pol file then this will be false.
        input_pols = uvutils.polnum2str(uvd.polarization_array)
        full_pol_check = np.array_equal(np.sort(input_pols), np.sort(pol_list))

        if not full_pol_check:
            # if a file has more than one polarization but not all expected pols
            # raise an error that mixed pols are not allowed.
            if len(input_pols) > 1:
                base_fname = os.path.basename(filename)
                raise ValueError("The file: {fname} contains {npol} "
                                 "polarizations: {pol}. "
                                 "Currently only full lists of all expected "
                                 "polarization files or lists of "
                                 "files with single polarizations in the "
                                 "name of the file (e.g. zen.JD.pol.HH.uv) "
                                 "are allowed.".format(fname=base_fname,
                                                       npol=len(input_pols),
                                                       pol=input_pols))

            else:
                # if only one polarization then try the old regex method
                # assumes all files have the same number of polarizations
                break
    del uvd

    if full_pol_check:
        # Output of this function is a list of lists of files
        # We expect all full pol files to be unique JDs so
        # turn the list of files into a list of lists of each file.
        return [[f] for f in files]

    for filename in files:
        abspath = os.path.abspath(filename)
        # need to loop through groups of JDs already present
        in_list = False
        for jd_list in file_list:
            if abspath in jd_list:
                in_list = True
                break
        if not in_list:
            # try to find the other polarizations
            pols_exist = True
            file_pol = get_pol(filename)
            dirname = os.path.dirname(abspath)
            for pol in pol_list:
                # guard against strange directory names that might contain something that
                # looks like a pol string
                fn = re.sub(file_pol, pol, filename)
                full_filename = os.path.join(dirname, fn)
                if not os.path.exists(full_filename):
                    warnings.warn("Could not find " + full_filename + "; skipping that JD")
                    pols_exist = False
                    break
            if pols_exist:
                # add all pols to file_list
                jd_list = []
                for pol in pol_list:
                    fn = re.sub(file_pol, pol, filename)
                    full_filename = os.path.join(dirname, fn)
                    jd_list.append(full_filename)
                file_list.append(jd_list)

    return file_list


def get_metrics_dict():
    """Get a dictionary of combined metrics lists from hera_qm modules.

    Returns
    -------
    metrics_dict : dict
        Dictionary of all metrics and descriptions to be used in M&C database.

    """
    from hera_qm.ant_metrics import get_ant_metrics_dict
    from hera_qm.firstcal_metrics import get_firstcal_metrics_dict
    from hera_qm.omnical_metrics import get_omnical_metrics_dict
    metrics_dict = get_ant_metrics_dict()
    metrics_dict.update(get_firstcal_metrics_dict())
    metrics_dict.update(get_omnical_metrics_dict())
    return metrics_dict


def metrics2mc(filename, ftype):
    """Read in a metrics file and make contents suitable for ingestion into M&C.

    This function reads in a file containing quality metrics and stuffs them into
    a dictionary which can be used by M&C to populate the db.

    If one wishes to add a metric to the list that is tracked by M&C, it is
    (unfortunately) currently a four step process:
    1) Ensure your metric is written to the output files of the relevant module.
    2) Add the metric and a description to the get_X_metrics_dict() function in
       said module.
    3) Check that the metric is appropriately ingested in this function, and make
       changes as necessary.
    4) Add unit tests! Also check that the hera_mc tests still pass.

    Parameters
    ----------
    filename : str
        The path to the file to read and convert.
    ftype : {"ant", "firstcal", "omnical"}
        The type of metrics file.

    Returns
    -------
    d : dict
        Dictionary containing keys and data to pass to M&C.

        Structure is as follows:
            d['ant_metrics']: Dictionary with metric names
                d['ant_metrics'][metric]: list of lists, each containing [ant, pol, val]
            d['array_metrics']: Dictionary with metric names
                d['array_metrics'][metric]: Single metric value

    """
    d = {'ant_metrics': {}, 'array_metrics': {}}
    if ftype == 'ant':
        from hera_qm.ant_metrics import load_antenna_metrics
        data = load_antenna_metrics(filename)
        key2cat = {'final_metrics': 'ant_metrics',
                   'final_mod_z_scores': 'ant_metrics_mod_z_scores'}
        for key, category in key2cat.items():
            for met, array in data[key].items():
                metric = '_'.join([category, met])
                d['ant_metrics'][metric] = []
                for antpol, val in array.items():
                    d['ant_metrics'][metric].append([antpol[0], antpol[1], val])
            for met in ['crossed_ants', 'dead_ants', 'xants']:
                metric = '_'.join(['ant_metrics', met])
                d['ant_metrics'][metric] = []
                for antpol in data[met]:
                    d['ant_metrics'][metric].append([antpol[0], antpol[1], 1])
            d['ant_metrics']['ant_metrics_removal_iteration'] = []
            metric = 'ant_metrics_removal_iteration'
            for antpol, val in data['removal_iteration'].items():
                d['ant_metrics'][metric].append([antpol[0], antpol[1], val])

    elif ftype == 'firstcal':
        from hera_qm.firstcal_metrics import load_firstcal_metrics
        data = load_firstcal_metrics(filename)
        pol = str(data['pol'])
        met = 'firstcal_metrics_good_sol_' + pol
        d['array_metrics'][met] = data['good_sol']
        met = 'firstcal_metrics_agg_std_' + pol
        d['array_metrics'][met] = data['agg_std']
        met = 'firstcal_metrics_max_std_' + pol
        d['array_metrics'][met] = data['max_std']
        for met in ['ant_z_scores', 'ant_avg', 'ant_std']:
            metric = '_'.join(['firstcal_metrics', met])
            d['ant_metrics'][metric] = []
            for ant, val in data[met].items():
                d['ant_metrics'][metric].append([ant, pol, val])
        metric = 'firstcal_metrics_bad_ants'
        d['ant_metrics'][metric] = []
        for ant in data['bad_ants']:
            d['ant_metrics'][metric].append([ant, pol, 1.])
        metric = 'firstcal_metrics_rot_ants'
        d['ant_metrics'][metric] = []

        try:
            if data['rot_ants'] is not None:
                for ant in data['rot_ants']:
                    d['ant_metrics'][metric].append([ant, pol, 1.])
        except KeyError:
            # Old files simply did not have rot_ants
            pass

    elif ftype == 'omnical':
        from hera_qm.omnical_metrics import load_omnical_metrics
        full_mets = load_omnical_metrics(filename)
        pols = full_mets.keys()

        # iterate over polarizations (e.g. XX, YY, XY, YX)
        for i, pol in enumerate(pols):
            # unpack metrics from full_mets
            metrics = full_mets[pol]

            # pack array metrics
            cat = 'omnical_metrics_'
            for met in ['chisq_tot_avg', 'chisq_good_sol', 'ant_phs_std_max', 'ant_phs_std_good_sol']:
                catmet = cat + met + '_{}'.format(pol)
                try:
                    if metrics[met] is None:
                        continue
                    d['array_metrics'][catmet] = metrics[met]
                except KeyError:
                    pass

            # pack antenna metrics, only uses auto pol (i.e. XX or YY)
            if pol not in ['XX', 'YY']:
                continue
            cat = 'omnical_metrics_'
            for met in ['chisq_ant_avg', 'chisq_ant_std', 'ant_phs_std']:
                catmet = cat + met
                try:
                    if metrics[met] is None:
                        continue
                    # if catmet already exists extend it
                    if catmet in d['ant_metrics']:
                        d['ant_metrics'][catmet].extend([[a, metrics['ant_pol'].lower(), metrics[met][a]] for a in metrics[met]])
                    # if not, assign it
                    else:
                        d['ant_metrics'][catmet] = [[a, metrics['ant_pol'].lower(), metrics[met][a]] for a in metrics[met]]
                except KeyError:
                    pass

    else:
        raise ValueError('Metric file type ' + ftype + ' is not recognized.')

    return d


def dynamic_slice(arr, slice_obj, axis=-1):
    """Dynamically slice an arr along the axis given in the slice object.

    Parameters
    ----------
    arr : ndarray
        The array to take a slice of.
    slice_obj : numpy slice object
        The slice object defining which subset of the array to get.
    axis : int
        The axis along which to slice the array.

    Returns
    -------
    arr_slice : ndarray
        The slice of the arr as specified by the slice_obj.

    """
    if isinstance(axis, (int, np.integer)):
        axis = (axis,)
    if not isinstance(arr, np.ndarray):
        raise ValueError("arr must be an ndarray")
    slices = [slice(None) for i in range(arr.ndim)]
    for ax in axis:
        slices[ax] = slice_obj
    return arr[tuple(slices)]


def strip_extension(path, return_ext=False):
    """Remove the extension off a path.

    This function strips the extension off a path. Note this calls os.path.splitext,
    but we change the output slightly for convenience in our filename building.

    Parameters
    ----------
    path : str
        Path you wish to strip of its extension.
    return_ext : bool
        If True, return the extension as well. Default is False.

    Returns
    -------
    root : str
        The input path without its extension.
    ext : str, optional
        The extension of the input path (without the leading ".").
    """
    if return_ext:
        root, ext = os.path.splitext(path)
        return (root, ext[1:])
    else:
        return os.path.splitext(path)[0]
