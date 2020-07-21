# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License
"""Module for general purpose utility functions."""

import re
import os
import warnings
import argparse
import numpy as np
from pyuvdata import UVData
from pyuvdata import utils as uvutils


def _bytes_to_str(inbyte):
    return inbyte.decode('utf8')


def _str_to_bytes(instr):
    return instr.encode('utf8')


# argument-generating function for *_run wrapper functions
def get_metrics_ArgumentParser(method_name):
    """Get an ArgumentParser instance for working with metrics wrappers.

    Parameters
    ----------
    method_name : {"ant_metrics", "firstcal_metrics", "omnical_metrics", "xrfi_run",
                   "xrfi_apply", "xrfi_h1c_run", "delay_xrfi_h1c_idr2_1_run",
                   "xrfi_h3c_idr2_1_run"}
        The target wrapper desired.

    Returns
    -------
    ap : argparse.ArgumentParser
        An argparse.ArgumentParser instance with the relevant options for the selected method

    """
    methods = ["ant_metrics", "firstcal_metrics", "omnical_metrics", "xrfi_h1c_run",
               "delay_xrfi_h1c_idr2_1_run", "xrfi_run", "xrfi_apply", "day_threshold_run",
               "xrfi_h3c_idr2_1_run", "xrfi_run_data_only"]
    if method_name not in methods:
        raise AssertionError('method_name must be one of {}'.format(','.join(methods)))

    ap = argparse.ArgumentParser()

    if method_name == 'ant_metrics':
        ap.prog = 'ant_metrics.py'
        ap.add_argument('data_files', metavar='files', type=str, nargs='*', default=[],
                        help='4-pol visibility files used to compute antenna metrics')
        ap.add_argument('--apriori_xants', type=int, nargs='*', default=[],
                        help='space-delimited list of integer antenna numbers to exclude apriori.')
        ap.add_argument('--crossCut', default=5.0, type=float,
                        help='Modified z-score cut for most cross-polarized antenna. Default 5 "sigmas"')
        ap.add_argument('--deadCut', default=5.0, type=float,
                        help='Modified z-score cut for most likely dead antenna. Default 5 "sigmas"')
        ap.add_argument('--skip_cross_pols', action='store_false',
                        dest='run_cross_pols', default=True,
                        help=('Sets boolean flag to False. Flag determines if '
                              'mean_Vij_cross_pol_metrics is run. '
                              'Default: True'))
        ap.add_argument('--run_cross_pols_only', action='store_true',
                        dest='run_cross_pols_only', default=False,
                        help=('Define if cross pol metrics are the *only* '
                              'metrics to be run. Default is False.'))
        ap.add_argument('--metrics_path', default='', type=str,
                        help='Path to save metrics file to. Default is same directory as file.')
        ap.add_argument('--extension', default='.ant_metrics.json', type=str,
                        help='Extension to be appended to the file name. Default is ".ant_metrics.json"')
        ap.add_argument("--clobber", default=False, action="store_true",
                        help='overwrites existing firstcal_metrics file (default False)')
        ap.add_argument('--Nbls_per_load', default=None, type=int,
                        help='Number of baselines to load simultaneously.')
        ap.add_argument('-q', '--quiet', action='store_false', dest='verbose', default=True,
                        help='Silence feedback to the command line.')
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
        ap.add_argument('--ex_ants', default=None, type=str,
                        help='Comma-separated list of antennas to exclude. Flags of visibilities '
                        'formed with these antennas will be set to True.')
        ap.add_argument('--metrics_file', default=None, type=str,
                        help='Metrics file that contains a list of excluded antennas. Flags of '
                        'visibilities formed with these antennas will be set to True.')
        ap.add_argument("--clobber", default=False, action="store_true",
                        help='overwrites existing files (default False)')
    elif method_name == 'xrfi_run_data_only':
        ap.prog = 'xrfi_run_data_only.py'
        ap.add_argument('--data_file', default=None, type=str, help='Raw visibility '
                        'data file to flag on.')
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
        ap.add_argument('--ex_ants', default=None, type=str,
                        help='Comma-separated list of antennas to exclude. Flags of visibilities '
                        'formed with these antennas will be set to True.')
        ap.add_argument('--metrics_file', default=None, type=str,
                        help='Metrics file that contains a list of excluded antennas. Flags of '
                        'visibilities formed with these antennas will be set to True.')
        ap.add_argument("--clobber", default=False, action="store_true",
                        help='overwrites existing files (default False)')
        ap.add_argument("--median_filter_cross", default=False, action="store_true",
                        help="performs a median filter on cross-correlations. Adds significantly to runtime.")
        ap.add_argument("--skip_mean_filter_cross", default=False, action="store_true",
                        help="save i/o by skipping mean filter on cross correlations.")
    elif method_name == 'xrfi_h3c_idr2_1_run':
        ap.prog = 'xrfi_h3c_idr2_1_run.py'
        ap.add_argument('--ocalfits_files', nargs='+', type=str, help='Omnical '
                        'calfits files to use to flag on gains and chisquared values.')
        ap.add_argument('--acalfits_files', nargs='+', type=str, help='Abscal '
                        'calfits files to use to flag on gains and chisquared values.')
        ap.add_argument('--model_files', nargs='+', type=str, help='Model visibility '
                        'files to flag on.')
        ap.add_argument('--data_files', nargs='+', type=str, help='Raw visibility '
                        'data files to flag on.')
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
        ap.add_argument('--ex_ants', default=None, type=str,
                        help='Comma-separated list of antennas to exclude. Flags of visibilities '
                        'formed with these antennas will be set to True.')
        ap.add_argument('--metrics_file', default=None, type=str,
                        help='Metrics file that contains a list of excluded antennas. Flags of '
                        'visibilities formed with these antennas will be set to True.')
        ap.add_argument("--clobber", default=False, action="store_true",
                        help='overwrites existing files (default False)')
    elif method_name == 'day_threshold_run':
        ap.prog = 'xrfi_day_threshold_run.py'
        ap.add_argument('data_files', type=str, nargs='+', help='List of paths to \
                        the raw data files which have been used to calibrate and \
                        rfi flag so far.')
        ap.add_argument('--nsig_f', default=7.0, type=float,
                        help='The number of sigma above which to flag channels. Default is 7.0.')
        ap.add_argument('--nsig_t', default=7.0, type=float,
                        help='The number of sigma above which to flag integrations. Default is 7.0.')
        ap.add_argument('--nsig_f_adj', default=3.0, type=float,
                        help='The number of sigma above which to flag channels if they neighbor \
                        flagged channels. Default is 3.0.')
        ap.add_argument('--nsig_t_adj', default=3.0, type=float,
                        help='The number of sigma above which to flag integrations if they neighbor \
                        flagged integrations. Default is 7.0.')
        ap.add_argument("--clobber", default=False, action="store_true",
                        help='If True, overwrite existing files. Default is False.')
        ap.add_argument("--run_if_first", default=None, type=str, help='only run \
                        day_threshold_run if the first item in the sorted data_files \
                        list matches run_if_first (default None means always run)')
        ap.add_argument("--skip_making_flagged_abs_calfits", default=False, action="store_true",
                        help='If True, skip flagging the abscal files.')
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
    mdict : dict
        Dictionary containing keys and data to pass to M&C.

        Structure is as follows:
            d['ant_metrics']: Dictionary with metric names
                d['ant_metrics'][metric]: list of lists, each containing [ant, pol, val]
            d['array_metrics']: Dictionary with metric names
                d['array_metrics'][metric]: Single metric value

    """
    mdict = {'ant_metrics': {}, 'array_metrics': {}}
    if ftype == 'ant':
        from hera_qm.ant_metrics import load_antenna_metrics
        data = load_antenna_metrics(filename)
        key2cat = {'final_metrics': 'ant_metrics',
                   'final_mod_z_scores': 'ant_metrics_mod_z_scores'}
        for key, category in key2cat.items():
            for met, array in data[key].items():
                metric = '_'.join([category, met])
                mdict['ant_metrics'][metric] = []
                for antpol, val in array.items():
                    mdict['ant_metrics'][metric].append([antpol[0], antpol[1], val])
            for met in ['crossed_ants', 'dead_ants', 'xants']:
                metric = '_'.join(['ant_metrics', met])
                mdict['ant_metrics'][metric] = []
                for antpol in data[met]:
                    mdict['ant_metrics'][metric].append([antpol[0], antpol[1], 1])
            mdict['ant_metrics']['ant_metrics_removal_iteration'] = []
            metric = 'ant_metrics_removal_iteration'
            for antpol, val in data['removal_iteration'].items():
                mdict['ant_metrics'][metric].append([antpol[0], antpol[1], val])

    elif ftype == 'firstcal':
        from hera_qm.firstcal_metrics import load_firstcal_metrics
        data = load_firstcal_metrics(filename)
        pol = str(data['pol'])
        met = 'firstcal_metrics_good_sol_' + pol
        mdict['array_metrics'][met] = data['good_sol']
        met = 'firstcal_metrics_agg_std_' + pol
        mdict['array_metrics'][met] = data['agg_std']
        met = 'firstcal_metrics_max_std_' + pol
        mdict['array_metrics'][met] = data['max_std']
        for met in ['ant_z_scores', 'ant_avg', 'ant_std']:
            metric = '_'.join(['firstcal_metrics', met])
            mdict['ant_metrics'][metric] = []
            for ant, val in data[met].items():
                mdict['ant_metrics'][metric].append([ant, pol, val])
        metric = 'firstcal_metrics_bad_ants'
        mdict['ant_metrics'][metric] = []
        for ant in data['bad_ants']:
            mdict['ant_metrics'][metric].append([ant, pol, 1.])
        metric = 'firstcal_metrics_rot_ants'
        mdict['ant_metrics'][metric] = []

        try:
            if data['rot_ants'] is not None:
                for ant in data['rot_ants']:
                    mdict['ant_metrics'][metric].append([ant, pol, 1.])
        except KeyError:
            # Old files simply did not have rot_ants
            pass

    elif ftype == 'omnical':
        from hera_qm.omnical_metrics import load_omnical_metrics
        full_mets = load_omnical_metrics(filename)
        pols = full_mets.keys()

        # iterate over polarizations (e.g. XX, YY, XY, YX)
        for pol in pols:
            # unpack metrics from full_mets
            metrics = full_mets[pol]

            # pack array metrics
            cat = 'omnical_metrics_'
            for met in ['chisq_tot_avg', 'chisq_good_sol', 'ant_phs_std_max', 'ant_phs_std_good_sol']:
                catmet = cat + met + '_{}'.format(pol)
                try:
                    if metrics[met] is None:
                        continue
                    mdict['array_metrics'][catmet] = metrics[met]
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
                    if catmet in mdict['ant_metrics']:
                        mdict['ant_metrics'][catmet].extend([[a, metrics['ant_pol'].lower(),
                                                              metrics[met][a]] for a in metrics[met]])
                    # if not, assign it
                    else:
                        mdict['ant_metrics'][catmet] = [[a, metrics['ant_pol'].lower(),
                                                         metrics[met][a]] for a in metrics[met]]
                except KeyError:
                    pass

    else:
        raise ValueError('Metric file type ' + ftype + ' is not recognized.')

    return mdict


def dynamic_slice(arr, slice_obj, axis=-1):
    """Dynamically slice an arr along the axis given in the slice object.

    Parameters
    ----------
    arr : ndarray
        The array to take a slice of.
    slice_obj : numpy slice object
        The slice object defining which subset of the array to get.
    axis : int, optional
        The axis along which to slice the array. Default is -1.

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
    return_ext : bool, optional
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
