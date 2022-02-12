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
from pyuvdata import UVCal
from pyuvdata import UVFlag
from pyuvdata import utils as uvutils
from . import metrics_io
from pyuvdata.telescopes import KNOWN_TELESCOPES

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
    methods = ["ant_metrics", "auto_metrics", "firstcal_metrics", "omnical_metrics", "xrfi_h1c_run",
               "delay_xrfi_h1c_idr2_1_run", "xrfi_run", "xrfi_apply", "day_threshold_run",
               "xrfi_h3c_idr2_1_run", "xrfi_run_data_only"]
    if method_name not in methods:
        raise AssertionError('method_name must be one of {}'.format(','.join(methods)))

    ap = argparse.ArgumentParser()

    if method_name == 'ant_metrics':
        ap.prog = 'ant_metrics.py'
        ap.add_argument('sum_files', type=str, nargs='+',
                        help='4-pol visibility sum files used to compute antenna metrics')
        ap.add_argument('--diff_files', type=str, nargs='+', default=None,
                        help='4-pol visibility diff files used to compute antenna metrics. If not provided, even/odd will be formed from time interleaving.')
        ap.add_argument('--apriori_xants', type=int, nargs='*', default=[],
                        help='space-delimited list of integer antenna numbers to exclude apriori.')
        ap.add_argument('--a_priori_xants_yaml', type=str, default=None,
                        help=('path to a priori flagging YAML with xant information parsable by '
                              'hera_qm.metrics_io.read_a_priori_ant_flags()'))
        ap.add_argument('--crossCut', default=0.0, type=float,
                        help='Cut in cross-pol correlation metric below which to flag antennas as cross-polarized. Default 0.0.')
        ap.add_argument('--deadCut', default=0.4, type=float,
                        help='Cut in correlation metric below which antennas are most likely dead / not correlating. Default 0.4.')
        ap.add_argument('--metrics_path', default='', type=str,
                        help='Path to save metrics file to. Default is same directory as file.')
        ap.add_argument('--extension', default='.ant_metrics.json', type=str,
                        help='Extension to be appended to the file name. Default is ".ant_metrics.json"')
        ap.add_argument("--clobber", default=False, action="store_true",
                        help='overwrites existing ant_metrics file (default False)')
        ap.add_argument('--Nbls_per_load', default=None, type=int,
                        help='Number of baselines to load simultaneously.')
        ap.add_argument('--Nfiles_per_load', default=None, type=int,
                        help='Number of files to load simultaneously.')
        ap.add_argument('-q', '--quiet', action='store_false', dest='verbose', default=True,
                        help='Silence feedback to the command line.')

    elif method_name == 'auto_metrics':
        ap.prog = 'auto_metrics.py'
        ap.add_argument('metric_outfile', type=str,
                        help='Path to save auto_metrics hdf5 file.')
        ap.add_argument('raw_auto_files', type=str, nargs='+',
                        help='Paths to data files including autocorrelations.')
        ap.add_argument('--median_round_modz_cut', default=8., type=float,
                        help='Round 1 (median-based) cut on antenna modified Z-score.')
        ap.add_argument('--mean_round_modz_cut', default=4., type=float,
                        help='Round 2 (mean-based) cut on antenna modified Z-score.')
        ap.add_argument('--edge_cut', default=100, type=int,
                        help='Number of channels on either end to flag (i.e. ignore) when looking for antenna outliers.')
        ap.add_argument('--Kt', default=8, type=int,
                        help='Time kernel half-width for RFI flagging.')
        ap.add_argument('--Kf', default=8, type=int,
                        help='Frequency kernel half-width for RFI flagging.')
        ap.add_argument('--sig_init', default=5.0, type=float,
                        help='The number of sigmas above which to flag pixels.')
        ap.add_argument('--sig_adj', default=2.0, type=float,
                        help='The number of sigmas above which to flag pixels adjacent to flags.')
        ap.add_argument('--chan_thresh_frac', default=.05, type=float,
                        help='The fraction of flagged times (ignoring completely flagged times) above which to flag a whole channel.')
        ap.add_argument("--clobber", default=False, action="store_true",
                        help='Overwrites existing metric_outfile (default False).')

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
        ap.add_argument('--ocalfits_files', default=None, type=str, help='Omnical '
                        'calfits files to use to flag on gains and chisquared values.',
                        nargs='+')
        ap.add_argument('--acalfits_files', default=None, type=str, help='Abscal '
                        'calfits files to use to flag on gains and chisquared values.',
                        nargs='+')
        ap.add_argument('--model_files', default=None, type=str, help='Model visibility '
                        'files to flag on.',
                        nargs='+')
        ap.add_argument('--data_files', default=None, type=str, help='Raw visibility '
                        'data files to flag on.',
                        nargs='+')
        ap.add_argument('--a_priori_flag_yaml', default=None, type=str,
                        help=('Path to a priori flagging YAML with frequency, time, and/or '
                              'antenna flagsfor parsable by hera_qm.metrics_io.read_a_priori_*_flags()'))
        ap.add_argument('--a_apriori_times_and_freqs', default=False, action="store_true",
                        help="If True, ignore frequeny and time flags in apriori yaml file (only use ant flags).")
        ap.add_argument('--skip_cross_pol_vis', default=False, action="store_true",
                        help="Don't load or use cross-polarized visibilities, e.g. 'en' or 'ne', in flagging.")
        ap.add_argument('--xrfi_path', default='', type=str,
                        help='Path to save flag files to. Default is same directory as input file.')
        ap.add_argument('--kt_size', default=8, type=int,
                        help='Size of kernel in time dimension for detrend in xrfi '
                        'algorithm. Default is 8.')
        ap.add_argument('--kf_size', default=8, type=int,
                        help='Size of kernel in frequency dimension for detrend in '
                        'xrfi algorithm. Default is 8.')
        ap.add_argument('--sig_init_med', default=10.0, type=float,
                        help='Starting number of sigmas to flag on for medfilt round. Default is 10.0.')
        ap.add_argument('--sig_adj_med', default=4.0, type=float,
                        help='Number of sigmas to flag on for data adjacent to a flag for medfilt round. Default is 4.0.')
        ap.add_argument('--sig_init_mean', default=5.0, type=float,
                        help='Starting number of sigmas to flag on for meanfilt round. Default is 5.0.')
        ap.add_argument('--sig_adj_mean', default=2.0, type=float,
                        help='Number of sigmas to flag on for data adjacent to a flag for meanfilt round. Default is 2.0.')
        ap.add_argument('--ex_ants', default=None, type=str,
                        help='Comma-separated list of antennas to exclude. Flags of visibilities '
                        'formed with these antennas will be set to True.')
        ap.add_argument("--metrics_files", type=str, nargs='*', default=[],
                        help="path to file containing ant_metrics or auto_metrics readable by "
                             "hera_qm.metrics_io.load_metric_file. ex_ants here are combined "
                             "with antennas excluded via ex_ants. Flags of visibilities formed "
                             "with these antennas will be set to True.")
        ap.add_argument("--clobber", default=False, action="store_true",
                        help='overwrites existing files (default False)')
        ap.add_argument("--keep_edge_times", default=False, action="store_true",
                        help='keep metrics and flags within a convolution kernel width of the edges'
                        'of the time chunk being analyzed.')
        ap.add_argument("--Nwf_per_load", type=int, default=None,
                        help="Number of uvdata waterfalls to load simultaneously. Default, load all simultaneously.")
        ap.add_argument("--skip_omnical_median_filter", default=False, action="store_true", help="Do not perform omnical gains median filter.")
        ap.add_argument("--skip_omnical_mean_filter", default=False, action="store_true", help="Do not perform omnical gains mean filter.")
        ap.add_argument("--skip_omnivis_median_filter", default=False, action="store_true", help="Do not perform omnical visibility solutions median filter.")
        ap.add_argument("--skip_omnivis_mean_filter", default=False, action="store_true", help="Do not perform omnical visibility solutions mean filter.")
        ap.add_argument("--skip_omnical_chi2_median_filter", default=False, action="store_true", help="Do not perform omnical chi2 median filter.")
        ap.add_argument("--skip_omnical_chi2_mean_filter", default=False, action="store_true", help="Do not perform omnical chi2 mean filter.")
        ap.add_argument("--skip_omnical_zscore_filter", default=False, action="store_true", help="Do not perform global omnical regular or modified zscore filter.")
        ap.add_argument("--skip_abscal_median_filter", default=False, action="store_true", help="Do not perform abscal median filter.")
        ap.add_argument("--skip_abscal_mean_filter", default=False, action="store_true", help="Do not perform abscal mean filter.")
        ap.add_argument("--skip_abscal_chi2_median_filter", default=False, action="store_true", help="Do not perform abscal chi2 median filter.")
        ap.add_argument("--skip_abscal_chi2_mean_filter", default=False, action="store_true", help="Do not perform abscal chi2 mean filter.")
        ap.add_argument("--skip_abscal_zscore_filter", default=False, action="store_true", help="Do not perform global abscal  regular or modified zscore filter.")
        ap.add_argument("--skip_auto_median_filter", default=False, action="store_true", help="Do not perform autocorrelations median filter.")
        ap.add_argument("--skip_auto_mean_filter", default=False, action="store_true", help="Do not perform autocorrelations mean filter.")
        ap.add_argument("--use_cross_median_filter", default=False, action="store_true", help="Perform cross-correlations median filter (n.b. very expensive).")
        ap.add_argument("--skip_cross_mean_filter", default=False, action="store_true", help="Do not erform cross-correlations mean filter.")

    elif method_name == 'xrfi_run_data_only':
        ap.prog = 'xrfi_run_data_only.py'
        ap.add_argument('--data_files', default=None, type=str, help='Raw visibility '
                        'data files to flag on.', nargs='+')
        ap.add_argument('--a_priori_flag_yaml', default=None, type=str,
                        help=('Path to a priori flagging YAML with frequency, time, and/or '
                              'antenna flagsfor parsable by hera_qm.metrics_io.read_a_priori_*_flags()'))
        ap.add_argument('--skip_cross_pol_vis', default=False, action="store_true",
                        help="Don't load or use cross-polarized visibilities, e.g. 'en' or 'ne', in flagging.")
        ap.add_argument('--xrfi_path', default='', type=str,
                        help='Path to save flag files to. Default is same directory as input file.')
        ap.add_argument('--kt_size', default=8, type=int,
                        help='Size of kernel in time dimension for detrend in xrfi '
                        'algorithm. Default is 8.')
        ap.add_argument('--kf_size', default=8, type=int,
                        help='Size of kernel in frequency dimension for detrend in '
                        'xrfi algorithm. Default is 8.')
        ap.add_argument('--sig_init_med', default=10.0, type=float,
                        help='Starting number of sigmas to flag on for medfilt round. Default is 10.0.')
        ap.add_argument('--sig_adj_med', default=4.0, type=float,
                        help='Number of sigmas to flag on for data adjacent to a flag for medfilt round. Default is 4.0.')
        ap.add_argument('--sig_init_mean', default=5.0, type=float,
                        help='Starting number of sigmas to flag on for meanfilt round. Default is 5.0.')
        ap.add_argument('--sig_adj_mean', default=2.0, type=float,
                        help='Number of sigmas to flag on for data adjacent to a flag for meanfilt round. Default is 2.0.')
        ap.add_argument('--ex_ants', default=None, type=str,
                        help='Comma-separated list of antennas to exclude. Flags of visibilities '
                        'formed with these antennas will be set to True.')
        ap.add_argument("--metrics_files", type=str, nargs='*', default=[],
                        help="path to file containing ant_metrics or auto_metrics readable by "
                             "hera_qm.metrics_io.load_metric_file. ex_ants here are combined "
                             "with antennas excluded via ex_ants. Flags of visibilities formed "
                             "with these antennas will be set to True.")
        ap.add_argument("--clobber", default=False, action="store_true",
                        help='overwrites existing files (default False)')
        ap.add_argument("--cross_median_filter", default=False, action="store_true",
                        help="performs a median filter on cross-correlations. Adds significantly to runtime.")
        ap.add_argument("--skip_cross_mean_filter", default=False, action="store_true",
                        help="save i/o by skipping mean filter on cross correlations.")
        ap.add_argument("--keep_edge_times", default=False, action="store_true",
                        help='keep metrics and flags within a convolution kernel width of the edges'
                        'of the time chunk being analyzed.')
        ap.add_argument("--Nwf_per_load", type=int, default=None,
                        help="Number of uvdata waterfalls to load simultaneously. Default, load all simultaneously.")

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
        ap.add_argument('--a_priori_xants_yaml', type=str, default=None,
                        help=('path to a priori flagging YAML with xant information parsable by '
                              'hera_qm.metrics_io.read_a_priori_ant_flags()'))
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
        ap.add_argument('--a_priori_flag_yaml', default=None, type=str,
                        help=('Path to a priori flagging YAML with frequency, time, and/or '
                              'antenna flagsfor parsable by hera_qm.metrics_io.read_a_priori_*_flags()'))

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
        key2cat = {'final_metrics': 'ant_metrics'}
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


def apply_yaml_flags(uv, a_priori_flag_yaml, lat_lon_alt_degrees=None, telescope_name=None,
                     ant_indices_only=False, by_ant_pol=False, ant_pols=None,
                     flag_ants=True, flag_freqs=True, flag_times=True, throw_away_flagged_ants=False, unflag_first=False):
    """Apply frequency and time flags to a UVData or UVCal object

    This function takes in a uvdata or uvcal object and applies
    frequency, time, and antenna flags as appropriate.

    Parameters
    ----------
    uv : UVData or UVCal object or subclass thereof.
        uvdata or uvcal to apply frequency / time flags to.
    a_priori_flag_yaml : str
        path to yaml file with frequeny / time flags.
    lat_lon_alt_degrees : list, optional
        list of latitude, longitude, and altitude for telescope.
        latitude and longitude should be in degrees.
        altitude should be in meters.
        Default is None.
        If None, will determine location from uv.telescope_name
    telescope_name : str, optional
        string with name of telescope.
        Default is None. If None, will use uv.telescope_name
    ant_indices_only : bool
        If True, ignore polarizations and flag entire antennas when they appear, e.g. (1, 'Jee') --> 1.
    by_ant_pol : bool
        If True, expand all integer antenna indices into per-antpol entries using ant_pols
    ant_pols : list of str
        List of antenna polarizations strings e.g. 'Jee'. If not empty, strings in
        the YAML must be in here or an error is raised. Required if by_ant_pol is True.
    flag_ants : bool, optional
        specify whether or not to flag on antennas.
        default is True.
    flag_freqs : bool, optional
        specify whether or not to flag frequencies.
        default is True.
    flag_times : bool, optional
        specify whether or not to flag times
        default is True.
    throw_away_flagged_ants : bool, optional
        if True, remove flagged antennas from the data.
        default is False.
    unflag_first : bool, optional
        if True, remove existing flags in UVData/UVCal/UVFlag object
        (set all flag_array to False) before applying yaml flags.
        Default is False.
    Returns
    -------
        uv : UVData or UVCal object
            input uvdata / uvcal but now with flags applied
    """
    # check that uv is UVData or UVCal
    if not issubclass(uv.__class__, (UVData, UVCal, UVFlag)):
        raise NotImplementedError("uv must be a UVData, UVCal, or UVFlag object.")
    # only support single spw right now.
    if issubclass(uv.__class__, (UVData, UVCal)) and uv.Nspws > 1:
        raise NotImplementedError("apply_yaml_flags does not support multiple spws at this time.")
    # if UVCal provided and lst_array is None, get lst_array from times.
    # If lat_lon_alt is not specified, try to infer it from the telescope name, which calfits files generally carry around
    if unflag_first:
        uv.flag_array[:] = np.zeros_like(uv.flag_array, dtype=bool)
    if uv.lst_array is None:
        if lat_lon_alt_degrees is None:
            if telescope_name is None:
                telescope_name = uv.telescope_name
            if telescope_name.upper() in KNOWN_TELESCOPES:
                lat = KNOWN_TELESCOPES[telescope_name.upper()]['latitude'] * 180 / np.pi
                lon = KNOWN_TELESCOPES[telescope_name.upper()]['longitude'] * 180 / np.pi
                alt = KNOWN_TELESCOPES[telescope_name.upper()]['altitude']
                lat_lon_alt_degrees = np.asarray([lat, lon, alt])
            else:
                raise NotImplementedError(f'No known position for telescope {telescope_name}. lat_lon_alt_degrees must be specified.')

        # calculate LST grid in hours from time grid and lat_lon_alt
        lst_array = uvutils.get_lst_for_time(uv.time_array, *lat_lon_alt_degrees) * 12 / np.pi
    else:
        lst_array = np.unique(uv.lst_array) * 12  / np.pi

    time_array = np.unique(uv.time_array)
    # loop over spws to apply frequency flags.
    if flag_freqs:
        if len(uv.freq_array.shape) == 2:
            freqs = uv.freq_array[0]
        else:
            freqs = uv.freq_array
        flagged_channels = metrics_io.read_a_priori_chan_flags(a_priori_flag_yaml, freqs=freqs)
        if np.any(flagged_channels >= uv.Nfreqs):
            warnings.warn("Flagged channels were provided that exceed the maximum channel index. These flags are being dropped!")
        if np.any(flagged_channels < 0):
            warnings.warn("Flagged channels were provided with a negative channel index. These flags are being dropped!")
        flagged_channels = flagged_channels[(flagged_channels>=0) & (flagged_channels<=uv.Nfreqs)]
        if len(flagged_channels) > 0:
            if issubclass(uv.__class__, UVData) or (isinstance(uv, UVFlag) and uv.type == 'baseline'):
                uv.flag_array[:, 0, flagged_channels, :] = True
            elif issubclass(uv.__class__, UVCal) or (isinstance(uv, UVFlag) and uv.type == 'antenna'):
                uv.flag_array[:, 0, flagged_channels, :, :] = True
            elif isinstance(uv, UVFlag) and uv.type == 'waterfall':
                uv.flag_array[:, flagged_channels] = True
    if flag_times:
        # now do times.
        # get the integrations to flag
        flagged_integrations = metrics_io.read_a_priori_int_flags(a_priori_flag_yaml, lsts=lst_array, times=time_array)
        # ony select integrations less then Ntimes
        if np.any(flagged_integrations >= uv.Ntimes):
            warnings.warn("Flagged integrations were provided that exceed the maximum integration index. These flags are being dropped!")
        if np.any(flagged_integrations < 0):
            warnings.warn("Flagged integrations were provided with a negative integration index. These flags are being dropped!")
        flagged_integrations = flagged_integrations[(flagged_integrations>=0) & (flagged_integrations<=uv.Ntimes)]
        if len(flagged_integrations) > 0:
            flagged_times = time_array[flagged_integrations]
            for time in flagged_times:
                if issubclass(uv.__class__, UVData) or (isinstance(uv, UVFlag) and uv.type == 'baseline'):
                    uv.flag_array[uv.time_array == time, :, :, :] = True
                elif issubclass(uv.__class__, UVCal) or (isinstance(uv, UVFlag) and uv.type == 'antenna'):
                    uv.flag_array[:, :, :, uv.time_array == time, :] = True
            if isinstance(uv, UVFlag) and uv.type == 'waterfall':
                uv.flag_array[flagged_integrations, :] = True

    if flag_ants:
        # now do antennas.
        flagged_ants = metrics_io.read_a_priori_ant_flags(a_priori_flag_yaml, ant_indices_only=ant_indices_only,
                                                          by_ant_pol=by_ant_pol, ant_pols=ant_pols)
        npols = uv.flag_array.shape[-1]
        if issubclass(uv.__class__, UVData):
            pol_array = uv.polarization_array
        elif issubclass(uv.__class__, UVCal):
            pol_array = uv.jones_array
        elif issubclass(uv.__class__, UVFlag):
            pol_array = uv.polarization_array

        for ant in flagged_ants:
            if isinstance(ant, int):
                pol_selection = np.ones(npols, dtype=bool)
                antnum = ant
            elif isinstance(ant, (list, tuple, np.ndarray)):
                pol_num = uvutils.jstr2num(ant[1], x_orientation=uv.x_orientation)
                if pol_num in pol_array:
                    pol_selection = np.where(pol_array == pol_num)[0]
                else:
                    pol_selection = np.zeros(npols, dtype=bool)
                antnum = ant[0]
            if issubclass(uv.__class__, UVData) or (isinstance(uv, UVFlag) and uv.type == 'baseline'):
                blt_selection = np.logical_or(uv.ant_1_array == antnum, uv.ant_2_array == antnum)
                if np.any(blt_selection):
                    for bltind in np.where(blt_selection)[0]:
                        uv.flag_array[bltind, :, :, pol_selection] = True
            elif issubclass(uv.__class__, UVCal) or (isinstance(uv, UVFlag) and uv.type == 'antenna'):
                ant_selection = uv.ant_array == antnum
                if np.any(ant_selection):
                    for antind in np.where(ant_selection)[0]:
                        uv.flag_array[antind, :, :, :, pol_selection] =True
        if throw_away_flagged_ants:
            if ant_indices_only:
                if issubclass(uv.__class__, UVCal) or (isinstance(uv, UVFlag) and uv.type == 'antenna'):
                    antennas_to_keep = [a for a in uv.ant_array if a not in flagged_ants]
                    uv.select(antenna_nums=antennas_to_keep)
                elif issubclass(uv.__class__, UVData) or (isinstance(uv, UVFlag) and uv.type == 'baseline'):
                    antennas_to_keep = [a for a in np.unique(np.hstack([uv.ant_1_array, uv.ant_2_array])) if a not in flagged_ants]
                    uv.select(antenna_nums=antennas_to_keep)
            else:
                raise NotImplementedError("throwing away flagged antennas only implemented for ant_indices_only=True")
    # return uv with flags applied.
    return uv
