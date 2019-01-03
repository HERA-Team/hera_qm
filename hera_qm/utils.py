# -*- coding: utf-8 -*-
# Copyright (c) 2018 the HERA Project
# Licensed under the MIT License

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
    """
    Function to get an ArgumentParser instance for working with metrics wrappers.

    Args:
        method_name -- target wrapper, must be "ant_metrics", "firstcal_metrics",
                       "omnical_metrics", "xrfi_run", "delay_xrfi_run", or "xrfi_apply"
    Returns:
        a -- an argparse.ArgumentParser instance with the relevant options for the selected method
    """
    methods = ["ant_metrics", "firstcal_metrics", "omnical_metrics", "xrfi_run",
               "delay_xrfi_run", "xrfi_apply"]
    if method_name not in methods:
        raise AssertionError('method_name must be one of {}'.format(','.join(methods)))

    a = argparse.ArgumentParser()

    if method_name == 'ant_metrics':
        a.prog = 'ant_metrics.py'
        a.add_argument('-p', '--pol', default='', type=str,
                       help="Comma-separated list of polarizations included. Default is ''")
        a.add_argument('--crossCut', default=5.0, type=float,
                       help='Modified z-score cut for most cross-polarized antenna. Default 5 "sigmas"')
        a.add_argument('--deadCut', default=5.0, type=float,
                       help='Modified z-score cut for most likely dead antenna. Default 5 "sigmas"')
        a.add_argument('--alwaysDeadCut', default=10.0, type=float,
                       help='Modified z-score cut for antennas that are definitely dead. Default 10 "sigmas".'
                            'These are all thrown away at once without waiting to iteratively throw away only the worst offender.')
        a.add_argument('--extension', default='.ant_metrics.json', type=str,
                       help='Extension to be appended to the file name. Default is ".ant_metrics.json"')
        a.add_argument('--metrics_path', default='', type=str,
                       help='Path to save metrics file to. Default is same directory as file.')
        a.add_argument('--vis_format', default='miriad', type=str,
                       help='File format for visibility files. Default is miriad.')
        a.add_argument('-q', '--quiet', action='store_false', dest='verbose', default=True,
                       help='Silence feedback to the command line.')
        a.add_argument('files', metavar='files', type=str, nargs='*', default=[],
                       help='*.uv files for which to calculate ant_metrics.')

        a.add_argument('--run_mean_vij', action='store_true',
                       dest='run_mean_vij',
                       help=('Sets boolean flag to True. Flag determines if '
                             'mean_vij_metrics is run. Default: True'))
        a.add_argument('--skip_mean_vij', action='store_false',
                       dest='run_mean_vij',
                       help=('Sets boolean flag to False. Flag determines if '
                             'mean_vij_metrics is run. Default: True'))
        a.set_defaults(run_mean_vij=True)

        a.add_argument('--run_red_corr', action='store_true',
                       dest='run_red_corr',
                       help=('Sets boolean flag to True. Flag determines if '
                             'red_corr_metrics is run. Default: True'))
        a.add_argument('--skip_red_corr', action='store_false',
                       dest='run_red_corr',
                       help=('Sets boolean flag to False. Flag determines if '
                             'red_corr_metrics is run. Default: True'))
        a.set_defaults(run_red_corr=True)

        a.add_argument('--run_cross_pols', action='store_true',
                       dest='run_cross_pols',
                       help=('Sets boolean flag to True. Flag determines if '
                             'mean_Vij_cross_pol_metrics and '
                             'red_corr_cross_pol_metrics are run. '
                             'Default: True'))
        a.add_argument('--skip_cross_pols', action='store_false',
                       dest='run_cross_pols',
                       help=('Sets boolean flag to False. Flag determines if '
                             'mean_Vij_cross_pol_metrics and '
                             'red_corr_cross_pol_metrics are run. '
                             'Default: True'))
        a.set_defaults(run_cross_pols=True)

    elif method_name == 'firstcal_metrics':
        a.prog = 'firstcal_metrics.py'
        a.add_argument('--std_cut', default=0.5, type=float,
                       help='Delay standard deviation cut for good / bad determination. Default 0.5')
        a.add_argument('--extension', default='.firstcal_metrics.json', type=str,
                       help='Extension to be appended to the file name. Default is ".firstcal_metrics.json"')
        a.add_argument('--metrics_path', default='', type=str,
                       help='Path to save metrics file to. Default is same directory as file.')
        a.add_argument('files', metavar='files', type=str, nargs='*', default=[],
                       help='*.calfits files for which to calculate firstcal_metrics.')
    elif method_name == 'omnical_metrics':
        a.prog = 'omnical_metrics.py'
        a.add_argument('--fc_files', metavar='fc_files', type=str, default=None,
                       help='[optional] *.first.calfits files of firstcal solutions to perform omni-firstcal comparison metrics.'
                       ' If multiple pols exist in a single *.omni.calfits file, feed .pol.first.calfits fcfiles as comma-delimited.'
                       ' If operating on multiple .omni.calfits files, feed separate comma-delimited fcfiles as vertical bar-delimited.'
                       ' Ex1: omnical_metrics_run.py --fc_files=zen.xx.first.calfits,zen.yy.first.calfits zen.omni.calfits'
                       ' Ex2: omnical_metrics_run.py --fc_files=zen1.xx.first.calfits,zen1.yy.first.calfits|zen2.xx.first.calfits,zen2.yy.first.calfits zen1.omni.calfits zen2.omni.calfits')
        a.add_argument('--no_bandcut', action='store_true', default=False,
                       help="flag to turn off cutting of frequency band edges before calculating metrics")
        a.add_argument('--phs_std_cut', type=float, default=0.3,
                       help="set gain phase stand dev cut. see OmniCal_Metrics.run_metrics() for details.")
        a.add_argument('--chisq_std_zscore_cut', type=float, default=4.0,
                       help="set chisq stand dev. zscore cut. see OmniCal_Metrics.run_metrics() for details.")
        a.add_argument('--make_plots', action='store_true', default=False,
                       help="make .png plots of metrics")
        a.add_argument('--extension', default='.omni_metrics.json', type=str,
                       help='Extension to be appended to the metrics file name. Default is ".omni_metrics.json"')
        a.add_argument('--metrics_path', default='', type=str,
                       help='Path to save metrics file to. Default is same directory as file.')
        a.add_argument('files', metavar='files', type=str, nargs='*', default=[],
                       help='*.omni.calfits files for which to calculate omnical_metrics.')
    elif method_name == 'xrfi_run':
        a.prog = 'xrfi_run.py'
        a.add_argument('--infile_format', default='miriad', type=str,
                       help='File format for input files. Not currently used while '
                       'we use generic read function in pyuvdata, But will '
                       'be implemented for partial io.')
        a.add_argument('--extension', default='flags.h5', type=str,
                       help='Extension to be appended to input file name. Default is "flags.h5".')
        a.add_argument('--summary', action='store_true', default=False,
                       help='Run summary of RFI flags and store in npz file.')
        a.add_argument('--summary_ext', default='flag_summary.h5',
                       type=str, help='Extension to be appended to input file name'
                       ' for summary file. Default is "flag_summary.h5"')
        a.add_argument('--xrfi_path', default='', type=str,
                       help='Path to save flag files to. Default is same directory as input file.')
        a.add_argument('--algorithm', default='xrfi_simple', type=str,
                       help='RFI-flagging algorithm to use. Default is xrfi_simple.')
        a.add_argument('--model_file', default=None, type=str, help='Model visibility '
                       'file to flag on.')
        a.add_argument('--model_file_format', default='uvfits', type=str,
                       help='File format for input files. Not currently used while '
                       'we use generic read function in pyuvdata, But will '
                       'be implemented for partial io.')
        a.add_argument('--calfits_file', default=None, type=str, help='Calfits file '
                       'to use to flag on gains and/or chisquared values.')
        a.add_argument('--nsig_df', default=6.0, type=float, help='Number of sigma '
                       'above median value to flag in f direction for xrfi_simple. Default is 6.')
        a.add_argument('--nsig_dt', default=6.0, type=float,
                       help='Number of sigma above median value to flag in t direction'
                       ' for xrfi_simple. Default is 6.')
        a.add_argument('--nsig_all', default=0.0, type=float,
                       help='Number of overall sigma above median value to flag'
                       ' for xrfi_simple. Default is 0 (skip).')
        a.add_argument('--kt_size', default=8, type=int,
                       help='Size of kernel in time dimension for detrend in xrfi '
                       'algorithm. Default is 8.')
        a.add_argument('--kf_size', default=8, type=int,
                       help='Size of kernel in frequency dimension for detrend in '
                       'xrfi algorithm. Default is 8.')
        a.add_argument('--sig_init', default=6.0, type=float,
                       help='Starting number of sigmas to flag on. Default is 6.')
        a.add_argument('--sig_adj', default=2.0, type=float,
                       help='Number of sigmas to flag on for data adjacent to a flag. Default is 2.')
        a.add_argument('--px_threshold', default=0.2, type=float,
                       help='Fraction of flags required to trigger a broadcast across'
                       ' baselines for a given (time, frequency) pixel. Default is 0.2.')
        a.add_argument('--freq_threshold', default=0.5, type=float,
                       help='Fraction of channels required to trigger broadcast across'
                       ' frequency (single time). Default is 0.5.')
        a.add_argument('--time_threshold', default=0.05, type=float,
                       help='Fraction of times required to trigger broadcast across'
                       ' time (single frequency). Default is 0.05.')
        a.add_argument('--ex_ants', default=None, type=str,
                       help='Comma-separated list of antennas to exclude. Flags of visibilities '
                       'formed with these antennas will be set to True.')
        a.add_argument('--metrics_file', default=None, type=str,
                       help='Metrics file that contains a list of excluded antennas. Flags of '
                       'visibilities formed with these antennas will be set to True.')
        a.add_argument('filename', metavar='filename', nargs='*', type=str, default=[],
                       help='file for which to flag RFI (only one file allowed).')
    elif method_name == 'delay_xrfi_run':
        a.prog = 'delay_xrfi_run.py'
        a.add_argument('filename', metavar='filename', nargs='?', type=str, default=None,
                       help='file for which to flag RFI (only one file allowed).')
        x = a.add_argument_group(title='XRFI options', description='Options related to '
                                 'the RFI flagging routine.')
        x.add_argument('--infile_format', default='miriad', type=str,
                       help='File format for input files. DEPRECATED, but kept for legacy.')
        x.add_argument('--extension', default='flags.h5', type=str,
                       help='Extension to be appended to input file name. Default is "flags.h5".')
        x.add_argument('--summary', action='store_true', default=False,
                       help='Run summary of RFI flags and store in npz file.')
        x.add_argument('--summary_ext', default='flag_summary.h5',
                       type=str, help='Extension to be appended to input file name'
                       ' for summary file. Default is "flag_summary.h5"')
        x.add_argument('--xrfi_path', default='', type=str,
                       help='Path to save flag files to. Default is same directory as input file.')
        x.add_argument('--algorithm', default='xrfi_simple', type=str,
                       help='RFI-flagging algorithm to use. Default is xrfi_simple.')
        x.add_argument('--model_file', default=None, type=str, help='Model visibility '
                       'file to flag on.')
        x.add_argument('--model_file_format', default='uvfits', type=str,
                       help='File format for input files. DEPRECATED, but kept for legacy.')
        x.add_argument('--calfits_file', default=None, type=str, help='Calfits file '
                       'to use to flag on gains and/or chisquared values.')
        x.add_argument('--nsig_df', default=6.0, type=float, help='Number of sigma '
                       'above median value to flag in f direction for xrfi_simple. Default is 6.')
        x.add_argument('--nsig_dt', default=6.0, type=float,
                       help='Number of sigma above median value to flag in t direction'
                       ' for xrfi_simple. Default is 6.')
        x.add_argument('--nsig_all', default=0.0, type=float,
                       help='Number of overall sigma above median value to flag'
                       ' for xrfi_simple. Default is 0 (skip).')
        x.add_argument('--kt_size', default=8, type=int,
                       help='Size of kernel in time dimension for detrend in xrfi '
                       'algorithm. Default is 8.')
        x.add_argument('--kf_size', default=8, type=int,
                       help='Size of kernel in frequency dimension for detrend in '
                       'xrfi algorithm. Default is 8.')
        x.add_argument('--sig_init', default=6.0, type=float,
                       help='Starting number of sigmas to flag on. Default is 6.')
        x.add_argument('--sig_adj', default=2.0, type=float,
                       help='Number of sigmas to flag on for data adjacent to a flag. Default is 2.')
        x.add_argument('--px_threshold', default=0.2, type=float,
                       help='Fraction of flags required to trigger a broadcast across'
                       ' baselines for a given (time, frequency) pixel. Default is 0.2.')
        x.add_argument('--freq_threshold', default=0.5, type=float,
                       help='Fraction of channels required to trigger broadcast across'
                       ' frequency (single time). Default is 0.5.')
        x.add_argument('--time_threshold', default=0.05, type=float,
                       help='Fraction of times required to trigger broadcast across'
                       ' time (single frequency). Default is 0.05.')
        x.add_argument('--ex_ants', default='', type=str,
                       help='Comma-separated list of antennas to exclude. Flags of visibilities '
                       'formed with these antennas will be set to True.')
        x.add_argument('--metrics_file', default='', type=str,
                       help='Metrics file that contains a list of excluded antennas. Flags of '
                       'visibilities formed with these antennas will be set to True.')
        d = a.add_argument_group(title='Delay filter options', description='Options '
                                 'related to the delay filter which is applied before flagging.')
        d.add_argument("--standoff", type=float, default=15.0, help='fixed additional delay beyond the horizon (default 15 ns)')
        d.add_argument("--horizon", type=float, default=1.0, help='proportionality constant for bl_len where 1.0 (default) is the horizon\
                                  (full light travel time)')
        d.add_argument("--tol", type=float, default=1e-7, help='CLEAN algorithm convergence tolerance (default 1e-7). '
                       'NOTE: default is different from default when running delay_filter_run.py.')
        d.add_argument("--window", type=str, default="tukey", help='window function for frequency filtering (default "tukey",\
                                  see aipy.dsp.gen_window for options')
        d.add_argument("--skip_wgt", type=float, default=0.1, help='skips filtering rows with unflagged fraction ~< skip_wgt (default 0.1)')
        d.add_argument("--maxiter", type=int, default=100, help='maximum iterations for aipy.deconv.clean to converge (default 100)')
        d.add_argument("--alpha", type=float, default=.5, help='alpha parameter to use for Tukey window (ignored if window is not Tukey)')
        d.add_argument('--waterfalls', default=None, type=str, help='comma separated '
                       'list of npz files containing waterfalls of flags to broadcast '
                       'to full flag array and apply before delay filter.')
    elif method_name == 'xrfi_apply':
        a.prog = 'xrfi_apply.py'
        a.add_argument('--infile_format', default='miriad', type=str,
                       help='File format for input files. Not currently used while '
                       'we use generic read function in pyuvdata, But will '
                       'be implemented for partial io.')
        a.add_argument('--xrfi_path', default='', type=str,
                       help='Path to save output to. Default is same directory as input file.')
        a.add_argument('--outfile_format', default='miriad', type=str,
                       help='File format for output files. Default is miriad.')
        a.add_argument('--extension', default='R', type=str,
                       help='Extension to be appended to input file name. Default is "R".')
        a.add_argument('--overwrite', action='store_true', default=False,
                       help='Option to overwrite output file if it already exists.')
        a.add_argument('--flag_file', default=None, type=str, help='npz file '
                       'containing full flag array to insert into data file.')
        a.add_argument('--waterfalls', default=None, type=str, help='comma separated '
                       'list of npz files containing waterfalls of flags to broadcast '
                       'to full flag array and union with flag array in flag_file.')
        a.add_argument('--output_uvflag', default=True, type=bool,
                       help='Whether to save a uvflag object with the final flag array. '
                       'The flag array will be identical to what is stored in the data.')
        a.add_argument('--output_uvflag_ext', default='flags.h5', type=str,
                       help='Extension to be appended to input file name. Default is "flags.h5".')
        a.add_argument('filename', metavar='filename', nargs='*', type=str, default=[],
                       help='file for which to flag RFI (only one file allowed).')
    return a


def get_pol(fname):
    """Strips the filename of a HERA visibility to its polarization
    Args:
        fname -- name of file (string). Note that just the file name should be
                 passed in, to avoid pathological problems like directories that
                 may match the structure being searched for.
    Returns:
        polarization -- polarization label e.g. "xx" (string)
    """
    # XXX: assumes file naming format:
    #    zen.ddddddd.ddddd.pp.*
    # the 'd' values are the 7-digit Julian day and 5-digit fractional Julian
    # date. The 'pp' is the polarization extracted. It need not be 2 characters,
    # and the parser will return everying between the two periods.
    fn = re.findall(r'zen\.\d{7}\.\d{5}\..*', fname)[0]
    return fn.split('.')[3]


def generate_fullpol_file_list(files, pol_list):
    """Generate a list of unique JDs that have all four polarizations available
    Args:
       files -- list of files to look for
       pol_list -- list of polarizations to look for, as strings (e.g.,
                   ['xx', 'xy', 'yx', 'yy'])
    Returns:
       jd_list -- list of lists of JDs where all supplied polarizations could be found

    This function, when given a list of files, will look for the specified polarizations,
    and add the JD to the returned list if all polarizations were found. The return is a
    list of lists, where the outer list is a single JD and the inner list is a "full set"
    of polarizations, based on the polarization list provided.
    """
    # initialize
    file_list = []

    # Check if all input files are full-pol files
    # if so return the input files as the full list

    # First create the check variable
    pol_check = np.copy(pol_list)

    # use reduce to compare the input pol list with the pols of the file
    # save into pol_check
    # this reduce with intersection will compute the mininum matching
    # polarizations for all input files
    uvd = UVData()

    for filename in files:
        if filename.split('.')[-1] == 'uvh5':
            uvd.read_uvh5(filename, read_data=False)
        else:
            uvd.read(filename)

        input_pols = uvutils.polnum2str(uvd.polarization_array)
        # Using reduce here saves us a for-loop
        # though it can easily be re-written as one for clarity
        pol_check = reduce(np.intersect1d, [input_pols, pol_list, pol_check])

    del uvd

    if np.array_equal(np.sort(pol_list), np.sort(pol_check)):
        return files
    elif pol_check.size > 1:
        raise ValueError("At least one input file only contains pols: {pol}."
                         "Must provide a list of full polarization files or "
                         "files with single polarizations in the name of the "
                         "file (e.g. zen.JD.pol.HH.uv)")

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
    """ Function to combine metrics lists from hera_qm modules.

    Returns:
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
    """ Read in file containing quality metrics and stuff into a dictionary which
    can be used by M&C to populate db.
    If one wishes to add a metric to the list that is tracked by M&C, it is
    (unfortunately) currently a four step process:
    1) Ensure your metric is written to the output files of the relevant module.
    2) Add the metric and a description to the get_X_metrics_dict() function in
       said module.
    3) Check that the metric is appropriately ingested in this function, and make
       changes as necessary.
    4) Add unit tests! Also check that the hera_mc tests still pass.

    Args:
        filename: (str) file to read and convert
        ftype: (str) Type of metrics file. Options are ['ant', 'firstcal', 'omnical']
    Returns:
        d: (dict) Dictionary containing keys and data to pass to M&C.
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


def lst_from_uv(uv):
    ''' Calculate the lst_array for a UVData or UVCal object.
    Args:
        uv: a UVData or UVCal object.
    Returns:
        lst_array: lst_array corresponding to time_array and at telecope location.
                   Units are radian.
    '''
    if not isinstance(uv, (UVCal, UVData)):
        raise ValueError('Function lst_from_uv can only operate on '
                         'UVCal or UVData object.')

    tel = uvtel.get_telescope(uv.telescope_name)
    lat, lon, alt = tel.telescope_location_lat_lon_alt_degrees
    lst_array = uvutils.get_lst_for_time(uv.time_array, lat, lon, alt)
    return lst_array


def mean(a, weights=None, axis=None, returned=False):
    ''' Function to average data. This is similar to np.average, except it
    handles infs (by giving them zero weight) and zero weight axes (by forcing
    result to be inf with zero output weight).
    Args:
        a - array to process
        weights - weights for average. If none, will default to equal weight for
                  all non-infinite data.
        axis - axis keyword to pass to np.sum
        returned - whether to return sum of weights. Default is False.
    '''
    a = copy.deepcopy(a)  # avoid changing outside
    if weights is None:
        weights = np.ones_like(a)
    w = weights * np.logical_not(np.isinf(a))
    a[np.isinf(a)] = 0
    wo = np.sum(w, axis=axis)
    o = np.sum(w * a, axis=axis)
    where = (wo > 1e-10)
    o = np.true_divide(o, wo, where=where)
    o = np.where(where, o, np.inf)
    if returned:
        return o, wo
    else:
        return o


def absmean(a, weights=None, axis=None, returned=False):
    ''' Function to average absolute value
    Args:
        a - array to process
        weights - weights for average
        axis - axis keyword to pass to np.mean
        returned - whether to return sum of weights. Default is False.
    '''
    return mean(np.abs(a), weights=weights, axis=axis, returned=returned)


def quadmean(a, weights=None, axis=None, returned=False):
    ''' Function to average in quadrature
    Args:
        a - array to process
        weights - weights for average
        axis - axis keyword to pass to np.mean
        returned - whether to return sum of weights. Default is False.
    '''
    o = mean(np.abs(a)**2, weights=weights, axis=axis, returned=returned)
    if returned:
        return np.sqrt(o[0]), o[1]
    else:
        return np.sqrt(o)


def or_collapse(a, weights=None, axis=None, returned=False):
    ''' Function to collapse axes using OR operation
    Args:
        a - boolean array to process
        weights - NOT USED, but kept for symmetry with other averaging functions
        axis - axis or axes over which to OR
        returned - whether to return dummy weights array. NOTE: the dummy weights
                   will simply be an array of ones. Default is False.
    '''
    if a.dtype != np.bool:
        raise ValueError('Input to or_collapse function must be boolean array')
    o = np.any(a, axis=axis)
    if (weights is not None) and not np.all(weights == weights.reshape(-1)[0]):
        warnings.warn('Currently weights are not handled when OR-ing boolean arrays.')
    if returned:
        return o, np.ones_like(o, dtype=np.float)
    else:
        return o


# Dictionary to map different methods for averaging data.
averaging_dict = {'mean': mean, 'absmean': absmean, 'quadmean': quadmean,
                  'or': or_collapse}


def flags2waterfall(uv, flag_array=None, keep_pol=False):
    """
    Convert a flag array to a 2D waterfall of dimensions (Ntimes, Nfreqs).
    Averages over baselines and polarizations (in the case of visibility data),
    or antennas and jones parameters (in case of calibrationd data).
    Args:
        uv -- A UVData or UVCal object which defines the times and frequencies,
              and supplies the flag_array to convert (if flag_array not specified)
        flag_array -- Optional flag array to convert instead of uv.flag_array.
                      Must have same dimensions as uv.flag_array.
        keep_pol -- Option to keep the polarization axis in tact. Default is False.
    Returns:
        waterfall -- 2D waterfall of averaged flags, for example fraction of baselines
                     which are flagged for every time and frequency (in case of UVData input)
                     Size is (Ntimes, Nfreqs) or (Ntimes, Nfreqs, Npols).
    """
    if not isinstance(uv, (UVData, UVCal)):
        raise ValueError('flags2waterfall() requires a UVData or UVCal object as '
                         'the first argument.')
    if flag_array is None:
        flag_array = uv.flag_array
    if uv.flag_array.shape != flag_array.shape:
        raise ValueError('Flag array must align with UVData or UVCal object.')

    if isinstance(uv, UVCal):
        if keep_pol:
            waterfall = np.swapaxes(np.mean(flag_array, axis=(0, 1)), 0, 1)
        else:
            waterfall = np.mean(flag_array, axis=(0, 1, 4)).T
    else:
        if keep_pol:
            waterfall = np.zeros((uv.Ntimes, uv.Nfreqs, uv.Npols))
            for i, t in enumerate(np.unique(uv.time_array)):
                waterfall[i, :] = np.mean(flag_array[uv.time_array == t, 0, :, :],
                                          axis=0)
        else:
            waterfall = np.zeros((uv.Ntimes, uv.Nfreqs))
            for i, t in enumerate(np.unique(uv.time_array)):
                waterfall[i, :] = np.mean(flag_array[uv.time_array == t, 0, :, :],
                                          axis=(0, 2))

    return waterfall


def dynamic_slice(arr, slice_obj, axis=-1):
    """
    Dynamic slicing of arr along axis given slice object.

    Parameters
    ----------
    arr : ndarray

    slice_obj : slice object

    axis : integer

    Returns
    -------
    arr_slice : ndarray
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
    """
    Function to strip the extension off a path. Note this calls os.path.splitext,
    but we change the output slightly for convenience in our filename building.

    Parameters
    ----------
    path : str
        Path you wish to strip of its extension
    return_ext : bool
        If True, return the extension as well. Default is False.

    Returns
    -------
    root : str
        The input path without its extension.
    ext : str, optional
        The extension of the input path (without the leading ".")
    """
    if return_ext:
        root, ext = os.path.splitext(path)
        return (root, ext[1:])
    else:
        return os.path.splitext(path)[0]
