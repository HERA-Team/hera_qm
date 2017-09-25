from __future__ import print_function, division, absolute_import
import re
import os
import warnings
import argparse
import numpy as np


# argument-generating function for *_run wrapper functions
def get_metrics_ArgumentParser(method_name):
    """
    Function to get an ArgumentParser instance for working with metrics wrappers.

    Args:
        method_name -- target wrapper, must be "ant_metrics", "firstcal_metrics", "omnical_metrics", or "xrfi"
    Returns:
        a -- an argparse.ArgumentParser instance with the relevant options for the selected method
    """
    methods = ["ant_metrics", "firstcal_metrics", "xrfi", "omnical_metrics"]
    if method_name not in methods:
        raise AssertionError('method_name must be one of {}'.format(','.join(methods)))

    a = argparse.ArgumentParser()

    if method_name == 'ant_metrics':
        a.prog = 'ant_metrics.py'
        a.add_argument('-C', '--cal', type=str,
                       help='Calibration file to be used. Must be specified.')
        a.add_argument('-p', '--pol', default='', type=str,
                       help="Comma-separated list of polarizations included. Default is ''")
        a.add_argument('--crossCut', default=5.0, type=float,
                       help='Modified z-score cut for most cross-polarized antenna. Default 5 "sigmas"')
        a.add_argument('--deadCut', default=5.0, type=float,
                       help='Modified z-score cut for most likely dead antenna. Default 5 "sigmas"')
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
        a.add_argument('--fc_files', metavar='fc_files', type=str, nargs='*', default=[],
                       help='*.first.calfits files of firstcal solutions to perform omni-firstcal comparison metrics')
        a.add_argument('--no_bandcut', action='store_true', default=False,
                       help="flag to turn off cutting of frequency band edges before calculating metrics")
        a.add_argument('--phs_noise_cut', type=float, default=1.5,
                       help="set phase noise level cut. see OmniCal_Metrics.run_metrics() for details.")
        a.add_argument('--phs_std_cut', type=float, default=0.3,
                       help="set phase stand dev cut. see OmniCal_Metrics.run_metrics() for details.")
        a.add_argument('--chisq_std_cut', type=float, default=5.0,
                       help="set chisq stand dev cut. see OmniCal_Metrics.run_metrics() for details.")
        a.add_argument('--make_plots', action='store_true', default=False,
                       help="make .png plots of metrics")
        a.add_argument('--extension', default='.omni_metrics.json', type=str,
                       help='Extension to be appended to the metrics file name. Default is ".omni_metrics.json"')
        a.add_argument('--metrics_path', default='', type=str,
                       help='Path to save metrics file to. Default is same directory as file.')
        a.add_argument('files', metavar='files', type=str, nargs='*', default=[],
                       help='*.omni.calfits files for which to calculate omnical_metrics.')
    elif method_name == 'xrfi':
        a.prog = 'xrfi_run.py'
        a.add_argument('--infile_format', default='miriad', type=str,
                       help='File format for input files. Default is miriad.')
        a.add_argument('--outfile_format', default='miriad', type=str,
                       help='File format for output files. Default is miriad.')
        a.add_argument('--extension', default='R', type=str,
                       help='Extension to be appended to input file name. Default is "R".')
        a.add_argument('--summary', action='store_true', default=False,
                       help='Run summary of RFI flags and store in npz file.')
        a.add_argument('--summary_ext', default='.flag_summary.npz',
                       type=str, help='Extension to be appended to input file name'
                       ' for summary file. Default is ".flag_summary.npz"')
        a.add_argument('--xrfi_path', default='', type=str,
                       help='Path to save flagged file to. Default is same directory as input file.')
        a.add_argument('--algorithm', default='xrfi_simple', type=str,
                       help='RFI-flagging algorithm to use. Default is xrfi_simple.')
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
        a.add_argument('--broadcast', action='store_true', default=False,
                       help='Broadcast flags across data based on thresholds. Default is False.')
        a.add_argument('--bl_threshold', default=0., type=float,
                       help='Fraction of flags required to trigger a broadcast across'
                       ' baselines. Default is 0.')
        a.add_argument('--freq_threshold', default=0.9, type=float,
                       help='Fraction of channels required to trigger broadcast across'
                       ' frequency (single time). Default is 0.9.')
        a.add_argument('--time_threshold', default=0.9, type=float,
                       help='Fraction of times required to trigger broadcast across'
                       ' time (single frequency). Default is 0.9.')
        a.add_argument('files', metavar='files', type=str, nargs='*', default=[],
                       help='files for which to flag RFI.')
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
    fn = re.findall('zen\.\d{7}\.\d{5}\..*', fname)[0]
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
        for met in ['ant_z_scores', 'ant_avg', 'ant_std']:
            metric = '_'.join(['firstcal_metrics', met])
            d['ant_metrics'][metric] = []
            for ant, val in data[met].items():
                d['ant_metrics'][metric].append([ant, pol, val])
        metric = 'firstcal_metrics_bad_ants'
        d['ant_metrics'][metric] = []
        for ant in data['bad_ants']:
            d['ant_metrics'][metric].append([ant, pol, 1.])

    elif ftype == 'omnical':
        from hera_qm.omnical_metrics import load_omnical_metrics
        data = load_omnical_metrics(filename)
        pol = str(data['pol'])

        # pack array metrics
        cat = 'omnical_metrics_'
        for met in ['tot_chisq', 'tot_phs_noise', 'tot_phs_std', 'phs_noise_good_sol', 'phs_std_good_sol']:
          catmet = cat + met
          d['array_metrics'][catmet] = data[met]

        # pack antenna metrics
        cat = 'omnical_metrics_'
        for met in ['chisq_ant_avg', 'ant_phs_noise', 'ant_phs_std']:
          catmet = cat + met
          d['ant_metrics'][catmet] = [[a, pol, data[met][a]] for a in data[met]]

        for met in ['chisq_bad_ants', 'phs_noise_bad_ants', 'phs_std_bad_ants']:
          catmet = cat + met
          d['ant_metrics'][catmet] = [[a, pol, 1.] for a in data[met]]

    else:
        raise ValueError('Metric file type ' + ftype + ' is not recognized.')

    return d
