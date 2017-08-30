from __future__ import print_function, division, absolute_import
import re
import os
import warnings
import optparse
import numpy as np


# option-generating function for *_run wrapper functions
def get_metrics_OptionParser(method_name):
    """
    Function to get an OptionParser instance for working with metrics wrappers.

    Args:
        method_name -- target wrapper, must be "ant_metrics", "firstcal_metrics", or "xrfi"
    Returns:
        o -- an optparse.OptionParser instance with the relevant options for the selected method
    """
    methods = ["ant_metrics", "firstcal_metrics", "xrfi"]
    if method_name not in methods:
        raise AssertionError('method_name must be one of {}'.format(','.join(methods)))

    o = optparse.OptionParser()

    if method_name == 'ant_metrics':
        o.set_usage("ant_metrics_run.py -C [calfile] [options] *.uv")
        o.add_option('-C', '--cal', dest='cal', type='string',
                     help='Calibration file to be used. Must be specified.')
        o.add_option('-p', '--pol', dest='pol', default='', type='string',
                     help="Comma-separated list of polarizations included. Default is ''")
        o.add_option('--crossCut', dest='crossCut', default=5, type='float',
                     help='Modified z-score cut for most cross-polarized antenna. Default 5 "sigmas"')
        o.add_option('--deadCut', dest='deadCut', default=5, type='float',
                     help='Modified z-score cut for most likely dead antenna. Default 5 "sigmas"')
        o.add_option('--extension', dest='extension', default='.ant_metrics.json', type='string',
                     help='Extension to be appended to the file name. Default is ".ant_metrics.json"')
        o.add_option('--metrics_path', dest='metrics_path', default='', type='string',
                     help='Path to save metrics file to. Default is same directory as file.')
        o.add_option('--vis_format', dest='vis_format', default='miriad', type='string',
                     help='File format for visibility files. Default is miriad.')
        o.add_option('-q', '--quiet', action='store_false', dest='verbose', default=True,
                     help='Silence feedback to the command line.')
    elif method_name == 'firstcal_metrics':
        o.set_usage("firstcal_metrics_run.py [options] *.calfits")
        o.add_option('--std_cut', dest='std_cut', default=0.5, type='float',
                     help='Delay standard deviation cut for good / bad determination. Default 0.5')
        o.add_option('--extension', dest='extension', default='.firstcal_metrics.json', type='string',
                     help='Extension to be appended to the file name. Default is ".firstcal_metrics.json"')
        o.add_option('--metrics_path', dest='metrics_path', default='', type='string',
                     help='Path to save metrics file to. Default is same directory as file.')
    elif method_name == 'xrfi':
        o.set_usage("xrfi_run.py [options] *.uv")
        o.add_option('--infile_format', dest='infile_format', default='miriad', type='string',
                     help='File format for input files. Default is miriad.')
        o.add_option('--outfile_format', dest='outfile_format', default='miriad', type='string',
                     help='File format for output files. Default is miriad.')
        o.add_option('--extension', dest='extension', default='R', type='string',
                     help='Extension to be appended to input file name. Default is "R".')
        o.add_option('--summary', action='store_true', dest='summary', default=False,
                     help='Run summary of RFI flags and store in npz file.')
        o.add_option('--summary_ext', dest='summary_ext', default='.flag_summary.npz',
                     type='string', help='Extension to be appended to input file name'
                     ' for summary file. Default is ".flag_summary.npz"')
        o.add_option('--xrfi_path', dest='xrfi_path', default='', type='string',
                     help='Path to save flagged file to. Default is same directory as input file.')
        o.add_option('--algorithm', dest='algorithm', default='xrfi_simple', type='string',
                     help='RFI-flagging algorithm to use. Default is xrfi_simple.')
        o.add_option('--nsig_df', dest='nsig_df', default=6, type='float',
                     help='Number of sigma above median value to flag in f direction'
                     ' for xrfi_simple. Default is 6.')
        o.add_option('--nsig_dt', dest='nsig_dt', default=6, type='float',
                     help='Number of sigma above median value to flag in t direction'
                     ' for xrfi_simple. Default is 6.')
        o.add_option('--nsig_all', dest='nsig_all', default=0, type='float',
                     help='Number of overall sigma above median value to flag'
                     ' for xrfi_simple. Default is 0 (skip).')
        o.add_option('--kt_size', dest='kt_size', default=8, type='int',
                     help='Size of kernel in time dimension for detrend in xrfi '
                     'algorithm. Default is 8.')
        o.add_option('--kf_size', dest='kf_size', default=8, type='int',
                     help='Size of kernel in frequency dimension for detrend in '
                     'xrfi algorithm. Default is 8.')
        o.add_option('--sig_init', dest='sig_init', default=6, type='float',
                     help='Starting number of sigmas to flag on. Default is 6.')
        o.add_option('--sig_adj', dest='sig_adj', default=2, type='float',
                     help='Number of sigmas to flag on for data adjacent to a flag. Default is 2.')
    return o


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
    if ftype is 'ant':
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

    elif ftype is 'firstcal':
        from hera_qm.firstcal_metrics import load_firstcal_metrics
        data = load_firstcal_metrics(filename)
        pol = data['pol']
        d['array_metrics']['firstcal_metrics_good_sol'] = data['good_sol']
        d['array_metrics']['firstcal_metrics_agg_std'] = data['agg_std']
        for met in ['ant_z_scores', 'ant_avg', 'ant_std']:
            metric = '_'.join(['firstcal_metrics', met])
            d['ant_metrics'][metric] = []
            for ant, val in data[met].items():
                d['ant_metrics'][metric].append([ant, pol, val])
        metric = 'firstcal_metrics_bad_ants'
        d['ant_metrics'][metric] = []
        for ant in data['bad_ants']:
            d['ant_metrics'][metric].append([ant, pol, 1.])

    elif ftype is 'omnical':
        from pyuvdata import UVCal
        uvcal = UVCal()
        uvcal.read_calfits(filename)
        pol_dict = {-5: 'x', -6: 'y'}
        d['ant_metrics']['omnical_quality'] = []
        for pi, pol in enumerate(uvcal.jones_array):
            try:
                pol = pol_dict[pol]
            except KeyError:
                raise ValueError('Invalid polarization for ant_metrics in M&C.')
            for ai, ant in enumerate(uvcal.ant_array):
                val = np.median(uvcal.quality_array[ai, 0, :, :, pi], axis=0)
                val = np.mean(val)
                d['ant_metrics']['omnical_quality'].append([ant, pol, val])
        if uvcal.total_quality_array is not None:
            d['array_metrics']['omnical_total_quality'] = np.mean(uvcal.total_quality_array)

    else:
        raise ValueError('Metric file type ' + ftype + ' is not recognized.')

    return d
