from __future__ import print_function, division, absolute_import
import re
import os
import warnings
import optparse
import aipy


# option-generating function for *_run wrapper functions
def get_metrics_OptionParser(method_name):
    """
    Function to get an OptionParser instance for working with metrics wrappers.

    Args:
        method_name -- target wrapper, must be "ant_metrics" or "xrfi"
    Returns:
        o -- an optparse.OptionParser instance with the relevant options for the selected method
    """
    methods = ["ant_metrics", "xrfi"]
    if method_name not in methods:
        raise AssertionError('method_name must be one of {}'.format(','.join(methods)))

    o = optparse.OptionParser()

    if method_name == 'ant_metrics':
        o.set_usage("ant_metrics_run.py -C [calfile] [options] *.uv")
        aipy.scripting.add_standard_options(o, cal=True)
        o.add_option('-p', '--pol', dest='pol', default='', type='string',
                     help="Comma-separated list of polarizations included. Default is ''")
        o.add_option('--crossCut', dest='crossCut', default=5, type='float',
                     help='Modified z-score cut for most cross-polarized antenna. Default 5 "sigmas"')
        o.add_option('--deadCut', dest='deadCut', default=5, type='float',
                     help='Modified z-score cut for most likely dead antenna. Default 5 "sigmas"')
        o.add_option('--extension', dest='extension', default='.ant_metrics.json', type='string',
                     help='Extension to be appended to the file name. Default is ".ant_metrics.json"')
        o.add_option('--metrics_path', dest='metrics_path', default='', type='string',
                     help='Path to save metrics file to. Default is same directory as file')
        o.add_option('--vis_format', dest='vis_format', default='miriad', type='string',
                     help='File format for visibility files. Default is miriad.')
        o.add_option('-q', '--quiet', action='store_false', dest='verbose', default=True,
                     help='Silence feedback to the command line.')
    elif method_name == 'xrfi':
        o.set_usage("xrfi_run.py")

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

