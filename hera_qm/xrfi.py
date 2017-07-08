import numpy as np
from scipy.signal import medfilt

#############################################################################
# Functions useful for understanding the statistics in RFI selection/flagging
#############################################################################

def medmin(d):
    '''Calculate the median minus minimum statistic of array.
    Args:
        d (array): 2D data array of the shape (time,frequency).
    Returns:
        (array): array with the statistic applied.
    '''
    mn = np.min(d, axis=0)
    return 2 * np.median(mn) - np.min(mn)


def medminfilt(d, Kt=8, Kf=8, complex=False):
    '''Filter an array on scales of Kt,Kf indexes with medmin.
    Args:
        d (array): 2D data array of the shape (time,frequency).
        Kt (int, optional): integer representing box dimension in time to apply statistic.
        Kf (int, optional): integer representing box dimension in frequency to apply statistic.
    Returns:
        array: filtered array. Same shape as input array.
    '''
    d_sm = np.empty_like(d)
    if complex:
        for i in xrange(d.shape[0]):
            for j in xrange(d.shape[1]):
                i0, j0 = max(0, i - Kt), max(0, j - Kf)
                i1, j1 = min(d.shape[0], i + Kt), min(d.shape[1], j + Kf)
                d_sm[i, j] = medmin(d[i0:i1, j0:j1].real) + 1j*medmin(d[i0:i1, j0:j1].imag)
        return d_sm
    else:
        for i in xrange(d.shape[0]):
            for j in xrange(d.shape[1]):
                i0, j0 = max(0, i - Kt), max(0, j - Kf)
                i1, j1 = min(d.shape[0], i + Kt), min(d.shape[1], j + Kf)
                d_sm[i, j] = medmin(d[i0:i1, j0:j1])
        return d_sm

def detrend_deriv(d, dt=True, df=True):
    '''XXX This only works ok on sparse RFI.'''
    if df:
        d_df = np.empty_like(d)
        d_df[:, 1:-1] = (d[:, 1:-1] - .5 *
                         (d[:, :-2] + d[:, 2:])) / np.sqrt(1.5)
        d_df[:, 0] = (d[:, 0] - d[:, 1]) / np.sqrt(2)
        d_df[:, -1] = (d[:, -1] - d[:, -2]) / np.sqrt(2)
    else:
        d_df = d
    if dt:
        d_dt = np.empty_like(d_df)
        d_dt[1:-1] = (d_df[1:-1] - .5 * (d_df[:-2] + d_df[2:])) / np.sqrt(1.5)
        d_dt[0] = (d_df[0] - d_df[1]) / np.sqrt(2)
        d_dt[-1] = (d_df[-1] - d_df[-2]) / np.sqrt(2)
    else:
        d_d = d_df
    d2 = np.abs(d_dt)**2
    # model sig as separable function of 2 axes                                                                                                                 
    sig_f = np.median(d2, axis=0)
    sig_f.shape = (1, -1)
    sig_t = np.median(d2, axis=1)
    sig_t.shape = (-1, 1)
    sig = np.sqrt(sig_f * sig_t / np.median(sig_t))
    return d_dt / sig


def detrend_medminfilt(d, Kt=8, Kf=8, complex=False):
    """Detrend array using medminfilt statistic. See medminfilt.
    Args:
        d (array): data array of the shape (time, frequency) to detrend   
        Kt (int): size in time to apply medminfilter over
        Kf (int): size in frequency to apply medminfilter over
    Returns:        
         bool array: boolean array of flags    
    """
    if complex:
        d_sm = medminfilt(np.abs(d)*np.angle(d), 2*Kt + 1, 2*Kf + 1)
        d_rs = d - d_sm
        d_sq = np.abs(d_rs)**2
        # puts minmed on same scale as average
        sig = np.sqrt(medminfilt(d_sq, 2 * Kt + 1, 2 * Kf + 1,complex=True)) * (n.sqrt(Kt**2 + Kf**2) / .64)
        f = d_rs / sig
        return f
    else:
        d_sm = medminfilt(np.abs(d), 2 * Kt + 1, 2 * Kf + 1)
        d_rs = d - d_sm
        d_sq = np.abs(d_rs)**2
        # puts minmed on same scale as average
        if Kt == Kf:
            sig = np.sqrt(medminfilt(d_sq, 2 * Kt + 1)) * (Kt / .64)
        else:
            sig = np.sqrt(medminfilt(d_sq, 2 * Kt + 1, 2 * Kf + 1)) * (np.sqrt(Kt**2 + Kf**2) / .64)
        f = d_rs / sig
        return f

def detrend_medfilt(d, Kt=8, Kf=8):
    """Detrend array using a median filter.    
    Args:        
        d (array): data array to detrend.        
        K (int, optional): box size to apply medminfilt over    
    Returns:        
        bool array: boolean array of flags    
    """
    d = np.concatenate([d[Kt - 1::-1], d, d[:-Kt - 1:-1]], axis=0)
    d = np.concatenate([d[:, Kf - 1::-1], d, d[:, :-Kf - 1:-1]], axis=1)
    d_sm = medfilt(d, kernel_size=(2 * Kt + 1, 2 * Kf + 1))
    d_rs = d - d_sm
    d_sq = np.abs(d_rs)**2
    # puts median on same scale as average    
    # .456 scaling?    
    sig = np.sqrt(medfilt(d_sq, kernel_size=(2 * Kt + 1, 2 * Kf + 1)) / .456)
    f = d_rs / sig
    return f[Kt:-Kt, Kf:-Kf]



#############################################################################
# Various techniques for flagging RFI in interferometer visibilities
############################################################################# 

def watershed_flag(d, f=None, sig_init=6, sig_adj=2):
    '''Generates a mask for flags using a watershed algorithm.
    Returns a watershed flagging of an array that is in units of standard
    deviation (i.e. how many sigma the datapoint is from the center).

    Args:
        d (array)[time,freq]: 2D array to perform watershed on.
            d should be in units of standard deviations.
        f (array, optional): input flags. Same size as d.
        sig_init (int): number of sigma to flag above, initially.
        sig_adj (int): number of sigma to flag above for points
            near flagged points.

    Returns:
        bool array: Array of mask values for d.
    '''
    # mask off any points above 'sig' sigma and nan's.
    f1 = np.ma.array(d, mask=np.where(d > sig_init, 1, 0))
    f1.mask |= np.isnan(f1)
    if f is not None:
        f1.mask |= f

    # Loop over flagged points and examine adjacent points to see if they exceed sig_adj
    # Start the watershed
    prevx, prevy = 0, 0
    x, y = np.where(f1.mask)
    while x.size != prevx and y.size != prevy:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            prevx, prevy = x.size, y.size
            xp, yp = (x + dx).clip(0,f1.shape[0] - 1), (y + dy).clip(0, f1.shape[1] - 1)
            i = np.where(f1[xp, yp] > sig_adj)[0]  # if sigma > 'sigl'
            f1.mask[xp[i], yp[i]] = 1
            x, y = np.where(f1.mask)
    return f1.mask


def xrfi_simple(d, f=None, nsig_df=6, nsig_dt=6, nsig_all=0):
    '''Flag RFI using derivatives in time and frequency.
    Args:
        d (array): 2D data array of the shape (time, frequency) to flag 
        f (array, optional): input flags, defaults to zeros
        nsig_df (float, optional): number of sigma above median to flag in frequency direction
        nsig_dt (float, optional): number of sigma above median to flag in time direction
        nsig_all (float, optional): overall flag above some sigma. Skip if 0.
    Returns:
        bool array: mask array for flagging.
    '''
    if f is None:
        f = np.zeros(d.shape, dtype=np.bool)
    if nsig_df > 0:
        d_df = d[:, 1:-1] - .5 * (d[:, :-2] + d[:, 2:])
        d_df2 = np.abs(d_df)**2
        sig2 = np.median(d_df2, axis=1)
        sig2.shape = (-1, 1)
        f[:, 0] = 1
        f[:, -1] = 1
        f[:, 1:-1] = np.where(d_df2 / sig2 > nsig_df**2, 1, f[:, 1:-1])
    if nsig_dt > 0:
        d_dt = d[1:-1, :] - .5 * (d[:-2, :] + d[2:, :])
        d_dt2 = np.abs(d_dt)**2
        sig2 = np.median(d_dt2, axis=0)
        sig2.shape = (1, -1)
        f[0, :] = 1
        f[-1, :] = 1
        f[1:-1, :] = np.where(d_dt2 / sig2 > nsig_dt**2, 1, f[1:-1, :])
    if nsig_all > 0:
        ad = np.abs(d)
        med = np.median(ad)
        sig = np.sqrt(np.median(np.abs(ad - med)**2))
        f = np.where(ad > med + nsig_all * sig, 1, f)
    return f

def xrfi(d, f=None, Kt=8, Kf=8, sig_init=6, sig_adj=2):
    """Run best rfi excision we have. Uses detrending and watershed algorithms above.
    Args:
        d (array): 2D of data array.
        f (array, optional): input flag array
        K (int, optional): Box size for detrend
        sig_init (float, optional): initial sigma to flag.
        sig_adj (float, optional): number of sigma to flag adjacent to flagged data (sig_init)

    Returns:
        bool array: array of flags
    """
    nsig = detrend_medfilt(d, Kt=Kt, Kf=Kf)
    f = watershed_flag(np.abs(nsig), f=f, sig_init=sig_init, sig_adj=sig_adj)
    return f

# XXX split off median filter as one type of flagger
