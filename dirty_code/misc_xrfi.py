

def toss_times_freqs(mask, sig_t=6, sig_f=6):
    """XXX what does this function do? Needs test."""
    f1ch = np.average(f1.mask, axis=0); f1ch.shape = (1,-1)
    #The cut off value is a made up number here...sig = 'sig' if none flagged.
    f1.mask = np.logical_or(f1.mask, np.where(f1 > sig_init*(1-f1ch), 1, 0))
    f1t = np.average(f1.mask, axis=1) # band-avg flag vs t
    ts = np.where(f1t > 2*np.median(f1t))
    f1.mask[ts] = 1
    f1f_sum = np.sum(f1.filled(0), axis=0)
    f1f_wgt = np.sum(np.logical_not(f1.mask), axis=0)
    f1f = f1f_sum / f1f_wgt.clip(1,np.Inf)
    fs = np.where(f1f > 2)
    f1.mask[:,fs] = 1
    mask = f1.mask
    return mask
