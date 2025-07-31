import datetime

import numpy as np
import scipy.special

def _get_scan_HWHM_function(fnumber):
    """
    returns a "scan function" according to a function number
    (matching the LBLRTM documentation), and a suggested window size for 
    the array product.

    The returned scan function takes 3 inputs: mono_wn, HWHM, wn;
    the mono_wn is a 1D vector with the wavenumber grid, and HWHN and wn
    describe the width and center frequency of the scan function.

    the window size is in multiples of the HWHM (half-width, half-max.)
    The input mono_wn should cover +/- window size * HWHM away from the v0.
    Note, particularly for type 3/4 (sinc), the convolution window
    must be very wide to minimize errors; particuly with sinc, the 
    default window might not be wide enough.

    scan functions by number, with default window sizes (multiples
    of HWHM):
    0: square "tophat" function. equal to 1 if |v-v0| < HWHM; 1.1
    1: triangle:        2.1
    2: gaussian:        5
    3: sinc**2:        25 
    4: sinc:           50
    5: beer window (not fully implemented yet)
    6: hamming window: 20
    7: hann window:    10
    """
    if fnumber == 0:
        window = 1.1
        def F(mono_wn, HWHM, wn):
            s = (np.abs(mono_wn-wn) < HWHM).astype(float)
            return s
    elif fnumber == 1:
        window = 2.1
        def F(mono_wn, HWHM, wn):
            a = HWHM * 2.0
            s = 1 - np.abs((mono_wn - wn)/a)
            s[s<0] = 0
            return s
    elif fnumber == 2:
        window = 5.0
        def F(mono_wn, HWHM, wn):
            a = HWHM / np.sqrt(np.log(4.0))
            u = (mono_wn-wn) / a
            s = np.exp(-0.5 * u**2)
            return s
    elif fnumber == 3:
        window = 40.0
        def F(mono_wn, HWHM, wn):
            # from scipy optimize
            # f = lambda x: np.sinc(x)**2 - 0.5
            # 1/scipy.optimize.newton(f, 0.2)
            # Note the LBLRTM doc is somewhat wrong with factors of pi.
            a = 2.2576091382 * HWHM
            u = (mono_wn-wn) / a
            s = np.sinc(u)**2
            return s
    elif fnumber == 4:
        window = 80.0
        def F(mono_wn, HWHM, wn):
            # f = lambda x: np.sinc(x) - 0.5
            # 1/scipy.optimize.newton(f, 0.2)
            a = 1.657400240 * HWHM
            u = (mono_wn-wn) / a
            s = np.sinc(u)
            return s
    elif fnumber == 5:
        raise NotImplementedError('fnumber 5 is not yet implemented')
        # I think this is the right math, but I'm not sure how to deal with the
        # zero here - there is a 0/0 singularity, not trivially evaluatable 
        # via L'hopital's rule (the Bessel function only has a recurrance
        # relationship for the derivative, and since this is not an integer
        # value it will always be 0/inf at x=0.)
        # Also, this function (scipy.special.jv) is very slow, compared to
        # others; as such, I don't think this is usable at the moment,
        # So I have left it NotImplemented.
        window = 40.0
        def F(mono_wn, HWHM, wn):
            a = 2.100669 * HWHM
            u = np.abs(mono_wn - wn) * (2*np.pi / a)
            s = scipy.special.jv(2.5, u) / u**2.5
            return s
    elif fnumber == 6:
        window = 20.0
        def F(mono_wn, HWHM, wn):
            a = 2.195676 * HWHM
            # not sure where LBLRTM number comes from
            # seems a little off, but close to 
            # https://en.wikipedia.org/wiki/Window_function#Hamming_window
            # the a0 value there (0.53836), convert to c1:
            # c1 = (1/a0 - 1)/2 = 0.428747
            c1 = 0.428752
            a = 1.0978355588 * HWHM
            u = (mono_wn-wn) / a
            s = np.sinc(u) + c1*(np.sinc(u + 1) + np.sinc(u - 1))
            return s
    elif fnumber == 7:
        window = 10.0
        def F(mono_wn, HWHM, wn):
            a = HWHM
            u = (mono_wn-wn) / a
            s = np.sinc(u) + 0.5*(np.sinc(u + 1) + np.sinc(u - 1))
            return s
    else:
        raise ValueError('unknown fnumber')

    return F, window


def apply_scanfn(mono_wn, mono_y, HWHM, v0, v1, dv, fnumber,
                 window_size_scale=1.0, div=None):
    """
    apply scan function in a convolution with sampling.

    Parameters
    ----------

    mono_wn : ndarray
        1D array of monochromatic wavenumbers. typically, this is [1/cm].
    mono_y : ndarray
        1D array of monochromatic values (usually the radiance).
        Typically, this is [mW/(m^2 sr cm^-1)]
    HWHM : float
        scalar half-width half max of the scanning function. Must match
        the units given in mono_wn
    v0 : float
        scalar, starting wavenumber for the sample grid.
        same units as mono_wn.
    v1 : float
        scalar, ending wavenumber for the sample grid.
    dv : float
        scalar, sample spacing.
    fnumber : int
        scalar, number specifying the scan function.
        This is following the LBLRTM convention:
        0 = square "tophat" function. equal to 1 if abs(v-v0) < HWHM,
        1 = triangle,
        2 = gaussian,
        3 = sinc**2,
        4 = sinc,
        5 = beer window (not implemented yet),
        6 = hamming window,
        7 = hann window
    window_size: float
        scalar, multiplier for the window size used in the convolution.
        Each scan function has an (ad hoc) window size, this keyword can
        be used to expand or shrink the window. This is a tradeoff between
        accuracy/speed, except for the rectangle and triangle scan functions
        which have finite support.
        default is 1.0 (no scaling).
    div : int
        set to an integer to print back status to console for the calculation
        timing. This can be a very long calculation, so this information might
        be useful.
        default of None will mean this information is not printed.

    Returns
    -------

    sampled_wn : ndarray
        1D array containing the wavenumbers for the sampled data.
        This is equal to: np.arange(v0, v1, dv), for example,
        if v0, v1, dv are equal to 10, 16, 2, then the sample wn array
        would be [10, 12, 14].
    sampled_y : ndarray
        1D array, same shape as sampled_wn, with the convolved and sampled
        output.
    """


    wn_out = np.arange(v0, v1, dv, dtype=float)

    if mono_y.ndim == 1:
        y_out = np.zeros_like(wn_out)
    elif mono_y.ndim == 2:
        y_out = np.zeros(wn_out.shape + mono_y.shape[1:])
    else:
        raise ValueError('mono_y input must be 1D or 2D')

    F, window_size = _get_scan_HWHM_function(fnumber)
    window_size *= window_size_scale

    t0 = datetime.datetime.now()
    starts = mono_wn.searchsorted(wn_out - HWHM * window_size)
    stops = mono_wn.searchsorted(wn_out + HWHM * window_size)

    for w in range(wn_out.shape[0]):

        if div is not None:
            if ((w%div) == 0) and (w > 0):
                print('{0:d} of {1:d} etime {2:s}'.format(
                    w, wn_out.shape[0], str(datetime.datetime.now()-t0)))

        ss = slice(starts[w], stops[w])
        mono_wn_sub = mono_wn[ss]

        s = F(mono_wn_sub, HWHM, wn_out[w])
        s = s / s.sum()

        if mono_y.ndim == 1:
            mono_y_sub = mono_y[ss]
            y_out[w] = np.dot(s, mono_y_sub)
        else:
            mono_y_sub = mono_y[ss, :]
            s = s[np.newaxis, :]
            y_out[w, :] = np.dot(s, mono_y_sub)

    return wn_out, y_out
