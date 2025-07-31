import numpy as np

# global constants
_k = 1.380650e-23
_k_units = 'J K^-1'
_c = 2.997925e8
_c_units = 'm s^-1'
_h = 6.626069e-34
_h_units = 'J s'

# to simplify the blackbody calcs
_A = _h * _c / _k
_B = 2 * _h * _c**2

def rad_wavenum(temp, wavenum):
    """
    compute blackbody spectral radiance per wavenumber.
    Note the inputs can be scalar floats, or ndarrays. If the inputs are
    arrays, they should be the same shape or broadcastable shapes.

    Parameters
    ----------
    temp : ndarray
        array of temperatures in [K]
    wavenum : ndarray
        array of wavenumbers in [cm^-1]

    Returns
    -------
    radiance : ndarray
        Blackbody radiance with units [mW m^-2 sr^-1 (cm^-1)^-1].
        This will have the same shape as the input arrays, or the
        broadcasted shape.

    """

    # convert the cm^-1 to m^-1 (MKS units) to match constants.
    wavenum_m = wavenum * 100.0
    z1 = _B * np.power(wavenum_m, 3) 
    z2 = np.exp(_A * wavenum_m / temp) - 1

    L = z1 / z2
    # convert back to per cm^-1
    L *= 100
    # convert W to mW
    L *= 1000

    return L


def dradT_wavenum(temp, wavenum):
    """
    compute blackbody spectral radiance derivative with respect
    to temperature, per wavenumber.
    Note the inputs can be scalar floats, or ndarrays. If the inputs are
    arrays, they should be the same shape or broadcastable shapes.

    Parameters
    ----------
    temp : ndarray
        array of temperatures in [K]
    wavenum : ndarray
        array of wavenumbers in [cm^-1]

    Returns
    -------
    radiance : ndarray
        Blackbody radiance temperature derivative,
        with units [mW m^-2 sr^-1 (cm^-1)^-1 K^-1].
        This will have the same shape as the input arrays, or the
        broadcasted shape.
    """

    # convert the cm^-1 to m^-1 (MKS units) to match constants.
    wavenum_m = wavenum * 100.0
    e1 = np.exp(_A * wavenum_m / temp)
    z1 = _A * _B * np.power(wavenum_m, 4) * e1
    z2 = np.power( (e1 - 1)*temp, 2)

    dLdT = z1 / z2
    # convert back to per cm^-1
    dLdT *= 100
    # convert W to mW
    dLdT *= 1000

    return dLdT


def rad_wavelen(temp, wavelen):
    """
    compute blackbody spectral radiance per wavelength.
    Note the inputs can be scalar floats, or ndarrays. If the inputs are
    arrays, they should be the same shape or broadcastable shapes.

    Parameters
    ----------
    temp : ndarray
        array of temperatures in [K]
    wavelen : ndarray
        array of wavelengths in [um]

    Returns
    -------
    radiance : ndarray
        Blackbody radiance with units [W m^-2 sr^-1 um^-1].
        This will have the same shape as the input arrays, or the
        broadcasted shape.

    """

    # convert wavelen from um to m (MKS units) to match constants.
    wavelen_m = wavelen * 1e-6

    z1 = _B / np.power(wavelen_m, 5)
    z2 = np.exp(_A / (wavelen_m * temp)) - 1

    L = z1 / z2
    # convert back to per um
    L *= 1e-6

    return L

def dradT_wavelen(temp, wavelen):
    """
    compute blackbody spectral radiance derivative with respect
    to temperature, per wavelength.
    Note the inputs can be scalar floats, or ndarrays. If the inputs are
    arrays, they should be the same shape or broadcastable shapes.

    Parameters
    ----------
    temp : ndarray
        array of temperatures in [K]
    wavelen : ndarray
        array of wavelengths in [um]

    Returns
    -------
    radiance : ndarray
        Blackbody radiance temperature derivative,
        with units [W m^-2 sr^-1 um-1 K^-1].
        This will have the same shape as the input arrays, or the
        broadcasted shape.
    """

    # convert wavelen from um to m (MKS units) to match constants.
    wavelen_m = wavelen * 1e-6
    e1 = np.exp(_A / (wavelen_m * temp))
    z1 = _A * _B / np.power(wavelen_m, 6) * e1
    z2 = np.power( (e1 - 1)*temp, 2 )

    dLdT = z1 / z2
    # convert back to per um
    dLdT *= 1e-6

    return dLdT


def btemp_wavenum(rad, wavenum):
    """
    compute brightness temperature for an input radiance and
    wavenumber, using monochromatic assumptions.
    Note the inputs can be scalar floats, or ndarrays. If the inputs are
    arrays, they should be the same shape or broadcastable shapes.

    Parameters
    ----------
    rad : ndarray
        array of radiances, in [mW m^-2 sr^-1 (cm^-1)^-1].
        This is the same units as retured in rad_wavenum in this module.
    wavenum : ndarray
        array of wavenumbers in [cm^-1]

    Returns
    -------
    btemp : ndarray
        brightness temperature in [K]. This will have the same shape as
        the input arrays, or the broadcasted shape.
    """

    # convert the cm^-1 to m^-1 (MKS units) to match constants.
    wavenum_m = wavenum * 100.0
    # convert mW to W. (1e-3), and per cm^-1 to per m^-1. (1e-2)
    rad_Wm = 1e-5 * rad
    tmp = np.log( _B * np.power(wavenum_m, 3) / rad_Wm + 1 )
    BT = _A * wavenum_m / tmp

    return BT
