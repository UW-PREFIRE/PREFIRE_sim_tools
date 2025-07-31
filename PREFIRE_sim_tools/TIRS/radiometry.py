import numpy as np
import netCDF4
import os.path
from PREFIRE_sim_tools import paths

import scipy.interpolate

def _load_LUTdata(SRFfile=None, spec_grid='wl'):
    """ helper function to load the needed SRF data into a python dictionary. """
    if SRFfile is None:
        if spec_grid == 'wn':
            SRFfile = os.path.join(
                paths._shared_data_dir,
                'PREFIRE_SRF_v0.09.2_2020-02-21_PCRTM_grid.nc')
        else:
            SRFfile = os.path.join(
                paths._shared_data_dir,
                'PREFIRE_SRF_v0.09.2_2020-02-21.nc')
    with netCDF4.Dataset(SRFfile, 'r') as nc:
        NEDR = nc['NEDR'][:]
        rad = nc['rad'][:]
        drad_dT = nc['drad_dT'][:]
        T = nc['T_grid'][:]

    if spec_grid == 'wn':
        # I think the upstream SRF file has the wn - space NEDR
        # as W instead of mW, so convert here. this is a hack, and
        # should get an upstream fix...
        NEDR *= 1e3

    # a sort of hack to make this compatible with v0.09.2 (no footprint
    # dependence) or v0.10.4 or later (with fp-depedent axis in SRF.)
    # basically, makes the v0.9x versions 3D but with only 1 valid fp.
    if NEDR.ndim == 1:
        nfp = 8
        NEDR = np.tile(NEDR[:,np.newaxis], (1,nfp))
        rad = np.tile(rad[:,:,np.newaxis], (1,1,nfp))
        drad_dT = np.tile(drad_dT[:,:,np.newaxis], (1,1,nfp))

    LUTdata = dict(NEDR=NEDR, rad=rad, drad_dT=drad_dT,
                   spec_grid=spec_grid, T=T)

    return LUTdata


def btemp(channel_rad, footprint=1, LUTdata=None, SRFfile=None,
          spec_grid='wl'):
    """
    compute TIRS brightness temperatures, given an input array of
    channel radiances.

    Parameters
    ----------
    channel_rad : ndarray
        an array of channel radiances. This must have a
        an array shape such that the trailing dimension is (63), the
        number of TIRS channels. including the masked channels.
        this can be N dimensions; for example shapes of (63,), (i,63), 
        (i,j,63), etc, are all acceptable.

    footprint : integer
        The SRFs are footprint (scene) dependent, so this input specifies
        which footprint should be used. Value is (1-8).

    LUTdata : dict or None
        for efficiency, the LUTdata can be sent as an optional
        keyword input. If None, then the SRF data is loaded from file.

    SRFfile : str or None
        optional keyword to specify the stored SRF data.
        If None, then a hardcoded file is used (v0.03 at the moment.)

    spec_grid : str
        must be either 'wl', or 'wn', and specifies the units of the input
        channel radiance; 
        For wl, the units are assumed to be [W / (m^2 sr um)],
        For wn, the units are assumed to be [mW / (m^2 sr cm^-1)].
    

    Returns
    -------
    channel_bt : ndarray
        an array of the radiance converted to brightness temperature in K.
        This should have the same shape as the input channel_rad array.
    channel_NEDT : ndarray
        an array of the noise-equivalent delta-Temperature [K] for each
        of the input radiance values. Since NEDT depends on the radiance level
        itself (unlike the NEDR which is a function of the channel number, but
        not of the radiance level), this is computed for each channel radiance.

    Note that negative radiances will produce zero BT in the output array.
    Mathematically, a negative radiance does not have a defined BT, but since
    this method is LUT based, the extrapolator currently is fine extrapolating
    outside the table. This ends up making crazy outlier values, though, so
    safer to prevent that extrapolation so we set these to zero.
    """

    if spec_grid not in ('wl', 'wn'):
        raise ValueError('spec_grid must be "wl" or "wn"')

    if LUTdata is None:
        LUTdata = _load_LUTdata(SRFfile=SRFfile, spec_grid=spec_grid)
    else:
        if spec_grid != LUTdata['spec_grid']:
            raise ValueError('LUTdata did not contain the needed spec_grid = '+
                             spec_grid)

    fp = footprint - 1

    channel_BTemp = np.zeros_like(channel_rad)
    channel_NEDT = np.zeros_like(channel_rad)
    for c in range(LUTdata['NEDR'].shape[0]):
        if LUTdata['NEDR'].mask[c,fp]:
            continue
        Srad = scipy.interpolate.CubicSpline(
            LUTdata['rad'][:,c,fp], LUTdata['T'])
        # could use drad_dT, this seems simpler
        dSrad_dL = Srad.derivative(1)
        channel_BTemp[...,c] = Srad(channel_rad[...,c])
        channel_NEDT[...,c] = dSrad_dL(channel_rad[...,c]) * LUTdata['NEDR'][c,fp]
    negative_msk = channel_rad < 0
    if np.any(negative_msk):
        channel_BTemp[negative_msk] = 0
        channel_NEDT[negative_msk] = 0

    return channel_BTemp, channel_NEDT, LUTdata


def bbrad(temperature, footprint=1, LUTdata=None, SRFfile=None,
          spec_grid='wl'):
    """
    compute TIRS blackbody radiances, given an input array of
    temperatures

    Parameters
    ----------
    temperature : ndarray
        an array of channel temperatures, of any shape

    footprint : integer
        The SRFs are footprint (scene) dependent, so this input specifies
        which footprint should be used. Value is (1-8).

    LUTdata : dict or None
        for efficiency, the LUTdata can be sent as an optional
        keyword input. If None, then the SRF data is loaded from file.

    SRFfile : str or None
        optional keyword to specify the stored SRF data.
        If None, then a hardcoded file is used (v0.03 at the moment.)

    spec_grid : str
        must be either 'wl', or 'wn', and specifies the units of the input
        channel radiance;
        For wl, the output units will be [W / (m^2 sr um)],
        For wn, the output units will be [mW / (m^2 sr cm^-1)].

    Returns
    -------
    channel_rad : ndarray
        an array of channel radiances. Shape will be temperature shape + (63,)

    """

    if spec_grid not in ('wl', 'wn'):
        raise ValueError('spec_grid must be "wl" or "wn"')

    if LUTdata is None:
        LUTdata = _load_LUTdata(SRFfile=SRFfile, spec_grid=spec_grid)
    else:
        if spec_grid != LUTdata['spec_grid']:
            raise ValueError('LUTdata did not contain the needed spec_grid = '+
                             spec_grid)

    fp = footprint - 1

    rad = np.zeros(temperature.shape + (LUTdata['rad'].shape[1],))
    for c in range(rad.shape[-1]):
        if LUTdata['NEDR'].mask[c,fp]:
            continue
        Srad = scipy.interpolate.CubicSpline(
            LUTdata['T'], LUTdata['rad'][:,c,fp])
        rad[...,c] = Srad(temperature)

    return rad, LUTdata
