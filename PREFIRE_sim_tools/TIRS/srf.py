import numpy as np
import netCDF4
import os.path
from PREFIRE_sim_tools import paths

def load_SRFdata(SRFfile=None, spec_grid='wl'):
    """ helper function to load the needed SRF data into a python dictionary."""
    if SRFfile is None:
        if spec_grid == 'wn':
            SRFfile = os.path.join(
                paths._shared_data_dir,
                'PREFIRE_SRF_v0.09.2_2020-02-21_PCRTM_grid.nc')
        else:
            SRFfile = os.path.join(
                paths._shared_data_dir,
                'PREFIRE_SRF_v0.09.2_2020-02-21.nc')

    SRFdata = dict(spec_grid=spec_grid)

    with netCDF4.Dataset(SRFfile, 'r') as nc:
        # newer versions of netCDF4 automatically create MaskedArray
        # even when it isn't used. For now, for this to turn off -
        # though this might not be the best long term solution...
        # (v1.5.4 has this behavior, I don't think the v1.2.8 in the default
        # shared install does. so somewhere in between was the change.)
        nc.set_auto_mask(False)
        srf = nc['srf_normed'][:]
        if spec_grid == 'wn':
            SRFdata["srf_w"] = nc['wavenum'][:]
            w1 = nc['channel_wavenum1'][:]
            w2 = nc['channel_wavenum2'][:]
            try:
                SRFdata["idealized_w"] = nc["channel_center_wavenum"][:]
                SRFdata["SRFweighted_w"] = nc["channel_mean_wavenum"][:]
            except:
                pass
        else:
            SRFdata["srf_w"] = nc['wavelen'][:]
            w1 = nc['channel_wavelen1'][:]
            w2 = nc['channel_wavelen2'][:]
            try:
                SRFdata["idealized_w"] = nc["channel_center_wavelen"][:]
                SRFdata["SRFweighted_w"] = nc["channel_mean_wavelen"][:]
            except:
                pass
    # a sort of hack to make this compatible with v0.09.2 (no footprint
    # dependence) or v0.10.4 or later (with fp-depedent axis in SRF.)
    # basically, makes the v0.9x versions 3D by replicating 8 copies.
    if srf.ndim == 2:
        nfp = 8
        srf = np.tile(srf[:,:,np.newaxis], (1,1,nfp))
        w1 = np.tile(w1[:, np.newaxis], (1,nfp))
        w2 = np.tile(w2[:, np.newaxis], (1,nfp))

    SRFdata["w"] = 0.5 * (w1 + w2)

    if spec_grid == 'wn':
        SRFdata["wr"] = np.concatenate([w2, w1[[-1],:]])
    else:
        SRFdata["wr"] = np.concatenate([w1, w2[[-1],:]])

    SRFdata.update(dict(srf=srf, w1=w1, w2=w2))

    return SRFdata


def apply_SRF_wngrid(y, footprint=1, SRFdata=None, SRFfile=None,
                     spec_grid='wl'):
    """
    applies SRF to an array of simulated radiances from PCRTM.

    Parameters
    ----------
    y : ndarray
        simulated radiances at high resolution fixed wavenumber grid
        from PCRTM. This only works for the sensor_id=2 in PCRTM.
        shape must be (5421, n)

    footprint : integer
        The SRFs are footprint dependent, so this input specifies
        which footprint should be used. Value is (1-8).

    SRFdata : dict or None
        for efficiency, the SRFdata can be sent as an optional
        keyword input. If None, then the SRF data is loaded from file.

    SRFfile : str or None
        optional keyword to specify the stored SRF data.
        If None, then a hardcoded file is used (v0.09.2 at the moment.)

    spec_grid : str
        should the output be in 'wl', or 'wn'? (wavelength or
        wavenumber). Default is 'wl'.
        For wl, the output SRF-integrated radiances will be in units of
        [W / (m^2 sr um)].
        For wn, the output SRF-integrated radiances will be in units of
        [mW / (m^2 sr cm^-1)].

    Returns
    -------
    w : ndarray
        the center spectral position for each channel, shape (63,).
        The units are controlled by the spec_grid input, either 
        [um] for 'wl', or [cm^-1] for 'wn'.
    wr : ndarray
        the spectral edges for each channel, shape (64,)
    yc : ndarray
        channel radiance, shaped (63, n), with units controlled by the
        spec_grid input.
    SRFdata : dict
        dictionary containing fields:
    spec_grid
        'wn' or 'wl' to specify the spectral grid. This will be the value
        of spec_grid given via the input keyword.
    srf
        normalized SRF, shaped (n_srf, 63)
    srf_w
        spectral grid for SRF, shaped (n_srf, 64)
    w
        center spectral position (same as 'w' return, above)
    wr
        spectral edges (same as 'wr' return, above)

    Example
    -------

    For efficiency (for example, if SRF is applied to radiance spectra
    in a large loop), then it may be useful to store the SRFdata and then
    re-use it when the function is re-entered.
    For example, running over a list of spectra, ylist:

    >>> SRFdata=None
    >>> for n in range(nspectra):
    >>>      wl, wlr, yc, SRFdata = apply_SRF_wngrid(ylist[n], SRFdata=SRFdata)
    >>>      <some calculation with yc)

    Otherwise, the SRFdata return could be ignored; it is mainly useful as a
    data cache, as in the previous example. To ignore it, just use the standard
    python underscore syntax:

    >>> wl, wlr, yc, _ = apply_SRF_wngrid(y)

    """

    if spec_grid not in ('wl', 'wn'):
        raise ValueError('spec_grid must be "wl" or "wn"')

    if SRFdata is None:
        SRFdata = load_SRFdata(SRFfile=SRFfile, spec_grid=spec_grid)
    else:
        if spec_grid != SRFdata['spec_grid']:
            raise ValueError('SRFdata did not contain the needed spec_grid = '+
                             spec_grid)

    # cache some additional data for spec_grid = wl, this is the
    # hardcoded PCRTM channel wavelengths;
    if spec_grid == 'wl':
        if not 'PCRTM_wn' in SRFdata:
            SRFdata['PCRTM_wn'] = np.arange(50.0, 2760.1, 0.5)
            SRFdata['PCRTM_wl'] = 1e4/SRFdata['PCRTM_wn'][::-1]
            # scaling array to convert per wn (1/cm) to per wl (um).
            # 1e4 um / cm, to change from [1/um**2] to [cm^-1/um].
            # and 1e-3 to go from mW to W.
            SRFdata['scaling'] = 10.0 / SRFdata['srf_w']**2
            SRFdata['scaling'] = SRFdata['scaling'][:,np.newaxis]

    if y.ndim == 1:
        y = y[:,np.newaxis]

    fp = footprint - 1
    if spec_grid == 'wl':
        # here, interpolate the PCRTM output (in wn space), to wl space,
        # and scale by the conversion factor. (1/wl^2)
        y_intp = np.zeros((SRFdata['srf_w'].shape[0], y.shape[1]))
        for n in range(y.shape[1]):
            # the left=0.0 is important, otherwise the interpolation to small 
            # wavelengths will extrapolate a small constant value towards wl=0.
            # this turns into a huge radiance spike (it would be like assuming
            # a radiance of 1 mW per cm^-1 at 50000 cm^-1, which is a lot.)
            y_intp[:,n] = np.interp(
                SRFdata['srf_w'], SRFdata['PCRTM_wl'],
                y[::-1,n], left=0.0)
        y_intp = y_intp * SRFdata['scaling']
        yc = np.dot(SRFdata['srf'][:,:,fp].T, y_intp)
    else:
        yc = np.dot(SRFdata['srf'][:,:,fp].T, y)

    w = SRFdata['w'][:,fp]
    wr = SRFdata['wr'][:,fp]

    return w, wr, yc, SRFdata


def apply_SRF_PCscore(y, footprint=1, SRFdata=None, SRFfile=None,
                      spec_grid='wl'):
    """
    applies SRF to an array of PC scores from PCRTM.

    Parameters
    ----------
    y : ndarray
        This array contains simulated PC scores from PCRTM. There are two options
        for input:
        First is a 2D float array, shaped (n, nPC), which for example
        would be loaded from the stored PC score data from the GFDL simulation.
        Second, this could be a (nband,) shaped object ndarray, which the (n, nPC)
        arrays containing each band's PC scores. This would be the EOF return from
        the PCRTM forward run.
        The stored PC spectra we are currently we are using are for 
        the sensor_id=2 (CLARREO 0.5 1/cm) which has nPC = 920, but only 400 of
        these (100 per band) are used.

    fp : integer
        The SRFs are footprint dependent, so this input specifies
        which footprint should be used. Value is (1-8).

    SRFdata : dict or None
       for efficiency, the SRFdata can be sent as an optional
       keyword input. If None, then the SRF data is loaded from file.

    SRFfile : str or None
       optional keyword to specify the stored SRF data.
       If None, then a hardcoded file is used (v0.09.2 at the moment.)

    spec_grid : str
        should the output be in 'wl', or 'wn'? (wavelength or
        wavenumber)
        For wl, the output SRF-integrated radiances will be in units of
        [W / (m^2 sr um)].
        For wn, the output SRF-integrated radiances will be in units of
        [mW / (m^2 sr cm^-1)].


    Returns
    -------
    w : ndarray
        the center spectral position for each channel, shape (63,).
        The units are controlled by the spec_grid input, either 
        [um] for 'wl', or [cm^-1] for 'wn'.
    wr : ndarray
        the spectral edges for each channel, shape (64,)
    yc : ndarray
        channel radiance, shaped (63, n), with units controlled by the
        spec_grid input.
    SRFdata : dict
        dictionary containing fields:
    spec_grid
        'wn' or 'wl' to specify the spectral grid
    C
        offset array, shaped (1,63)
    T
        transformation matrix, shaped (400,63)
    w
        center spectral position (same as 'w' return, above)
    wr
        spectral edges (same as 'wr' return, above)


    Example
    -------

    For efficiency (for example, if SRF is applied to radiance spectra
    in a large loop), then it may be useful to store the SRFdata and then
    re-use it when the function is re-entered.
    For example, running over a list of spectra, ylist:

    >>> SRFdata=None
    >>> for n in range(nspectra):
    >>>      _, _, yc, SRFdata = apply_SRF_PCscore(ylist[n], SRFdata=SRFdata)
    >>>      <some calculation with yc>

    """

    if spec_grid not in ('wl', 'wn'):
        raise ValueError('spec_grid must be "wl" or "wn"')

    if SRFdata is None:
        if SRFfile is None:
            SRFfile = os.path.join(
                paths._shared_data_dir, 'PREFIRE_SRF_v0.09.2_2020-02-21_PCRTM_grid.nc')
        with netCDF4.Dataset(SRFfile, 'r') as nc:
            if spec_grid == 'wn':
                w1 = nc['channel_wavenum1'][:]
                w2 = nc['channel_wavenum2'][:]
            else:
                w1 = nc['channel_wavelen1'][:]
                w2 = nc['channel_wavelen2'][:]

        SRFfile_aux = SRFfile.replace('_PCRTM_grid.nc', '_PCRTM_compound_SRF.nc')
        with netCDF4.Dataset(SRFfile_aux, 'r') as nc:
            numPC = nc['numPC'][:]
            if spec_grid == 'wn':
                C = nc['C_wn'][:]
                T = nc['T_wn'][:]
            else:
                C = nc['C_wl'][:]
                T = nc['T_wl'][:]

        # a sort of hack to make this compatible with v0.09.2 (no footprint
        # dependence) or v0.10.4 or later (with fp-depedent axis in SRF.)
        # basically, makes the v0.9x versions 3D by replicating 8 copies.
        if w1.ndim == 1:
            nfp = 8
            w1 = np.tile(w1[:,np.newaxis], (1,nfp))
            w2 = np.tile(w2[:,np.newaxis], (1,nfp))
            C = np.tile(C[:,:,np.newaxis], (1,1,nfp))
            T = np.tile(T[:,:,np.newaxis], (1,1,nfp))

        w = 0.5 * (w1 + w2)
        if spec_grid == 'wn':
            wr = np.concatenate([w2, w1[[-1],:]])
        else:
            wr = np.concatenate([w1, w2[[-1],:]])

        SRFdata = dict(C=C, T=T, numPC=numPC,
                       w=w, wr=wr, spec_grid=spec_grid)
    else:
        if spec_grid != SRFdata['spec_grid']:
            raise ValueError('SRFdata did not contain the needed spec_grid = '+
                             spec_grid)

    if y.dtype == np.object:
        nband = SRFdata['numPC'].shape[0]
        if y.shape[0] != nband:
            raise ValueError(
                'Mismatch between number of bands in '+
                'input y ({0:d})'.format(y.shape[0]) +
                'and PC data in SRF ({0:d})'.format(nband))
        y_perband = y
        if y_perband[0].ndim == 1:
            n_spectra = 1
        else:
            n_spectra = y_perband[0].shape[0]
        total_numPC = int(np.sum(SRFdata['numPC']))
        y = np.zeros((n_spectra, total_numPC))
        for b in range(nband):
            dst_slice = slice(
                np.sum(SRFdata['numPC'][:b]),
                np.sum(SRFdata['numPC'][:b+1]))
            src_slice = slice(0, SRFdata['numPC'][b])
            if y_perband[b].ndim == 1:
                y[:, dst_slice] = y_perband[b][src_slice, np.newaxis]
            else:
                y[:, dst_slice] = y_perband[b][:,src_slice]

    fp = footprint - 1

    T = SRFdata['T'][:,:,fp]
    C = SRFdata['C'][:,:,fp]
    yc = (np.dot(y, T) + C).T

    w = SRFdata['w'][:,fp]
    wr = SRFdata['wr'][:,fp]

    return w, wr, yc, SRFdata
