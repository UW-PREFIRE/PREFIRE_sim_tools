"""
process_srf.py:

load raw output from MATLAB model, and created SRF at the PCRTM
wavenumber grid.

This is intended to be run directly from the scripts subdirectory, 
invoked at command line:
$ python process_srf.py

There are assumed local paths for the input file and output location.
"""


import os.path
import subprocess
import datetime

import numpy as np
import netCDF4
from scipy.io.matlab import loadmat

from PREFIRE_sim_tools import PCRTM_utils, paths
from PREFIRE_sim_tools.utils import blackbody

def assess_pooled_NEDR():
    """
    analysis function, mainly hardcoded stuff, to get the classification thresholds
    "good/higher noise/so much noise we don't use it" or something.
    The instruments do have a sort of bimodality to the noise.
    """

    srfdir = '/home/mmm/projects/PREFIRE_L1_data/'
    srf1file = srfdir + 'instrument_1_NETD_NEDR_SRF.mat'
    srf2file = srfdir + 'instrument_2_NETD_NEDR_SRF.mat'

    srf1 = loadmat(srf1file)
    srf2 = loadmat(srf2file)

    # drop channel 0 (total radiance)
    TIRS1_NEDR = srf1['NEDR'][1:]
    TIRS2_NEDR = srf2['NEDR'][1:]

    # guard against typos/etc - make sure shapes match, and the
    # data in TIRS1 and TIRS2 is actually different
    assert TIRS1_NEDR.shape == TIRS2_NEDR.shape
    assert np.all(TIRS1_NEDR == TIRS2_NEDR) == 0

    pooled_NEDR = np.concatenate([TIRS1_NEDR, TIRS2_NEDR], axis=1)

    pooled_sorted_NEDR = np.sort(pooled_NEDR, axis=1)

    min_NEDR = pooled_sorted_NEDR[:,[0]]

    scaled_thresh_low = 2.5
    scaled_thresh_high = 10.0

    NEDR_threshold_low = min_NEDR[:,0] * scaled_thresh_low
    NEDR_threshold_high = min_NEDR[:,0] * scaled_thresh_high

    TIRS1_scaled_NEDR = TIRS1_NEDR / min_NEDR
    TIRS2_scaled_NEDR = TIRS2_NEDR / min_NEDR

    TIRS1_mask = np.zeros(TIRS1_NEDR.shape, np.int8)
    TIRS2_mask = np.zeros(TIRS2_NEDR.shape, np.int8)

    TIRS1_mask[TIRS1_scaled_NEDR > scaled_thresh_low] = 1
    TIRS1_mask[TIRS1_scaled_NEDR > scaled_thresh_high] = 2
    TIRS1_mask[np.isinf(TIRS1_NEDR)] = 3
    TIRS2_mask[TIRS2_scaled_NEDR > scaled_thresh_low] = 1
    TIRS2_mask[TIRS2_scaled_NEDR > scaled_thresh_high] = 2
    TIRS2_mask[np.isinf(TIRS2_NEDR)] = 3

    with netCDF4.Dataset('../data/PREFIRE_TIRS1_SRF_v11_FPA_mask.nc', 'w') as nc:

        nc.createDimension('channel', TIRS1_NEDR.shape[0])
        nc.createDimension('footprint', TIRS1_NEDR.shape[1])

        nc.createVariable('NEDR_threshold_low', int, ('channel',))
        nc.createVariable('NEDR_threshold_high', int, ('channel',))

        nc.createVariable('detector_mask', np.int8, ('channel', 'footprint'))

        nc.createVariable('NEDR', float, ('channel', 'footprint'))
        nc.createVariable('scaled_NEDR', float, ('channel', 'footprint'))

        nc['NEDR_threshold_low'][:] = NEDR_threshold_low
        nc['NEDR_threshold_high'][:] = NEDR_threshold_high
        
        nc['detector_mask'][:] = TIRS1_mask
        nc['detector_mask'].setncattr(
            'long_name', 'detector mask')
        nc['detector_mask'].setncattr(
            'description', '0=normal 1=high noise 2=extreme noise 3=masked')

        nc['NEDR'][:] = TIRS1_NEDR
        nc['NEDR'].setncattr(
            'long_name', 'Noise-Equivalent Delta Radiance')
        nc['NEDR'].setncattr(
            'description', '1-sigma radiance noise level')
        nc['NEDR'].setncattr('units', 'W/m^2/sr/um')

        nc['scaled_NEDR'][:] = TIRS1_scaled_NEDR
        nc['scaled_NEDR'].setncattr(
            'long_name', 'Noise-Equivalent Delta Radiance '+
            'scaled to the minimum value across all scenes pooled from '+
            'both TIRS instruments')
        nc['scaled_NEDR'].setncattr(
            'description', 'scaled 1-sigma radiance noise level')
        nc['scaled_NEDR'].setncattr('units', 'none')

    with netCDF4.Dataset('../data/PREFIRE_TIRS2_SRF_v11_FPA_mask.nc', 'w') as nc:

        nc.createDimension('channel', TIRS2_NEDR.shape[0])
        nc.createDimension('footprint', TIRS2_NEDR.shape[1])

        nc.createVariable('NEDR_threshold_low', int, ('channel',))
        nc.createVariable('NEDR_threshold_high', int, ('channel',))

        nc.createVariable('detector_mask', np.int8, ('channel', 'footprint'))

        nc.createVariable('NEDR', float, ('channel', 'footprint'))
        nc.createVariable('scaled_NEDR', float, ('channel', 'footprint'))

        nc['NEDR_threshold_low'][:] = NEDR_threshold_low
        nc['NEDR_threshold_high'][:] = NEDR_threshold_high

        nc['detector_mask'][:] = TIRS2_mask
        nc['detector_mask'].setncattr(
            'long_name', 'detector mask')
        nc['detector_mask'].setncattr(
            'description', '0=normal 1=high noise 2=extreme noise 3=masked')

        nc['NEDR'][:] = TIRS2_NEDR
        nc['NEDR'].setncattr(
            'long_name', 'Noise-Equivalent Delta Radiance')
        nc['NEDR'].setncattr(
            'description', '1-sigma radiance noise level')
        nc['NEDR'].setncattr('units', 'W/m^2/sr/um')

        nc['scaled_NEDR'][:] = TIRS2_scaled_NEDR
        nc['scaled_NEDR'].setncattr(
            'long_name', 'Noise-Equivalent Delta Radiance '+
            'scaled to the minimum value across all scenes pooled from '+
            'both TIRS instruments')
        nc['scaled_NEDR'].setncattr(
            'description', 'scaled 1-sigma radiance noise level')
        nc['scaled_NEDR'].setncattr('units', 'none')


def create_btemp_tables(SRF_file, T_min=150.0, T_max=350.0,
                        T_step=1.0, dT_incr=0.05):
    """
    create brightness temperature tables, to facilitate conversion
    of TIRS radiances to brightness temps.

    this is basically a brute force numeric integration of the SRF
    over a sweep in Temperature.

    Also, creates dL/dT for each channel, with a finite difference
    calculation given by dT_incr.

    outputs data in a python dictionary.

    Parameters
    ----------
    SRF_file : str
        path to the SRF data file, the netCDF4 produced by process_srf().
    T_min : float
        minimum temperature to use in the calculation grid [K]
    T_max : float
        maximum temperature to use in the calculation grid [K]
    T_step : float
        temperature step size to use in the calculation grid [K]
    T_incr : float
        delta to use in the finite difference calculation of dL/dT [K]

    Returns
    -------
    dat : dict
        python dictionary with the following keys:
    T
        the temperature grid. this is whatever grid is selected by the
        values of T_min, T_max, T_step. shaped (num_T,)
    rad
        the spectral radiance in the wavelength grid. This is computed for
        each channel, with resulting shape (num_T, num_channel).
        If the input SRF file contains wavelength grid, the units are
        Units are [W / (m2 sr um)]; if the grid was wavenumber, the units
        are [mW / (m2 sr cm-1)].
    drad_dT
        the radiance derivative, the same shape as rad
        units are the same as rad with an additional [1/K] multiplier.
    """

    def B_wn(T, wn):
        # returns BB radiance directly in mW / (m2 sr cm-1)
        return blackbody.rad_wavenum(T, wn)
    def B_wl(T, wl):
        # returns BB radiance directly in W / (m2 sr um)
        return blackbody.rad_wavelen(T, wl)

    with netCDF4.Dataset(SRF_file, 'r') as nc:
        if 'wavenum' in nc.variables:
            w_grid = nc['wavenum'][:]
            B = B_wn
        else:
            w_grid = nc['wavelen'][:]
            B = B_wl
        SRF = nc['srf_normed'][:]

    Nw, Nc, Nfp = SRF.shape

    T_grid = np.arange(T_min, T_max+0.5*T_step, T_step)

    NT = T_grid.shape[0]
    rad = np.zeros((NT, Nc, Nfp))
    drad_dT = np.zeros((NT, Nc, Nfp))

    for t in range(NT):
        # WL integration is easy since it is on even grid, use 
        # dot product with the normalized SRF.
        B1 = B(T_grid[t], w_grid)
        B2 = B(T_grid[t] + dT_incr, w_grid)
        for fp in range(Nfp):
            rad[t,:,fp] = np.dot(B1, SRF[:,:,fp])
            drad_dT[t,:,fp] = np.dot((B2 - B1) / dT_incr, SRF[:,:,fp])

    dat = dict(T=T_grid, rad=rad, drad_dT=drad_dT)

    return dat


def compute_compound_SRF(SRF_file):
    """
    experimental function to generate the linear transformation
    to apply to radiance spectra represented by PCRTM PC scores
    to directly get SRF-convolved PREFIRE radiances.

    The output netCDF file has two variables, T, C;
    for an array of PCRTM radiances, L_PCRTM, shaped (nwn, nspec)
    (e.g., (5421, 100) for 100 spectra);
    then TIRS radiances are: 

    L_TIRS = np.dot(T, L_PCRTM) + C
    """

    with netCDF4.Dataset(SRF_file, 'r') as nc:
        wl_grid = nc['wavelen'][:]
        SRF_wl = nc['srf'][:]

    # the PCRTM fixed wnum grid.
    wn1, wn2 = 50.0, 2760.0
    dwn = 0.5
    wn_grid = np.arange(wn1, wn2+0.1*dwn, dwn)
    N = wn_grid.shape[0]
    Nwl, Nc, Nfp = SRF_wl.shape

    # interpolate SRF to the wn_grid, and normalize.
    SRF = np.zeros((N, Nc, Nfp))
    x = 1e4/wl_grid[::-1]
    for c, fp in np.ndindex((Nc, Nfp)):
        if np.sum(SRF_wl[:,c,fp]) > 0:
            y = SRF_wl[::-1,c,fp]
            SRF[:,c,fp] = np.interp(wn_grid, x, y)
            SRF[:,c,fp] = SRF[:,c,fp]/np.sum(SRF[:,c,fp])

    # compute a second version that can be applied to the wn_grid, but
    # converts the radiance directly from per-wn to per-wl.
    SRF_per_wl = SRF.copy()

    wl = 1e4/wn_grid
    wl_diff = np.diff(wl)
    wl_delta = np.r_[ wl[1]-wl[0], 0.5*(wl_diff[:-1]+wl_diff[1:]), wl[-1]-wl[-2] ]
    for c, fp in np.ndindex((Nc, Nfp)):
        if np.sum(SRF[:,c,fp]) > 0:
            SRF_per_wl[:,c,fp] = SRF[:,c,fp] * wl_delta / wl**2 / np.trapz(SRF[:,c,fp], wl)

    # now, combine with the PC Coefficient matrix.

    PCfile = os.path.join(paths._data_dir, 'PCRTM_Pcnew_id2.dat')
    pcdat = PCRTM_utils.read_PC_file(PCfile)
    nband = pcdat['numPC_perband'].shape[0]

    # makes a block "diagonal" matrix, with zeros for cross terms between
    # the PCRTM bands. shaped (nPC, nch)
    U = np.zeros((pcdat['numPC'], pcdat['numch']))
    pc_ct = 0
    ch_ct = 0
    for b in range(nband):
        r1 = pc_ct
        r2 = pc_ct + pcdat['numPC_perband'][b]
        c1 = ch_ct
        c2 = ch_ct + pcdat['numch_perband'][b]
        U[r1:r2, c1:c2] = pcdat['PCcoef'][b]
        pc_ct = r2
        ch_ct = c2

    # broadcasted multiplication of P over each row in U.
    Pscale = np.concatenate(pcdat['Pstd'])[np.newaxis,:]
    Uscaled = U * Pscale

    # T = transformation matrix, shaped (nPC, nSRFchannel) = (400,63)
    # C = constant offset vector, shaped (1, nSRFchannel) = (1,63)
    # there are now 8 of these (one per footprint).
    # use np.einsum to do this in one step:
    # Uscaled (400,5421) * SRF(5421,63,8) -> T (400,63,8)
    # Pscale    (1,5421) * SRF(5421,63,8) -> C (1,63,8)
    T_wn = np.einsum('ij,jkl', Uscaled, SRF)
    C_wn = np.einsum('ij,jkl', Pscale, SRF)

    T_wl = np.einsum('ij,jkl', Uscaled, SRF_per_wl)
    C_wl = np.einsum('ij,jkl', Pscale, SRF_per_wl)

    # final unit conversion. PCRTM computes mW/(m2 sr cm-1) natively,
    # and the conversion to mW/(m2 sr um) will require a 1e4 factor
    # (1e4 um per cm); since we want W/(m2 sr um), then this means
    # just a factor of 10 for the wl arrays.
    T_wl *= 10
    C_wl *= 10

    out_file = os.path.split(SRF_file)[-1]
    out_file = out_file.replace('.nc', '_PCRTM_compound_SRF.nc')
    out_file = os.path.join(paths._data_dir, out_file)

    with netCDF4.Dataset(out_file, 'w') as nc:

        nc.createDimension('PC_score', pcdat['numPC'])
        nc.createDimension('TIRS_channel', 63)
        nc.createDimension('single', 1)
        nc.createDimension('bands', nband)
        nc.createDimension('footprints', Nfp)

        nc.createVariable('T_wn', float, ('PC_score', 'TIRS_channel', 'footprints'))
        nc.createVariable('C_wn', float, ('single', 'TIRS_channel', 'footprints'))
        nc.createVariable('T_wl', float, ('PC_score', 'TIRS_channel', 'footprints'))
        nc.createVariable('C_wl', float, ('single', 'TIRS_channel', 'footprints'))
        nc.createVariable('numPC', int, ('bands',))
        nc.createVariable('numCh', int, ('bands',))

        nc['C_wn'][:] = C_wn
        nc['T_wn'][:] = T_wn

        nc['C_wl'][:] = C_wl
        nc['T_wl'][:] = T_wl

        nc['C_wn'].setncattr(
            'description', 
            'constant offset for conversion to per-wavenumber TIRS channel '+
            'radiance [mW / (m^2 sr cm^-1)]')
        nc['T_wn'].setncattr(
            'description', 
            'matrix transform for conversion to per-wavenumber TIRS channel '+
            'radiance [mW / (m^2 sr cm^-1)]')
        nc['C_wl'].setncattr(
            'description', 
            'constant offset for conversion to per-wavelen TIRS channel '+
            'radiance [W / (m^2 sr um)]')
        nc['T_wl'].setncattr(
            'description', 
            'matrix transform for conversion to per-wavelen TIRS channel '+
            'radiance [W / (m^2 sr um)]')

        nc['numPC'][:] = pcdat['numPC_perband']
        nc['numCh'][:] = pcdat['numch_perband']
        nc['numPC'].setncattr('description', 'number of PCs per band')
        nc['numCh'].setncattr('description', 'number of channels per band')


def _create_common_variables(nc, mdat):
    # new for v0.11: many of the ancillary engineering values are
    # not easily obtainiable. For now, I decided to just remove those,
    # and then see who complains about them missing.

    # these initial 5 vars should still be there.
    nc.createVariable('channel_wavelen1', float, ('channel', 'footprint'))
    nc.createVariable('channel_wavelen2', float, ('channel', 'footprint'))
    nc.createVariable('channel_wavenum1', float, ('channel', 'footprint'))
    nc.createVariable('channel_wavenum2', float, ('channel', 'footprint'))
    nc.createVariable('filter_number', np.int8, ('channel',))
    nc.createVariable('detector_mask', np.int8, ('channel', 'footprint'))
    nc.createVariable('NEDR', float, ('channel', 'footprint'))

    # these may not be
    if 'responsivity' in mdat:
        nc.createVariable('responsivity', float, ('channel', 'footprint'))
    if 'reflectivity' in mdat:
        nc.createVariable('reflectivity', float, ('channel',))

def _write_common_data(nc, mdat, invalid_channel, NEDR):
    # new for v0.11: many of the ancillary engineering values are
    # not easily obtainiable. For now, I decided to just remove those,
    # and then see who complains about them missing.

    # new for v0.10.4 - trim first channel if there are 64; that is
    # the undispersed channel.
    tmpwave = mdat['channel_wavelens'][:]
    filter_number = mdat['filter_channel'][0,:]
    if tmpwave.shape[0] == 64:
        tmpwave = tmpwave[1:,...]
        filter_number = filter_number[1:]

    nc['channel_wavelen1'][:] = tmpwave[...,0]
    nc['channel_wavelen2'][:] = tmpwave[...,1]
    nc['channel_wavenum1'][:] = 1e4/tmpwave[...,1]
    nc['channel_wavenum2'][:] = 1e4/tmpwave[...,0]

    nc['filter_number'][:] = filter_number
    nc['detector_mask'][:] = invalid_channel
    nc['NEDR'][:] = np.ma.masked_where(invalid_channel, NEDR)

    if 'TIRS' in mdat:
        R = mdat['TIRS']['R'][0,0]
        # the reflect model is not fp-dependent, so shaped (64,1).
        refl = mdat['TIRS']['Reflectivity'][0,0][:,0]
        if R.shape[0] == 64:
            R = R[1:,...]
            refl = refl[1:]
        nc['responsivity'][:] = R
        nc['reflectivity'][:] = refl

    # MATLAB is always two-D, and the loadmat has to use an extra
    # intermediate object array to deal with the MATLAB structures.
    # hence the seemingly redundant [0,0] index
    if 'TIRS' in mdat:
        for varname in ('bandwidth', 'fno', 'pix_size',
                        'slit_width', 'noise'):
            nc.createVariable(varname, float, dimensions=())
        nc['bandwidth'][0] = mdat['TIRS']['bandwidth'][0,0][0,0]
        nc['fno'][0] = mdat['TIRS']['fno'][0,0][0,0]
        nc['pix_size'][0] = mdat['TIRS']['d'][0,0][0,0]
        nc['slit_width'][0] = mdat['TIRS']['sw'][0,0][0,0]
        nc['noise'][0] = mdat['TIRS']['noise'][0,0][0,0]


def _write_common_attr(nc, mfile, mdat):

    nc['srf'].setncattr(
        'long_name', 'spectral response function')
    nc['srf'].setncattr(
        'description', 'spectral throughput for the TIRS channel')
    nc['srf_normed'].setncattr(
        'long_name', 'normalized spectral response function')
    nc['srf_normed'].setncattr(
        'description', 'SRF normalized to have a unit sum')

    nc['channel_wavelen1'].setncattr('units', 'micron')
    nc['channel_wavelen2'].setncattr('units', 'micron')
    nc['channel_wavenum1'].setncattr('units', '1/cm')
    nc['channel_wavenum2'].setncattr('units', '1/cm')

    nc['channel_wavelen1'].setncattr(
        'long_name', 'channel wavelength at detector (min)')
    nc['channel_wavelen2'].setncattr(
        'long_name', 'channel wavelength at detector (max)')

    nc['channel_wavenum1'].setncattr(
        'long_name', 'channel wavenumber at detector (min)')
    nc['channel_wavenum2'].setncattr(
        'long_name', 'channel wavenumber at detector (max)')

    nc['filter_number'].setncattr(
        'long_name', 'filter number at detector')
    nc['filter_number'].setncattr(
        'description', 'Order-sorting filter number for the channel')

    nc['detector_mask'].setncattr('long_name', 'detector mask')
    nc['detector_mask'].setncattr(
        'description', '0 = good, any other values should be '+
        'considered bad (TBD bit valies)')

    nc['NEDR'].setncattr(
        'long_name', 'Noise-Equivalent Delta Radiance')
    nc['NEDR'].setncattr(
        'description', '1-sigma radiance noise level')

    # a bunch of variables, no longer in v0.11 SRF dataset.
    # here just assume all are defined, or none are defined.
    if 'bandwidth' in nc.variables:
        nc['bandwidth'].setncattr(
            'long_name', 'Instrument total bandwidth')
        nc['bandwidth'].setncattr('units', 'micron')
    
        nc['fno'].setncattr(
            'long_name', 'Instrument F-number')

        nc['pix_size'].setncattr(
            'long_name', 'Instrument pixel size in spectral dimension')
        nc['pix_size'].setncattr('units', 'micron')

        nc['slit_width'].setncattr(
            'long_name', 'Instrument slit width')
        nc['slit_width'].setncattr('units', 'micron')

        nc['noise'].setncattr(
            'long_name', 'Detector noise level')
        nc['noise'].setncattr('units', 'nV/sqrt(Hz)')

        nc['responsivity'].setncattr(
            'long_name', 'Detector responsivity')
        nc['responsivity'].setncattr('units', 'V/W')

        nc['reflectivity'].setncattr(
            'long_name', 'Internal sensor optics total reflectivity')
        nc['reflectivity'].setncattr(
            'description', 'The total optical reflectivity represents '+
            'the optical throughput through the optical system')

    # global attributes
    md5_return = subprocess.check_output(['md5sum', mfile])
    md5sum = md5_return.decode().split()[0]
    nc.setncattr('comment2', 'Source file: ' + mfile)
    nc.setncattr('comment3', 'With md5sum: ' + md5sum)
    if 'git_short_hash' in mdat:
        nc.setncattr(
            'comment4', 'MATLAB code git hash: ' + mdat['git_short_hash'][0])

    git_short_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'])
    git_short_hash = git_short_hash.decode().strip()
    nc.setncattr(
        'comment5', 'process_srf.py git hash: ' + git_short_hash)
    nc.setncattr(
        'date_created', str(datetime.datetime.now()))


def process_srf(mfile, mfile_patch=None, outfile=None, maskfile=None):
    """
    main function to process the SRF data.
    Takes the MATLAB file as input, and creates a netCDF file with metadata,
    with a few value added fields (radiance as a function of blackbody
    temperature, to facilitate brightness temperature calculation;
    radiance derivative with respect to blackbody temperature)

    Also creates a per-wn version for direct use with PCRTM.

    Returns the file names to the two created netCDF4 files.

    New for v0.11: not all data is contained in the 'final' datafile
    from B. An additional 'patch' file is used that contains
    the extra needed fields (wavelens, filter_channel array.)
    See 'PREFIRE_SRF_model_v11_patch' for the creation of the patch file.
    additionally, use an optional outfile to set the output file name
    (since this linkage was also messed up).
    """

    # use PCRTM wn grid: every 0.5 wn.
    wn_start = 50.0
    wn_stop = 2760.0
    wn_delta = 0.5
    rtm_wngrid = np.arange(wn_start, wn_stop+wn_delta*0.1, wn_delta)

    mdat = loadmat(mfile)

    if mfile_patch:
        mdat_patch = loadmat(mfile_patch)
        mdat.update(mdat_patch)

    # nwlen is the number of high-res samples in the original matlab.
    # nwnum is the number of samples in the PCRTM wnum grid.
    NEDR_wl = mdat['NEDR']
    srf_wl = mdat['SRF']
    nwlen, nchannels, nfp = srf_wl.shape
    nwnum = rtm_wngrid.shape[0]

    # new for v0.10.4 - if the number of channels in the matlab file is 64,
    # then remove the leading index. This is the 'undispersed' channel,
    # which we have been labeling 'channel 0' and ignoring in the past.
    if nchannels == 64:
        nchannels = 63
        srf_wl = srf_wl[:,1:,:]
        NEDR_wl = NEDR_wl[1:,:]

    srf_wl_normed = srf_wl / srf_wl.sum(0)[np.newaxis,...]

    srf_wlgrid = mdat['wavelen'][:,0]
    wl_delta = srf_wlgrid[1] - srf_wlgrid[0]
    srf_wngrid = 1e4/srf_wlgrid[::-1]

    srf_wn = np.zeros((nwnum, nchannels, nfp))
    srf_wn_normed = np.zeros((nwnum, nchannels, nfp))

    NEDR_wn = np.zeros_like(NEDR_wl)

    invalid_channel = np.zeros((nchannels, nfp), bool)

    for c, fp in np.ndindex((nchannels, nfp)):

        srf_wn_c = np.interp(rtm_wngrid, srf_wngrid, srf_wl[::-1,c,fp])
        srf_wn_norm = srf_wn_c.sum()
        srf_wn[:,c,fp] = srf_wn_c
        if srf_wn_norm > 0:
            srf_wn_normed[:,c,fp] = srf_wn_c / srf_wn_norm
            # conversion of NEDR in wl grid, to wn grid.
            srf_wn_c_integ = np.trapz(srf_wl[::-1,c,fp], srf_wngrid)
            srf_wl_c_integ = np.trapz(srf_wl[:,c,fp], srf_wlgrid)
            NEDR_wn[c,fp] = NEDR_wl[c,fp] * srf_wl_c_integ / srf_wn_c_integ
        else:
            invalid_channel[c,fp] = True

    _, mfile_only = os.path.split(mfile)

    # write direct copy of MATLAB contents on WL grid
    if outfile is None:
        outfile_wl = os.path.join(
            '../data', mfile_only.replace('.mat', '.nc'))
    else:
        outfile_wl = outfile

    with netCDF4.Dataset(outfile_wl, 'w') as nc:

        nc.createDimension('spectrum', nwlen)
        nc.createDimension('channel', nchannels)
        nc.createDimension('footprint', nfp)
        
        nc.createVariable('wavelen', float, ('spectrum'))
        nc.createVariable('srf', float,
                          ('spectrum', 'channel', 'footprint'))
        nc.createVariable('srf_normed', float,
                          ('spectrum', 'channel', 'footprint'))

        _create_common_variables(nc, mdat)

        nc['wavelen'][:] = srf_wlgrid
        nc['srf'][:] = srf_wl
        nc['srf_normed'][:] = srf_wl_normed

        _write_common_data(nc, mdat, invalid_channel, NEDR_wl)

        nc['NEDR'].setncattr('units', 'W/m^2/sr/um')

        nc.setncattr(
            'comment1', 'TIRS Spectral Response Functions (SRF) from '+
            'MATLAB model then converted to netCDF4 with process_srf.py')

        _write_common_attr(nc, mfile, mdat)

        # new for V11: overwrite the mask (which before contained location
        # for masked channels only) with new version.
        if maskfile:
            with netCDF4.Dataset(maskfile, 'r') as masknc:
                nc['detector_mask'][:] = masknc['detector_mask'][:]
                nc['detector_mask'].setncattr(
                    'long_name', masknc['detector_mask'].long_name)
                nc['detector_mask'].setncattr(
                    'description', masknc['detector_mask'].description)
                    
                
    # write PCRTM verison
    if outfile is None:
        outfile_wn = os.path.join(
            '../data', mfile_only.replace('.mat', '_PCRTM_grid.nc'))
    else:
        outfile_wn = outfile.replace('.nc', '_PCRTM_grid.nc')

    with netCDF4.Dataset(outfile_wn, 'w') as nc:

        nc.createDimension('spectrum', nwnum)
        nc.createDimension('channel', nchannels)
        nc.createDimension('footprint', nfp)

        nc.createVariable('wavenum', float, ('spectrum'))
        nc.createVariable('srf', float,
                          ('spectrum', 'channel', 'footprint'))
        nc.createVariable('srf_normed', float,
                          ('spectrum', 'channel', 'footprint'))

        _create_common_variables(nc, mdat)

        nc['wavenum'][:] = rtm_wngrid
        nc['srf'][:] = srf_wn
        nc['srf_normed'][:] = srf_wn_normed

        _write_common_data(nc, mdat, invalid_channel, NEDR_wn)

        # new for V11: overwrite the mask (which before contained location
        # for masked channels only) with new version.
        if maskfile:
            with netCDF4.Dataset(maskfile, 'r') as masknc:
                nc['detector_mask'][:] = masknc['detector_mask'][:]
                nc['detector_mask'].setncattr(
                    'long_name', masknc['detector_mask'].long_name)
                nc['detector_mask'].setncattr(
                    'description', masknc['detector_mask'].description)

        nc['NEDR'].setncattr('units', 'mW/m^2/sr/cm^-1')

        nc.setncattr(
            'comment1', 'TIRS Spectral Response Functions (SRF) from '+
            'MATLAB model then converted to PCRTM grid and netCDF4 with '+
            'process_srf.py')

        _write_common_attr(nc, mfile, mdat)


    # add BT tables
    dat_wl = create_btemp_tables(outfile_wl)
    dat_wn = create_btemp_tables(outfile_wn)
    
    with netCDF4.Dataset(outfile_wl, 'a') as nc:

        nc.createDimension('temperature', dat_wl['T'].shape[0])
        nc.createVariable('T_grid', float, ('temperature',))
        nc.createVariable('rad', float, ('temperature', 'channel', 'footprint'))
        nc.createVariable('drad_dT', float, ('temperature', 'channel', 'footprint'))

        nc['T_grid'][:] = dat_wl['T']
        nc['rad'][:] = dat_wl['rad']
        nc['drad_dT'][:] = dat_wl['drad_dT']

        nc['T_grid'].setncattr('units', 'K')
        nc['rad'].setncattr('units', 'W / (m^2 sr um)')
        nc['drad_dT'].setncattr('units', 'W / (m^2 sr um K)')

        nc['T_grid'].setncattr('long_name', 'blackbody temperature')
        nc['rad'].setncattr('long_name', 'blackbody radiance')
        nc['drad_dT'].setncattr(
            'long_name',
            'derivative of blackbody radiance with respect to temperature')

    with netCDF4.Dataset(outfile_wn, 'a') as nc:

        nc.createDimension('temperature', dat_wl['T'].shape[0])
        nc.createVariable('T_grid', float, ('temperature',))
        nc.createVariable('rad', float, ('temperature', 'channel', 'footprint'))
        nc.createVariable('drad_dT', float, ('temperature', 'channel', 'footprint'))

        nc['T_grid'][:] = dat_wn['T']
        nc['rad'][:] = dat_wn['rad']
        nc['drad_dT'][:] = dat_wn['drad_dT']

        nc['T_grid'].setncattr('units', 'K')
        nc['rad'].setncattr('units', 'mW / (m^2 sr cm^-1)')
        nc['drad_dT'].setncattr('units', 'mW / (m^2 sr cm^-1 K)')

        nc['T_grid'].setncattr('long_name', 'blackbody temperature')
        nc['rad'].setncattr('long_name', 'blackbody radiance')
        nc['drad_dT'].setncattr(
            'long_name',
            'derivative of blackbody radiance with respect to temperature')

    return outfile_wl, outfile_wn


if __name__ == "__main__":

    import sys

    # kludginess to deal with the split data needed for v0.11.
    # main SRF matfile is lacking the wavelengths, so we need to read
    # those from a 'patch' data file.
    # new detector mask, is loaded from a file created via
    # the 'assess_pooled_NEDR' function (see above).
    #
    # so, we need 3 input files:
    # srf_matfile  wavelength_matfile  mask_ncfile
    #
    # syntax should now be, something like:
    # data_dir=~/projects/PREFIRE_L1/dist/ancillary/instrument_model/
    # python process_srf.py \
    #      ${data_dir}/instrument_1_NETD_NEDR_SRF.mat \
    #      ${data_dir}/PREFIRE_SRF_v0.11_inst1_wavelengths_2022-12-01.mat \
    #      ../data/PREFIRE_TIRS1_SRF_v0.11_FPA_mask.nc \
    #      ../data/PREFIRE_TIRS1_SRF_v0.11_2022-12-01.nc

    if len(sys.argv) != 5:
        raise ValueError('wrong number of inputs')

    mfile = sys.argv[1]
    mfile_patch = sys.argv[2]
    maskfile = sys.argv[3]
    outfile = sys.argv[4]

    SRF_file, _ = process_srf(mfile,
                              mfile_patch=mfile_patch,
                              outfile=outfile,
                              maskfile=maskfile)
    compute_compound_SRF(SRF_file)
